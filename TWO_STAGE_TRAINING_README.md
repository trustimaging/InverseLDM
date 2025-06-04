# Two-Stage Training for View-Based Conditioning

This document explains the complete two-stage training approach for view-based brain slice generation.

## Overview

The training is split into two stages:

1. **Stage 1: Base Model Training** - Train autoencoder + diffusion on ALL views combined (unconditioned)
2. **Stage 2: View Conditioning** - Add view conditioning on top of base model (ONE unified model)

## Stage 1: Base Model Training

### Purpose
- Train a solid foundation model on ALL available data
- No conditioning - learns general brain slice generation
- Uses data from sagittal + coronal + axial combined

### Configuration
- **Config**: `base_model_all_views_config.yml`
- **Data path**: `/scratch_brain/acd23/code/2d_slices_dataset/vp_slices` (parent directory)
- **Conditioning**: `mode: null` (no conditioning)
- **Training**: Both autoencoder (10 epochs) and diffusion (5 epochs) from scratch

### SLURM Script
```bash
# Run base model training
sbatch slurm.sh
```

### Expected Output
```
exps/base_model_all_views/
├── logs/
│   ├── autoencoder/checkpoints/autoencoder_ckpt_latest.pth
│   └── diffusion/checkpoints/diffusion_ckpt_latest.pth
```

---

## Stage 2: Unified View Conditioning

### Purpose
- Add view conditioning to the base model
- **Single unified model** that can generate all 3 views
- Trains on all views simultaneously with uniform sampling
- Only trains the conditioner, base models stay frozen

### Key Features
- ✅ **Uniform Sampling**: Loads from sagittal, coronal, axial directories equally
- ✅ **Single Model**: ONE model handles all view types
- ✅ **Transfer Learning**: Uses base model checkpoints as starting point
- ✅ **Efficient**: Only trains the conditioning network

### Configuration
- **Config**: `unified_view_conditioning_config.yml`
- **Data path**: `/scratch_brain/acd23/code/2d_slices_dataset/vp_slices` (parent directory)
- **Conditioning**: `mode: view` (3-channel view conditioning)
- **Training**: Only conditioner (10 epochs), base models frozen

### SLURM Script
```bash
# Run view conditioning training (after base model is complete)
sbatch slurm_view_conditioning.sh
```

### Expected Output
```
exps/unified_view_conditioning/
├── logs/
│   ├── autoencoder/checkpoints/  # Updated with conditioning
│   └── diffusion/checkpoints/    # Updated with conditioning
└── samples/                      # Generated samples for all views
```

---

## Training Workflow

### Option 1: Sequential (Recommended)
```bash
# Step 1: Train base model
sbatch slurm.sh

# Wait for completion, then...

# Step 2: Train view conditioning
sbatch slurm_view_conditioning.sh
```

### Option 2: Manual
```bash
# Step 1: Base model
python train.py \
    --config base_model_all_views_config.yml \
    --name base_model_all_views \
    --gpu_ids [0,1,2,3,4,5,6,7] \
    --overwrite -y

# Step 2: View conditioning
python transfer_train.py \
    --config unified_view_conditioning_config.yml \
    --name unified_view_conditioning \
    --gpu_ids [0,1,2,3,4,5,6,7] \
    --overwrite -y
```

---

## Technical Details

### Data Loading Strategy

**Stage 1 (Base Model)**:
- Loads from all view subdirectories: `sagittal/`, `coronal/`, `axial/`
- No view information provided to model
- Learns general brain slice patterns

**Stage 2 (View Conditioning)**:
- Loads from all view subdirectories uniformly
- Extracts view type from file path: `/path/sagittal/file.npy` → 'sagittal'
- Creates view-specific conditioning tensors:
  - **Sagittal**: Vertical patterns
  - **Coronal**: Horizontal patterns  
  - **Axial**: Radial patterns

### Model Architecture

**Base Model**:
```yaml
autoencoder:
  embbeded_channels: 1          # No conditioning
  condition: { mode: null }
diffusion:
  condition: { mode: null }     # No conditioning
```

**View Conditioning Model**:
```yaml
autoencoder:
  embbeded_channels: 3          # 3-channel view conditioning
  condition: { mode: concat }
diffusion:
  condition: 
    mode: addition
    in_channels: 3              # 3-channel view conditioning
    strength: 0.7
```

### Uniform Sampling

The dataset automatically provides uniform sampling across views:

1. **Loading**: Discovers all `.npy` files in `sagittal/`, `coronal/`, `axial/`
2. **Shuffling**: PyTorch DataLoader shuffles all files together
3. **Conditioning**: Each file gets appropriate view conditioning based on its path
4. **Result**: Model sees roughly equal amounts of each view type

---

## Advantages of This Approach

### ✅ Better Base Model
- Trained on MORE data (all views combined)
- Stronger foundation for conditioning
- Better generalization

### ✅ Efficient Training
- Stage 2 only trains conditioner (~5% of model parameters)
- Fast conditioning training
- Can experiment with different conditioning strengths

### ✅ Single Unified Model
- One model handles all views
- No need to switch between models
- Consistent quality across views

### ✅ Uniform View Representation
- Equal training on all view types
- Balanced conditioning learning
- Fair treatment of all anatomical perspectives

---

## Monitoring and Results

### Weights & Biases
- **Stage 1**: Project `view_conditioning`, Run `base_model_all_views`
- **Stage 2**: Project `view_conditioning`, Run `unified_view_conditioning`

### Generated Samples
View conditioning model automatically generates samples for all views:
- `exps/unified_view_conditioning/logs/diffusion/samples/`
- Samples include sagittal, coronal, and axial examples

### Usage After Training
```python
# The trained model can generate any view type by providing appropriate conditioning
# This is handled automatically during sampling based on the view_values config
```

---

## Troubleshooting

### Stage 1 Issues
```bash
# Check base model exists
ls exps/base_model_all_views/logs/*/checkpoints/
```

### Stage 2 Issues
```bash
# The script will check for base model automatically
# If base model missing, it will show clear error message
```

### View Detection Debug
```bash
# Check dataset debug output for view detection
# Should show: "DEBUG-VIEW: Extracted view type: sagittal/coronal/axial"
```

---

## Next Steps

1. ✅ Train base model: `sbatch slurm.sh`
2. ✅ Train view conditioning: `sbatch slurm_view_conditioning.sh` 
3. 🎯 Evaluate results across different views
4. 🎯 Experiment with conditioning strength
5. 🎯 Generate view-specific samples for analysis

This approach gives you the best of both worlds: a strong base model trained on all available data, plus efficient view-specific conditioning that works uniformly across all anatomical views! 