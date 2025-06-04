# View-Based Conditioning for InverseLDM

This document explains the new view-based conditioning feature that allows the InverseLDM model to condition generation on anatomical view type (sagittal, coronal, axial).

## Overview

The view conditioning system enables the model to generate brain slices that are appropriate for different anatomical views:
- **Sagittal**: Side view (emphasizes vertical structures)
- **Coronal**: Front view (emphasizes horizontal/depth structures)  
- **Axial**: Top view (emphasizes radial/circular structures)

## How It Works

### 1. View Detection
The system automatically detects the view type from the data path:
- `/path/to/sagittal/data` → sagittal view
- `/path/to/coronal/data` → coronal view
- `/path/to/axial/data` → axial view

### 2. Conditioning Tensor Generation
For each view type, the system creates distinct 3-channel conditioning tensors with:
- **Sagittal**: Vertical patterns and gradients
- **Coronal**: Horizontal patterns and gradients
- **Axial**: Radial and circular patterns

### 3. Model Architecture
- **Autoencoder**: Updated to handle 3-channel conditioning input (`embbeded_channels: 3`)
- **Diffusion**: Conditioner network processes 3-channel view conditions (`in_channels: 3`)

## Setup and Usage

### 1. Data Structure
Ensure your data is organized by view type:
```
/scratch_brain/acd23/code/2d_slices_dataset/vp_slices/
├── sagittal/
│   ├── vp_123456_sagittal_168.npy
│   └── ...
├── coronal/
│   ├── vp_123456_coronal_168.npy
│   └── ...
└── axial/
    ├── vp_123456_axial_168.npy
    └── ...
```

### 2. Configuration
Use the provided `view_conditioning_config.yml` as a template:

```yaml
data:
    condition:
        mode: view  # Enable view conditioning
    sampling:
        in_channels: 3  # 3 channels for view conditioning
        view_values:    # Optional: specify views for sampling
        - sagittal
        - coronal
        - axial

autoencoder:
    model:
        embbeded_channels: 3  # Handle 3-channel conditioning

diffusion:
    model:
        condition:
            in_channels: 3    # 3 channels for view conditioning
            out_channels: 3
            strength: 0.7     # Conditioning strength
```

### 3. Training

#### Option A: Single View Training
```bash
# Train on sagittal view
python transfer_train.py \
    --config view_conditioning_config.yml \
    --name sagittal_experiment \
    --gpu_ids [0,1,2,3,4,5,6,7] \
    --overwrite -y
```

#### Option B: Multi-View Training (Recommended)
Use the provided training script:

```bash
# Setup configs for all views
python train_multi_view.py \
    --config view_conditioning_config.yml \
    --name multi_view_experiment \
    --views sagittal coronal axial

# Train sequentially on all views
python train_multi_view.py \
    --config view_conditioning_config.yml \
    --name multi_view_experiment \
    --views sagittal coronal axial \
    --sequential \
    --overwrite
```

### 4. Sampling/Inference

The model will automatically generate samples for different views based on the conditioning:

```python
# The sampling dataset will cycle through view types
# generating samples conditioned on each view
```

## File Structure

### New Files
- `invldm/utils/view_condition.py` - View conditioning utilities
- `view_conditioning_config.yml` - Example configuration
- `train_multi_view.py` - Multi-view training script
- `VIEW_CONDITIONING_README.md` - This documentation

### Modified Files
- `invldm/datasets/brain2d_dataset.py` - Added view conditioning support
- `invldm/datasets/brain2dsampling_dataset.py` - Added view conditioning for sampling

## Technical Details

### Conditioning Patterns

**Sagittal View (Side view)**:
- Vertical sinusoidal patterns
- Vertical gradients with modulation
- Radial patterns for depth perception

**Coronal View (Front view)**:
- Horizontal sinusoidal patterns  
- Horizontal gradients with modulation
- Concentric circles for depth

**Axial View (Top view)**:
- Strong radial patterns from center
- Angular patterns (spoke-like)
- Concentric rings with spiral components

### Model Modifications

1. **Autoencoder**: 
   - `embbeded_channels: 3` to handle 3-channel conditioning
   - Concatenation mode for conditioning integration

2. **Diffusion Model**:
   - Conditioner network with `in_channels: 3`
   - Addition mode for conditioning integration
   - Adjustable conditioning strength

## Usage Examples

### Basic Training
```bash
# Copy and modify the view conditioning config
cp view_conditioning_config.yml my_view_config.yml

# Edit data_path to point to your view directory
# data_path: /your/path/to/sagittal  # or coronal/axial

# Train the model
python transfer_train.py \
    --config my_view_config.yml \
    --name my_view_experiment \
    --gpu_ids [0,1,2,3] \
    --overwrite -y
```

### Multi-View Training
```bash
# Setup and train on all views
python train_multi_view.py \
    --config view_conditioning_config.yml \
    --name comprehensive_view_study \
    --views sagittal coronal axial \
    --sequential \
    --gpu_ids [0,1,2,3,4,5,6,7] \
    --overwrite
```

### Custom View Selection
```bash
# Train only on specific views
python train_multi_view.py \
    --config view_conditioning_config.yml \
    --name axial_coronal_study \
    --views axial coronal \
    --sequential
```

## Monitoring and Results

- **Weights & Biases**: Experiments are logged with view-specific names
- **Log Directories**: Separate directories for each view (`view_conditioning_sagittal`, etc.)
- **Samples**: Generated samples will show view-specific characteristics

## Troubleshooting

### Common Issues

1. **Data Path Not Found**:
   ```
   WARNING: Data path /path/to/coronal does not exist for coronal view!
   ```
   **Solution**: Ensure all view directories exist and contain data

2. **Channel Mismatch**:
   ```
   Expected 1 channels but got 3
   ```
   **Solution**: Update `embbeded_channels: 3` in autoencoder config

3. **Conditioning Not Working**:
   - Check `condition.mode: view` is set
   - Verify `in_channels: 3` in diffusion conditioner
   - Ensure data paths contain view type names

### Debug Information

The system provides debug output showing:
- Detected view type from paths
- Conditioning tensor statistics
- Channel configurations

## Next Steps

1. **Train on Multiple Views**: Use the multi-view training script
2. **Experiment with Conditioning Strength**: Adjust `diffusion.model.condition.strength`
3. **Custom View Patterns**: Modify `view_condition.py` for domain-specific patterns
4. **Evaluation**: Compare generated samples across different views

## Integration with Existing Slice Conditioning

The view conditioning system works alongside the existing slice conditioning:
- **Slice conditioning**: Controls which slice number to generate
- **View conditioning**: Controls which anatomical view to generate
- Both can potentially be combined for more precise control 