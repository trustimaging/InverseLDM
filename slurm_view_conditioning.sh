#!/bin/bash -l

##############################
#       Job blueprint        #
##############################

# Give your job a name, so you can recognize it in the queue overview
#SBATCH --job-name=unified_view_conditioning
#SBATCH --partition=brain_main
#SBATCH --qos=normal
# Each job will utilise all of brainstorm's resources.
# Here, we ask for 1 node with exclusive access.
# brainstorm has 256 CPU cores and 8 A100 GPUs.

#SBATCH -e /scratch_brain/acd23/code/lfr/slurm_outputs/slurm-%j.err              # File to redirect stderr
#SBATCH -o /scratch_brain/acd23/code/lfr/slurm_outputs/slurm-%j.out              # File to redirect stdout
#SBATCH --mem=1900gb              # Total memory for the job (1500 MB * 256 CPUs)
#SBATCH --time=20:00:00              # The walltime
#SBATCH --nodes=1                    # Run all processes on a single node
#SBATCH --ntasks=2                   # Number of tasks
##SBATCH --ntasks-per-socket=1       # Maximum number of tasks on each socket
#SBATCH --cpus-per-task=128          # Total CPUs for the task
#SBATCH --gres=gpu:8                 # Number of GPUs
# This is where the actual work is done.

export TORCH_HOME="/scratch_brain/acd23/torch_cache"

# Option 2: Use a temporary directory
# export TORCH_HOME="/tmp/torch_cache_$(whoami)"

# Option 3: Use current working directory
# export TORCH_HOME="$(pwd)/torch_cache"

# Create the directory if it doesn't exist
mkdir -p "$TORCH_HOME"

echo "TORCH_HOME set to: $TORCH_HOME"
# activate conda environment
source /scratch_brain/acd23/miniconda3/etc/profile.d/conda.sh
conda activate stride

# Set up experiment details
export WANDB_API_KEY='e709e9c43e2fcded8dc2dfd834d685f1bcb46d85'
export WANDB_NAME="unified_view_conditioning"
export WANDB_PROJECT="view_conditioning"

# Base model checkpoints (should exist from previous training)
BASE_AE_CKPT="/scratch_brain/acd23/code/InverseLDM/exps/base_model_all_views/logs/autoencoder/checkpoints/autoencoder_ckpt_latest.pth"
BASE_DIFF_CKPT="/scratch_brain/acd23/code/InverseLDM/exps/base_model_all_views/logs/diffusion/checkpoints/diffusion_ckpt_latest.pth"

echo "Starting unified view conditioning training..."
echo "Experiment name: $WANDB_NAME"
echo "Base autoencoder: $BASE_AE_CKPT"
echo "Base diffusion: $BASE_DIFF_CKPT"

# Check if base model checkpoints exist
if [ ! -f "$BASE_AE_CKPT" ]; then
    echo "ERROR: Base autoencoder checkpoint not found at $BASE_AE_CKPT"
    echo "Please run the base model training first (slurm.sh)"
    exit 1
fi

if [ ! -f "$BASE_DIFF_CKPT" ]; then
    echo "ERROR: Base diffusion checkpoint not found at $BASE_DIFF_CKPT"
    echo "Please run the base model training first (slurm.sh)"
    exit 1
fi

echo "✅ Base model checkpoints found!"
echo ""

# Train unified view conditioning (loads from all view directories uniformly)
echo "Starting unified view conditioning training..."
echo "- Training on: sagittal, coronal, axial (uniform sampling)"
echo "- Single model handles all views"
echo "- Only training the conditioner, base models frozen"

python /scratch_brain/acd23/code/InverseLDM/transfer_train.py \
    --config /scratch_brain/acd23/code/InverseLDM/unified_view_conditioning_config.yml \
    --name $WANDB_NAME \
    --logdir exps \
    --gpu_ids [0,1,2,3,4,5,6,7] \
    --overwrite -y

echo "Unified view conditioning training complete!"
echo ""
echo "Results:"
echo "  Unified view model: exps/$WANDB_NAME/"
echo "  This single model can generate all 3 view types!"
echo ""
echo "Model checkpoints:"
echo "  Autoencoder: exps/$WANDB_NAME/logs/autoencoder/checkpoints/"
echo "  Diffusion: exps/$WANDB_NAME/logs/diffusion/checkpoints/"
echo ""
echo "Generated samples: exps/$WANDB_NAME/logs/diffusion/samples/" 