#!/bin/bash -l

##############################
#       Job blueprint        #
##############################

# Give your job a name, so you can recognize it in the queue overview
#SBATCH --job-name=presbrainaxial
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

# python /raid/dverschu/InverseLDM/train.py --config /raid/dverschu/InverseLDM/config.yml --name strong_conditioning_new --overwrite -y --gpu_ids [0,1,2,3,4,5,6,7] 

PRETRAINED_AUTOENCODER="/scratch_brain/acd23/code/InverseLDM/exps/test_no_conditioning/logs/autoencoder/checkpoints/autoencoder_ckpt_latest.pth"
PRETRAINED_DIFFUSION="/scratch_brain/acd23/code/InverseLDM/exps/test_no_conditioning/logs/diffusion/checkpoints/diffusion_ckpt_latest.pth"


export WANDB_API_KEY='e709e9c43e2fcded8dc2dfd834d685f1bcb46d85'
export WANDB_NAME="checking_good_config"
export WANDB_PROJECT="conditioning"

# Update the config file with correct paths
echo "Updating config with pretrained model paths..."
sed -i "s|/scratch_brain/acd23/code/InverseLDM/test_no_conditioning/logs/autoencoder/checkpoints/autoencoder_ckpt_latest.pth|${PRETRAINED_AUTOENCODER}|g" transfer_learning_config.yml

# Create a modified transfer_diffusion_runner.py with correct path
sed -i "s|/scratch_brain/acd23/code/InverseLDM/exps/test_no_conditioning/logs/diffusion/checkpoints|$(dirname ${PRETRAINED_DIFFUSION})|g" invldm/runners/transfer_diffusion_runner.py

# Run the transfer learning
echo "Starting transfer learning..."
python /scratch_brain/acd23/code/InverseLDM/transfer_train.py \
    --config /scratch_brain/acd23/code/InverseLDM/good_config.yml \
    --name $WANDB_NAME \
    --logdir exps \
    --gpu_ids [0,1,2,3,4,5,6,7] \
    --overwrite -y

echo "Transfer learning complete!"