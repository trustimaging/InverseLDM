#!/bin/bash

# Transfer learning from unconditional to conditional models
# This script assumes you have pretrained models in /path/to/exps/test_no_conditioning/

# Set paths - UPDATE THESE TO YOUR ACTUAL PATHS
PRETRAINED_AUTOENCODER="/raid/dverschu/InverseLDM/exps/test_no_conditioning/logs/autoencoder/checkpoints/autoencoder_ckpt_latest.pth"
PRETRAINED_DIFFUSION="/raid/dverschu/InverseLDM/exps/test_no_conditioning/logs/diffusion/checkpoints/diffusion_ckpt_latest.pth"

# Update the config file with correct paths
echo "Updating config with pretrained model paths..."
sed -i "s|/raid/dverschu/InverseLDM/test_no_conditioning/logs/autoencoder/checkpoints/autoencoder_ckpt_latest.pth|${PRETRAINED_AUTOENCODER}|g" transfer_learning_config.yml

# Create a modified transfer_diffusion_runner.py with correct path
sed -i "s|/raid/dverschu/InverseLDM/exps/test_no_conditioning/logs/diffusion/checkpoints|$(dirname ${PRETRAINED_DIFFUSION})|g" invldm/runners/transfer_diffusion_runner.py

# Run the transfer learning
echo "Starting transfer learning..."
python /raid/dverschu/InverseLDM/transfer_train.py \
    --config /raid/dverschu/InverseLDM/transfer_learning_config.yml \
    --name transfer_conditioning \
    --logdir exps \
    --gpu_ids [0,1,2,3,4,5,6,7] \
    --overwrite -y

echo "Transfer learning complete!" 