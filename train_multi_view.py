#!/usr/bin/env python3
"""
Multi-View Training Script for InverseLDM with View Conditioning

This script demonstrates how to train the InverseLDM model with view-based conditioning
on multiple anatomical views (sagittal, coronal, axial).

Usage:
    python train_multi_view.py --config view_conditioning_config.yml --name multi_view_experiment
"""

import os
import sys
import argparse
import yaml
import subprocess
from pathlib import Path

def update_config_for_view(config_path, view_type, output_config_path, base_model_path=None):
    """
    Update the config file to point to the specified view directory
    
    Args:
        config_path: Path to the base config file
        view_type: 'sagittal', 'coronal', or 'axial'
        output_config_path: Path to save the updated config
        base_model_path: Optional path to base model directory (for transfer learning)
    """
    # Load the base config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Update the data path for the specific view
    base_path = config['data']['data_path']
    # Replace the view directory
    if 'sagittal' in base_path:
        new_path = base_path.replace('sagittal', view_type)
    elif 'coronal' in base_path:
        new_path = base_path.replace('coronal', view_type)
    elif 'axial' in base_path:
        new_path = base_path.replace('axial', view_type)
    else:
        # Assume we need to append the view type
        new_path = os.path.join(os.path.dirname(base_path), view_type)
    
    config['data']['data_path'] = new_path
    
    # Update checkpoint paths if base model path is provided
    if base_model_path:
        base_ae_ckpt = os.path.join(base_model_path, "logs/autoencoder/checkpoints/autoencoder_ckpt_latest.pth")
        base_diff_ckpt = os.path.join(base_model_path, "logs/diffusion/checkpoints/diffusion_ckpt_latest.pth")
        
        config['autoencoder']['model']['checkpoint'] = base_ae_ckpt
        config['diffusion']['model']['checkpoint'] = base_diff_ckpt
        print(f"Using base autoencoder: {base_ae_ckpt}")
        print(f"Using base diffusion: {base_diff_ckpt}")
    
    # Update experiment paths to include view type
    base_log_path = config['autoencoder']['log_path']
    config['autoencoder']['log_path'] = base_log_path.replace('view_conditioning_transfer', f'view_conditioning_transfer_{view_type}')
    config['autoencoder']['ckpt_path'] = config['autoencoder']['ckpt_path'].replace('view_conditioning_transfer', f'view_conditioning_transfer_{view_type}')
    config['autoencoder']['recon_path'] = config['autoencoder']['recon_path'].replace('view_conditioning_transfer', f'view_conditioning_transfer_{view_type}')
    
    config['diffusion']['log_path'] = config['diffusion']['log_path'].replace('view_conditioning_transfer', f'view_conditioning_transfer_{view_type}')
    config['diffusion']['ckpt_path'] = config['diffusion']['ckpt_path'].replace('view_conditioning_transfer', f'view_conditioning_transfer_{view_type}')
    config['diffusion']['samples_path'] = config['diffusion']['samples_path'].replace('view_conditioning_transfer', f'view_conditioning_transfer_{view_type}')
    
    # Save the updated config
    with open(output_config_path, 'w') as f:
        yaml.safe_dump(config, f, default_flow_style=False)
    
    print(f"Created config for {view_type} view: {output_config_path}")
    print(f"Data path: {new_path}")

def train_view(config_path, view_type, experiment_name, gpu_ids, overwrite=False, base_model_path=None):
    """
    Train the model on a specific view
    
    Args:
        config_path: Path to the config file
        view_type: 'sagittal', 'coronal', or 'axial'
        experiment_name: Base name for the experiment
        gpu_ids: List of GPU IDs to use
        overwrite: Whether to overwrite existing experiments
        base_model_path: Optional path to base model directory
    """
    # Create view-specific config
    view_config_path = f"view_conditioning_transfer_{view_type}_config.yml"
    update_config_for_view(config_path, view_type, view_config_path, base_model_path)
    
    # Check if data directory exists
    with open(view_config_path, 'r') as f:
        config = yaml.safe_load(f)
    data_path = config['data']['data_path']
    
    if not os.path.exists(data_path):
        print(f"WARNING: Data path {data_path} does not exist for {view_type} view!")
        print(f"Please make sure you have {view_type} data available.")
        return False
    
    # Create training command
    view_experiment_name = f"{experiment_name}_{view_type}"
    cmd = [
        "python", "transfer_train.py",
        "--config", view_config_path,
        "--name", view_experiment_name,
        "--logdir", "exps",
        "--gpu_ids", str(gpu_ids)
    ]
    
    if overwrite:
        cmd.extend(["--overwrite", "-y"])
    
    print(f"\n{'='*60}")
    print(f"Training on {view_type.upper()} view")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")
    
    # Run training
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print(f"Successfully completed training for {view_type} view")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error training {view_type} view: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Train InverseLDM with view conditioning on multiple anatomical views")
    parser.add_argument("--config", required=True, help="Path to the base config file")
    parser.add_argument("--name", required=True, help="Base experiment name")
    parser.add_argument("--views", nargs="+", default=["sagittal", "coronal", "axial"], 
                       choices=["sagittal", "coronal", "axial"],
                       help="Views to train on")
    parser.add_argument("--gpu_ids", default="[0,1,2,3,4,5,6,7]", help="GPU IDs to use")
    parser.add_argument("--overwrite", "-y", action="store_true", help="Overwrite existing experiments")
    parser.add_argument("--sequential", action="store_true", help="Train views sequentially instead of showing commands")
    parser.add_argument("--base_model_path", help="Path to base model experiment directory (e.g., exps/base_model_all_views)")
    
    args = parser.parse_args()
    
    print("Multi-View Training Script for InverseLDM")
    print(f"Base config: {args.config}")
    print(f"Experiment name: {args.name}")
    print(f"Views to train: {args.views}")
    print(f"GPU IDs: {args.gpu_ids}")
    if args.base_model_path:
        print(f"Base model path: {args.base_model_path}")
    
    # Check if base config exists
    if not os.path.exists(args.config):
        print(f"Error: Config file {args.config} not found!")
        sys.exit(1)
    
    # Check if base model path exists (if provided)
    if args.base_model_path and not os.path.exists(args.base_model_path):
        print(f"Error: Base model path {args.base_model_path} not found!")
        sys.exit(1)
    
    successful_views = []
    failed_views = []
    
    for view in args.views:
        print(f"\n{'='*80}")
        print(f"PROCESSING {view.upper()} VIEW")
        print(f"{'='*80}")
        
        if args.sequential:
            # Actually run the training
            success = train_view(args.config, view, args.name, args.gpu_ids, args.overwrite, args.base_model_path)
            if success:
                successful_views.append(view)
            else:
                failed_views.append(view)
        else:
            # Just prepare the config and show the command
            view_config_path = f"view_conditioning_transfer_{view}_config.yml"
            update_config_for_view(args.config, view, view_config_path, args.base_model_path)
            
            # Show the command that would be run
            view_experiment_name = f"{args.name}_{view}"
            cmd = [
                "python", "transfer_train.py",
                "--config", view_config_path,
                "--name", view_experiment_name,
                "--logdir", "exps",
                "--gpu_ids", args.gpu_ids
            ]
            
            if args.overwrite:
                cmd.extend(["--overwrite", "-y"])
            
            print(f"To train {view} view, run:")
            print(f"  {' '.join(cmd)}")
    
    if args.sequential:
        print(f"\n{'='*80}")
        print("TRAINING SUMMARY")
        print(f"{'='*80}")
        print(f"Successful views: {successful_views}")
        if failed_views:
            print(f"Failed views: {failed_views}")
        print(f"Total: {len(successful_views)}/{len(args.views)} views completed successfully")
    else:
        print(f"\n{'='*80}")
        print("SETUP COMPLETE")
        print(f"{'='*80}")
        print(f"Created {len(args.views)} view-specific config files.")
        print("Run the commands above to train each view, or use --sequential to run automatically.")

if __name__ == "__main__":
    main() 