#!/usr/bin/env python3
"""
Complete View-Based Training Workflow for InverseLDM

This script orchestrates the complete training process:
1. Train a base model (AE + Diffusion) from scratch on ALL view data combined (unconditioned)
2. Train view conditioning on top of the base model for each view

Usage:
    python complete_view_training.py --base_name base_model --view_name view_conditioning
"""

import os
import sys
import argparse
import subprocess
import time

def run_command(cmd, description):
    """Run a command and handle errors"""
    print(f"\n{'='*80}")
    print(f"RUNNING: {description}")
    print(f"COMMAND: {' '.join(cmd)}")
    print(f"{'='*80}")
    
    try:
        result = subprocess.run(cmd, check=True)
        print(f"✅ SUCCESS: {description}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ FAILED: {description}")
        print(f"Error: {e}")
        return False

def check_base_model_exists(base_model_path):
    """Check if base model exists and has required checkpoints"""
    ae_ckpt = os.path.join(base_model_path, "logs/autoencoder/checkpoints/autoencoder_ckpt_latest.pth")
    diff_ckpt = os.path.join(base_model_path, "logs/diffusion/checkpoints/diffusion_ckpt_latest.pth")
    
    ae_exists = os.path.exists(ae_ckpt)
    diff_exists = os.path.exists(diff_ckpt)
    
    print(f"Base model status:")
    print(f"  Autoencoder: {'✅' if ae_exists else '❌'} {ae_ckpt}")
    print(f"  Diffusion: {'✅' if diff_exists else '❌'} {diff_ckpt}")
    
    return ae_exists and diff_exists

def train_base_model(base_name, gpu_ids, overwrite=False):
    """Train the base model on all views combined"""
    print(f"\n{'='*80}")
    print("PHASE 1: TRAINING BASE MODEL ON ALL VIEWS")
    print(f"{'='*80}")
    
    cmd = [
        "python", "train.py",
        "--config", "base_model_all_views_config.yml",
        "--name", base_name,
        "--logdir", "exps",
        "--gpu_ids", str(gpu_ids)
    ]
    
    if overwrite:
        cmd.extend(["--overwrite", "-y"])
    
    success = run_command(cmd, "Base Model Training")
    
    if success:
        base_model_path = f"exps/{base_name}"
        if check_base_model_exists(base_model_path):
            print(f"✅ Base model training completed successfully!")
            return base_model_path
        else:
            print(f"❌ Base model training completed but checkpoints not found!")
            return None
    else:
        return None

def train_view_conditioning(view_name, base_model_path, views, gpu_ids, overwrite=False, sequential=False):
    """Train view conditioning using the base model"""
    print(f"\n{'='*80}")
    print("PHASE 2: TRAINING VIEW CONDITIONING")
    print(f"{'='*80}")
    
    cmd = [
        "python", "train_multi_view.py",
        "--config", "view_conditioning_transfer_config.yml",
        "--name", view_name,
        "--base_model_path", base_model_path,
        "--views"] + views + [
        "--gpu_ids", str(gpu_ids)
    ]
    
    if overwrite:
        cmd.extend(["--overwrite", "-y"])
        
    if sequential:
        cmd.append("--sequential")
    
    return run_command(cmd, "View Conditioning Training")

def main():
    parser = argparse.ArgumentParser(description="Complete view-based training workflow for InverseLDM")
    parser.add_argument("--base_name", required=True, help="Name for base model experiment")
    parser.add_argument("--view_name", required=True, help="Name for view conditioning experiment")
    parser.add_argument("--views", nargs="+", default=["sagittal", "coronal", "axial"], 
                       choices=["sagittal", "coronal", "axial"],
                       help="Views to train conditioning on")
    parser.add_argument("--gpu_ids", default="[0,1,2,3,4,5,6,7]", help="GPU IDs to use")
    parser.add_argument("--overwrite", "-y", action="store_true", help="Overwrite existing experiments")
    parser.add_argument("--sequential", action="store_true", help="Train view conditioning sequentially")
    parser.add_argument("--skip_base", action="store_true", help="Skip base model training (use existing)")
    parser.add_argument("--base_model_path", help="Path to existing base model (if skipping base training)")
    parser.add_argument("--view_only", action="store_true", help="Only train view conditioning (requires --base_model_path)")
    
    args = parser.parse_args()
    
    print("🚀 Complete View-Based Training Workflow")
    print(f"Base model name: {args.base_name}")
    print(f"View conditioning name: {args.view_name}")
    print(f"Views to train: {args.views}")
    print(f"GPU IDs: {args.gpu_ids}")
    
    # Check required configs exist
    required_configs = ["base_model_all_views_config.yml", "view_conditioning_transfer_config.yml"]
    for config in required_configs:
        if not os.path.exists(config):
            print(f"❌ Error: Required config file {config} not found!")
            sys.exit(1)
    
    base_model_path = None
    
    # Phase 1: Base Model Training
    if args.view_only:
        if not args.base_model_path:
            print("❌ Error: --base_model_path required when using --view_only")
            sys.exit(1)
        base_model_path = args.base_model_path
        if not check_base_model_exists(base_model_path):
            print(f"❌ Error: Base model not found at {base_model_path}")
            sys.exit(1)
    elif args.skip_base:
        if args.base_model_path:
            base_model_path = args.base_model_path
        else:
            base_model_path = f"exps/{args.base_name}"
        
        if check_base_model_exists(base_model_path):
            print(f"✅ Using existing base model at {base_model_path}")
        else:
            print(f"❌ Error: Base model not found at {base_model_path}")
            print("Remove --skip_base to train from scratch, or provide --base_model_path")
            sys.exit(1)
    else:
        # Train base model from scratch
        base_model_path = train_base_model(args.base_name, args.gpu_ids, args.overwrite)
        if not base_model_path:
            print("❌ Base model training failed. Stopping.")
            sys.exit(1)
    
    # Phase 2: View Conditioning Training
    success = train_view_conditioning(
        args.view_name, 
        base_model_path, 
        args.views, 
        args.gpu_ids, 
        args.overwrite,
        args.sequential
    )
    
    if success:
        print(f"\n🎉 COMPLETE SUCCESS!")
        print(f"Base model: {base_model_path}")
        print(f"View conditioning: exps/{args.view_name}_<view>/")
        print(f"Views trained: {args.views}")
    else:
        print(f"\n❌ View conditioning training failed.")
        sys.exit(1)

if __name__ == "__main__":
    main() 