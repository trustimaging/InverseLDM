import os
import wandb
import torch
import numpy as np
import matplotlib.pyplot as plt
from . import BaseLogger
import logging
from PIL import Image
import glob


class WandbLogger(BaseLogger):
    def __init__(self, args):
        super().__init__(args)
        
        logging.info("Initializing WandB logger...")
        
        # Get WandB configuration from environment variables
        api_key = os.environ.get('WANDB_API_KEY')
        project = os.environ.get('WANDB_PROJECT', 'conditioning')
        name = os.environ.get('WANDB_NAME', f"{args.name}_{args.run.run_name}")
        
        if not api_key:
            logging.error("WANDB_API_KEY not set! Please set it in your environment.")
            raise ValueError("WANDB_API_KEY environment variable is required")
        
        # Initialize WandB
        wandb.login(key=api_key)
        
        # Extract config from args
        config = self._extract_config(args)
        
        # Initialize WandB run
        self.run = wandb.init(
            project=project,
            name=name,
            config=config,
            dir=args.run.exp_folder
        )
        
        # Store paths for sample monitoring
        self.exp_folder = args.run.exp_folder
        self.samples_path = os.path.join(self.exp_folder, 'logs', 'diffusion', 'samples')
        
        # Create samples directory if it doesn't exist
        os.makedirs(self.samples_path, exist_ok=True)
        
        # Track last step we checked for images
        self.last_image_check_step = 0
        self.image_check_frequency = 50  # Check every 50 steps
        
        logging.info(f"WandB logger initialized successfully!")
        logging.info(f"  Project: {project}")
        logging.info(f"  Run name: {name}")
        logging.info(f"  Monitoring samples from: {self.samples_path}")
        logging.info(f"  WandB run URL: {self.run.url}")
            
        return None
    
    def _extract_config(self, args):
        """Extract configuration from args namespace"""
        config = {}
        for key, value in args.__dict__.items():
            if hasattr(value, '__dict__'):
                # Recursively extract nested namespaces
                config[key] = self._namespace_to_dict(value)
            else:
                config[key] = value
        return config
    
    def _namespace_to_dict(self, namespace):
        """Convert namespace to dictionary recursively"""
        result = {}
        for key, value in namespace.__dict__.items():
            if hasattr(value, '__dict__') and not isinstance(value, torch.nn.Module):
                result[key] = self._namespace_to_dict(value)
            else:
                result[key] = str(value) if not isinstance(value, (int, float, str, bool, list, dict, type(None))) else value
        return result
    
    def log_scalar(self, tag, val, step, **kwargs):
        """Log scalar values to WandB"""
        # Only log training and validation losses
        if "loss" in tag.lower():
            wandb.log({tag: val}, step=step)
            # Log to console periodically
            if step % 100 == 0:
                logging.info(f"WandB: Logged {tag}={val:.6f} at step {step}")
        
        # Periodically check for new sample images
        if step - self.last_image_check_step >= self.image_check_frequency:
            self._log_sample_images(step)
            self.last_image_check_step = step
        
        return None
    
    def log_figure(self, tag, fig, step, **kwargs):
        """Log matplotlib figures to WandB"""
        # Convert matplotlib figure to image
        wandb.log({tag: wandb.Image(fig)}, step=step)
        plt.close(fig)
        
        # Also check for new images in the samples directory
        self._log_sample_images(step)
        
        return None
    
    def log_hparams(self, hparam_dict, metric_dict, **kwargs):
        """Skip hyperparameter logging to avoid overhead"""
        return None
    
    def _log_sample_images(self, step):
        """Log all images from the samples directory"""
        if not os.path.exists(self.samples_path):
            return
        
        # Find all image files in the samples directory
        image_patterns = ['*.png', '*.jpg', '*.jpeg']
        image_files = []
        
        for pattern in image_patterns:
            pattern_files = glob.glob(os.path.join(self.samples_path, pattern))
            image_files.extend(pattern_files)
        
        # Also check subdirectories
        for subdir in ['training', 'validation', 'sampling']:
            subdir_path = os.path.join(self.samples_path, subdir)
            if os.path.exists(subdir_path):
                for pattern in image_patterns:
                    pattern_files = glob.glob(os.path.join(subdir_path, pattern))
                    image_files.extend(pattern_files)
        
        if not image_files:
            return
        
        # Sort by modification time to get the latest images
        image_files.sort(key=os.path.getmtime, reverse=True)
        
        # Log the most recent images (limit to avoid overwhelming the UI)
        max_images = 50  # Increased limit
        images_to_log = {}
        
        # Initialize logged images set if not exists
        if not hasattr(self, '_logged_images'):
            self._logged_images = set()
        
        new_images_count = 0
        for img_path in image_files[:max_images]:
            # Skip if we've already logged this exact file
            if img_path in self._logged_images:
                continue
                
            img_name = os.path.basename(img_path)
            relative_path = os.path.relpath(img_path, self.samples_path)
            
            try:
                # Load and log the image
                img = Image.open(img_path)
                
                # Create a more descriptive caption
                # Extract info from filename if possible
                caption_parts = [relative_path]
                if "epoch" in img_name:
                    caption_parts.append(f"(Step {step})")
                caption = " ".join(caption_parts)
                
                # Use relative path as key to organize better
                log_key = f"samples/{relative_path.replace(os.sep, '/')}"
                images_to_log[log_key] = wandb.Image(img, caption=caption)
                
                # Track that we've logged this image
                self._logged_images.add(img_path)
                new_images_count += 1
                
            except Exception as e:
                logging.warning(f"Failed to log image {img_path}: {e}")
        
        # Log all new images at once
        if images_to_log:
            wandb.log(images_to_log, step=step)
            logging.info(f"WandB: Logged {new_images_count} new sample images at step {step}")
    
    def __del__(self):
        """Cleanup when logger is destroyed"""
        if hasattr(self, 'run'):
            logging.info("Finishing WandB run...")
            self.run.finish() 