import torch
import logging
from .diffusion_runner import DiffusionRunner

class TransferDiffusionRunner(DiffusionRunner):
    """
    DiffusionRunner that can load pretrained unconditional model weights
    into a conditional model architecture.
    """
    
    def load_checkpoint(self, path=None, model_only=False):
        """
        Override to handle architecture mismatch when loading unconditional
        model into conditional architecture.
        """
        # Get the checkpoint path
        if not path:
            # Try to find latest checkpoint
            import os
            import fnmatch
            try:
                latest_ckpt_name = [f for f in os.listdir(self.args.ckpt_path) 
                                   if fnmatch.fnmatch(f, "*latest*")][0]
                path = os.path.join(self.args.ckpt_path, latest_ckpt_name)
            except (IndexError, FileNotFoundError):
                # Try the pretrained path
                pretrained_path = "/path/to/your/exps/test_no_conditioning/logs/diffusion/checkpoints"
                try:
                    latest_ckpt_name = [f for f in os.listdir(pretrained_path) 
                                       if fnmatch.fnmatch(f, "*latest*")][0]
                    path = os.path.join(pretrained_path, latest_ckpt_name)
                    logging.info(f"Found pretrained checkpoint: {path}")
                except (IndexError, FileNotFoundError):
                    logging.error("Could not find pretrained diffusion checkpoint")
                    return None
        
        logging.info(f"Loading diffusion checkpoint from {path}")
        
        # Load the checkpoint
        try:
            checkpoint = torch.load(path, map_location=self.device)
        except RuntimeError:
            checkpoint = torch.load(path, map_location="cpu")
        
        # Get current model state dict
        current_state = self.model.state_dict()
        pretrained_state = checkpoint["model_state_dict"]
        
        # Filter out mismatched keys
        filtered_state = {}
        mismatched_keys = []
        
        for key, value in pretrained_state.items():
            if key in current_state:
                if current_state[key].shape == value.shape:
                    filtered_state[key] = value
                else:
                    mismatched_keys.append((key, pretrained_state[key].shape, current_state[key].shape))
                    # Handle special cases
                    if "conv_in" in key and self.args.model.condition.mode == "concat":
                        # For concat mode, conv_in has more input channels
                        # Initialize the new channels randomly while keeping pretrained weights
                        old_channels = value.shape[1]
                        new_channels = current_state[key].shape[1]
                        
                        # Create new tensor with proper shape
                        new_weight = torch.randn_like(current_state[key]) * 0.01
                        # Copy pretrained weights to first channels
                        new_weight[:, :old_channels] = value
                        filtered_state[key] = new_weight
                        logging.info(f"Adapted {key} from {value.shape} to {current_state[key].shape}")
            else:
                # Key not in current model (probably attention layers for conditioning)
                logging.debug(f"Skipping key not in current model: {key}")
        
        # Initialize new parameters (conditioning-related) with small random values
        for key in current_state:
            if key not in filtered_state:
                if "cond" in key or "cross_attn" in key or "context" in key:
                    # These are new conditioning-related parameters
                    logging.info(f"Randomly initializing new conditioning parameter: {key}")
                else:
                    logging.warning(f"Missing key in pretrained model: {key}")
        
        # Log mismatched keys
        if mismatched_keys:
            logging.info("Shape mismatches (handled):")
            for key, old_shape, new_shape in mismatched_keys:
                logging.info(f"  {key}: {old_shape} -> {new_shape}")
        
        # Load the filtered state dict
        self.model.load_state_dict(filtered_state, strict=False)
        logging.info(f"Loaded {len(filtered_state)}/{len(current_state)} parameters from pretrained model")
        
        # Load training state if not model_only
        if not model_only:
            # Reset epoch and step counters for fine-tuning
            self.epoch = 1
            self.steps = 1
            logging.info("Reset training counters for fine-tuning")
            
            # Optionally load optimizer state (usually better to start fresh for transfer learning)
            # self.optimiser.load_state_dict(checkpoint["optimiser_state_dict"])
        
        return True 