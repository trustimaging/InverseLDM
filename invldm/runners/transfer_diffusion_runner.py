import torch
import logging
from .diffusion_runner import DiffusionRunner

class TransferDiffusionRunner(DiffusionRunner):
    """
    DiffusionRunner that can load pretrained unconditional model weights
    into a conditional model architecture.
    """
    
    def __init__(self, **kwargs):
        """
        Initialize with option to freeze main model and only train conditioner.
        """
        # Check if we should only train the conditioner
        self.train_conditioner_only = kwargs.get('train_conditioner_only', False)
        
        # Initialize parent class
        super().__init__(**kwargs)
        
        # If training conditioner only, freeze the main model and update optimizer
        if self.train_conditioner_only and hasattr(self, 'cond_proj'):
            self._setup_conditioner_only_training()
    
    def _setup_conditioner_only_training(self):
        """
        Freeze the main diffusion model and setup optimizer to only train conditioner.
        """
        logging.info("Setting up conditioner-only training mode")
        
        # Freeze main diffusion model parameters
        for param in self.model.parameters():
            param.requires_grad = False
        logging.info("Frozen main diffusion model parameters")
        
        # Unfreeze conditioner parameters
        trainable_params = []
        
        # Conditioner network parameters
        if hasattr(self, 'cond_proj'):
            for param in self.cond_proj.parameters():
                param.requires_grad = True
                trainable_params.append(param)
            logging.info(f"Enabled gradients for {sum(p.numel() for p in self.cond_proj.parameters())} conditioner parameters")
        
        # Slice embedding parameters if using slice conditioning
        if hasattr(self, 'slice_embedding'):
            for param in self.slice_embedding.parameters():
                param.requires_grad = True
                trainable_params.append(param)
            logging.info(f"Enabled gradients for {sum(p.numel() for p in self.slice_embedding.parameters())} slice embedding parameters")
        
        # Slice to spatial conversion parameters
        if hasattr(self, 'slice_to_spatial'):
            for param in self.slice_to_spatial.parameters():
                param.requires_grad = True
                trainable_params.append(param)
            logging.info(f"Enabled gradients for {sum(p.numel() for p in self.slice_to_spatial.parameters())} slice-to-spatial parameters")
        
        # Count total trainable parameters
        total_trainable = sum(p.numel() for p in trainable_params)
        total_params = sum(p.numel() for p in self.model.parameters()) + total_trainable
        logging.info(f"Total trainable parameters: {total_trainable:,} / {total_params:,} ({100.0 * total_trainable / total_params:.2f}%)")
        
        # Recreate optimizer with only trainable parameters
        if hasattr(self, 'optimiser') and trainable_params:
            # Get optimizer settings from config
            optim_kwargs = {
                'lr': self.args.optim.lr,
                'weight_decay': self.args.optim.weight_decay,
                'betas': self.args.optim.betas,
                'eps': self.args.optim.eps,
                'amsgrad': self.args.optim.amsgrad
            }
            
            # Create new optimizer with only trainable parameters
            if self.args.optim.optimiser.lower() == "adam":
                self.optimiser = torch.optim.Adam(trainable_params, **optim_kwargs)
            elif self.args.optim.optimiser.lower() == "adamw":
                self.optimiser = torch.optim.AdamW(trainable_params, **optim_kwargs)
            else:
                self.optimiser = torch.optim.Adam(trainable_params, **optim_kwargs)
            
            logging.info(f"Created new {self.args.optim.optimiser} optimizer for conditioner-only training")
    
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
                pretrained_path = "/raid/dverschu/InverseLDM/exps/test_no_conditioning/logs/diffusion/checkpoints"
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
        
        # Re-setup conditioner-only training if needed (in case optimizer was loaded)
        if self.train_conditioner_only and hasattr(self, 'cond_proj'):
            self._setup_conditioner_only_training()
        
        return True 