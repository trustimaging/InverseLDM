#!/usr/bin/env python3
"""
Transfer learning script to fine-tune pretrained unconditional models with conditioning.
"""

from invldm.utils.setup import setup_train
from invldm.runners.trainer import Trainer
from invldm.runners.transfer_diffusion_runner import TransferDiffusionRunner
import logging

class TransferTrainer(Trainer):
    """Modified trainer that uses TransferDiffusionRunner for transfer learning"""
    
    def __init__(self, args):
        # Call parent init but we'll override the diffusion runner
        super().__init__(args)
        
        # Replace the diffusion runner with our transfer learning version
        if self.args.diffusion.training.n_epochs > 0:
            # Delete the original runner
            if hasattr(self, 'diffusion_runner'):
                del self.diffusion_runner
            
            # Create transfer learning runner
            self.diffusion_runner = TransferDiffusionRunner(
                args=args.diffusion,
                args_run=args.run,
                args_logging=args.logging,
                autoencoder=self.autoencoder_runner.model,
                spatial_dims=args.autoencoder.model.spatial_dims,
                latent_channels=args.autoencoder.model.latent_channels,
                train_loader=self.diffusion_train_dataloader,
                valid_loader=self.diffusion_valid_dataloader,
            )
            
            # Load pretrained weights
            pretrained_path = getattr(args.diffusion.model, 'pretrained_checkpoint', None)
            if pretrained_path:
                logging.info(f"Loading pretrained diffusion model from: {pretrained_path}")
                self.diffusion_runner.load_checkpoint(pretrained_path, model_only=True)
            else:
                # Try to auto-find pretrained checkpoint
                logging.info("Attempting to load pretrained diffusion model...")
                self.diffusion_runner.load_checkpoint(model_only=True)


if __name__ == "__main__":
    # Setup training configuration
    args = setup_train()
    
    # Set the pretrained checkpoint path
    # This path should point to your unconditional diffusion model checkpoint
    pretrained_diffusion_path = "/raid/dverschu/InverseLDM/exps/test_no_conditioning/logs/diffusion/checkpoints/diffusion_ckpt_latest.pth"
    
    # Add the pretrained_checkpoint attribute to the config
    if hasattr(args.diffusion, 'model'):
        args.diffusion.model.pretrained_checkpoint = pretrained_diffusion_path
    
    # Create trainer with transfer learning support
    trainer = TransferTrainer(args)
    
    # Train with transfer learning
    trainer.train() 