#!/usr/bin/env python3
"""
Transfer learning script to fine-tune pretrained unconditional models with conditioning.
"""

from invldm.utils.setup import setup_train
from invldm.runners.trainer import Trainer
from invldm.runners.transfer_diffusion_runner import TransferDiffusionRunner
import logging
import torch
from torchsummary import summary

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
            
            # Get train_conditioner_only flag from config
            train_conditioner_only = getattr(args.diffusion, 'train_conditioner_only', False)
            
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
                train_conditioner_only=train_conditioner_only,  # Pass the flag
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
            
            # Log training mode
            if train_conditioner_only:
                logging.info("CONDITIONER-ONLY TRAINING MODE: The pretrained diffusion model is frozen.")
                logging.info("Only the conditioning network will be trained.")
    
    def train(self):
        """Override train method to add conditioner network summary"""
        logging.info(" ---- Dataset ---- ")
        logging.info(self.dataset)

        logging.info(" ---- Model - Autoencoder ----")
        sample = self.dataset[0]
        if isinstance(sample, tuple):
            sample, cond = sample
        else:
            cond = None
        sample = sample.to(self.autoencoder_runner.device)
        logging.info(summary(model=self.autoencoder_runner.model, input_size=sample.shape, device=self.autoencoder_runner.device))

        logging.info(" ---- Model - Diffusion ----")
        if self.args.diffusion.training.n_epochs > 0:
            with torch.no_grad():
                mu, sigma = self.diffusion_runner.autoencoder.encode(sample.unsqueeze(0).float())
                z = self.diffusion_runner.autoencoder.sampling(mu, sigma)
                t = torch.tensor([0]).repeat(z.shape[0])                

                if cond is not None:
                    c_input = cond.unsqueeze(0).float().to(z.device)
                    c = self.diffusion_runner.cond_proj(c_input)
                    if self.args.diffusion.model.condition.mode == "concat":
                        c = torch.nn.functional.interpolate(c, z.shape[2:])
                        z = torch.concat([z, c], dim=1)
                        input_data = (z, t)
                    elif self.args.diffusion.model.condition.mode == "crossattn":
                        c = c.flatten(start_dim=2)
                        input_data = (z, t, c)
                    else:
                        input_data = (z, t)
                else:
                    input_data = (z, t)

            # Main diffusion model summary
            logging.info(summary(model=self.diffusion_runner.model, input_size=input_data, device=self.diffusion_runner.device))
            logging.info(f"\n\nLatent size: {z.shape[1:]}")
            if "c" in locals():
                logging.info(f"\nCondition (latent) size: {c.shape[1:]}\n\n")
            
            # Conditioner network summary (if in conditioner-only mode)
            if hasattr(self.diffusion_runner, 'train_conditioner_only') and self.diffusion_runner.train_conditioner_only:
                logging.info(" ---- Conditioning Network Summary ----")
                if hasattr(self.diffusion_runner, 'cond_proj') and cond is not None:
                    # For spatial conditioner
                    cond_input_shape = cond.shape if len(cond.shape) > 1 else (1,)
                    try:
                        # Extract the actual conditioner module from Sequential wrapper
                        if isinstance(self.diffusion_runner.cond_proj, torch.nn.Sequential):
                            for module in self.diffusion_runner.cond_proj:
                                if hasattr(module, 'conv_in'):  # SpatialConditioner
                                    logging.info(summary(model=module, input_size=cond_input_shape, device=self.diffusion_runner.device))
                                    break
                                elif hasattr(module, 'embedding'):  # TransformerConditioner
                                    logging.info(summary(model=module, input_size=cond_input_shape, device=self.diffusion_runner.device))
                                    break
                        else:
                            logging.info(summary(model=self.diffusion_runner.cond_proj, input_size=cond_input_shape, device=self.diffusion_runner.device))
                    except Exception as e:
                        logging.warning(f"Could not generate conditioner summary: {e}")
                        # Fallback: manually count parameters
                        total_params = sum(p.numel() for p in self.diffusion_runner.cond_proj.parameters())
                        trainable_params = sum(p.numel() for p in self.diffusion_runner.cond_proj.parameters() if p.requires_grad)
                        logging.info(f"Conditioner Network: {trainable_params:,} trainable parameters out of {total_params:,} total")
                
                # Additional networks for slice conditioning
                if hasattr(self.diffusion_runner, 'slice_embedding'):
                    logging.info("\n ---- Slice Embedding Network ----")
                    slice_params = sum(p.numel() for p in self.diffusion_runner.slice_embedding.parameters())
                    logging.info(f"Slice Embedding: {slice_params:,} parameters")
                
                if hasattr(self.diffusion_runner, 'slice_to_spatial'):
                    logging.info("\n ---- Slice-to-Spatial Network ----")
                    spatial_params = sum(p.numel() for p in self.diffusion_runner.slice_to_spatial.parameters())
                    logging.info(f"Slice-to-Spatial: {spatial_params:,} parameters")
                
                # Summary of all trainable parameters
                logging.info("\n ---- Training Summary ----")
                all_trainable = []
                if hasattr(self.diffusion_runner, 'cond_proj'):
                    all_trainable.extend(p for p in self.diffusion_runner.cond_proj.parameters() if p.requires_grad)
                if hasattr(self.diffusion_runner, 'slice_embedding'):
                    all_trainable.extend(p for p in self.diffusion_runner.slice_embedding.parameters() if p.requires_grad)
                if hasattr(self.diffusion_runner, 'slice_to_spatial'):
                    all_trainable.extend(p for p in self.diffusion_runner.slice_to_spatial.parameters() if p.requires_grad)
                
                total_trainable = sum(p.numel() for p in all_trainable)
                total_model = sum(p.numel() for p in self.diffusion_runner.model.parameters())
                total_all = total_model + total_trainable
                
                logging.info(f"Main Diffusion Model: {total_model:,} parameters (FROZEN)")
                logging.info(f"Conditioning Networks: {total_trainable:,} parameters (TRAINABLE)")
                logging.info(f"Total: {total_all:,} parameters ({100.0 * total_trainable / total_all:.2f}% trainable)")
        
        logging.info(" ---- Autoencoder Training ---- ")
        self.autoencoder_runner.train()
        
        if "discriminator" in self.autoencoder_runner.__dict__:
            del self.autoencoder_runner.discriminator
        if "perceptual_loss_fn" in self.autoencoder_runner.__dict__:
            del self.autoencoder_runner.perceptual_loss_fn

        if self.args.diffusion.training.n_epochs > 0:
            logging.info(" ---- Diffusion Training ---- ")
            self.diffusion_runner.train()

        logging.info(" ---- Training Concluded without Errors ---- ")


if __name__ == "__main__":
    # Setup training configuration
    args = setup_train()
    
    # Set the pretrained checkpoint path
    # This path should point to your unconditional diffusion model checkpoint
    pretrained_diffusion_path = "/scratch_brain/dverschu/InverseLDM/exps/test_no_conditioning/logs/diffusion/checkpoints/diffusion_ckpt_latest.pth"
    
    # Add the pretrained_checkpoint attribute to the config
    if hasattr(args.diffusion, 'model'):
        args.diffusion.model.pretrained_checkpoint = pretrained_diffusion_path
    
    # Create trainer with transfer learning support
    trainer = TransferTrainer(args)
    
    # Train with transfer learning
    trainer.train() 