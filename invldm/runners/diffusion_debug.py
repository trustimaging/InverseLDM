import os
import torch
import torch.nn as nn
import logging
import numpy as np

from .diffusion_runner import DiffusionRunner

class DebugDiffusionRunner(DiffusionRunner):
    """
    A debugging version of the DiffusionRunner that logs intermediate values
    to help identify the source of NaN issues.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.step_counter = 0
        # Create debug log directory
        os.makedirs("debug_logs", exist_ok=True)
        # Set up debug logger
        self.debug_log = open("debug_logs/nan_debug.log", "w")
        self.debug_log.write("Starting debug session\n")
        self.debug_log.flush()
        
    def _log_tensor_stats(self, tensor, name):
        """
        Logs statistics of a tensor to help debug NaN values.
        """
        if tensor is None:
            self.debug_log.write(f"{name} is None\n")
            return
            
        try:
            is_nan = torch.isnan(tensor).any().item()
            has_inf = torch.isinf(tensor).any().item()
            if torch.is_tensor(tensor) and tensor.numel() > 0:
                stats = {
                    "shape": str(tensor.shape),
                    "dtype": str(tensor.dtype),
                    "min": tensor.min().item() if not is_nan and not has_inf else "N/A",
                    "max": tensor.max().item() if not is_nan and not has_inf else "N/A",
                    "mean": tensor.mean().item() if not is_nan and not has_inf else "N/A",
                    "std": tensor.std().item() if not is_nan and not has_inf else "N/A",
                    "has_nan": is_nan,
                    "has_inf": has_inf,
                    "requires_grad": tensor.requires_grad if hasattr(tensor, 'requires_grad') else "N/A",
                    "has_grad_fn": tensor.grad_fn is not None if hasattr(tensor, 'grad_fn') else "N/A"
                }
                self.debug_log.write(f"{name}: {stats}\n")
            else:
                self.debug_log.write(f"{name}: Empty tensor or not a tensor\n")
        except Exception as e:
            self.debug_log.write(f"Error analyzing {name}: {str(e)}\n")
        self.debug_log.flush()

    def train_step(self, input, **kwargs):
        """
        Override train_step to add debugging logs
        """
        self.step_counter += 1
        self.debug_log.write(f"\n--- Step {self.step_counter} ---\n")
        
        # Dictionary of outputs
        output = {}
        
        # Log input stats
        self._log_tensor_stats(input, "input")
        
        # Get condition from kwargs
        cond = kwargs.pop("condition", None)
        self._log_tensor_stats(cond, "condition")
        
        # Forward pass with detailed logging
        try:
            self.model.train()
            self.autoencoder.eval()
            
            # First, log everything but without modifying the computation graph
            with torch.no_grad():
                z_mu, z_sigma = self.autoencoder.encode(input)
                self._log_tensor_stats(z_mu, "z_mu")
                self._log_tensor_stats(z_sigma, "z_sigma")
                
                z_sample = self.autoencoder.sampling(z_mu, z_sigma)
                self._log_tensor_stats(z_sample, "z_sample")
                
                # Project and reshape condition for logging
                cond_mode = self.args.model.condition.mode
                if cond is not None and cond_mode is not None:
                    cond_before = cond.clone()
                    self._log_tensor_stats(cond_before, "cond_before_proj")
                
                # Log model parameters
                for i, (name, param) in enumerate(self.model.named_parameters()):
                    if i < 5:  # Log only first 5 parameters to avoid too much output
                        self._log_tensor_stats(param.data, f"model_param_{name}")
                        self.debug_log.write(f"param_{name}.requires_grad = {param.requires_grad}\n")
            
            # Now do the actual forward pass for training with proper gradient tracking
            # Get latent representation of input
            with torch.no_grad():
                z_mu, z_sigma = self.autoencoder.encode(input)
                z = self.autoencoder.sampling(z_mu, z_sigma)
                noise = torch.randn_like(z).to(self.device)
            
            # Project and reshape condition
            cond_mode = self.args.model.condition.mode
            if cond is not None and cond_mode is not None:
                with torch.no_grad():
                    cond = self.cond_proj(cond)
                    self._log_tensor_stats(cond, "cond_after_proj")
                    
                    if cond_mode in ["concat", "addition"]:
                        old_shape = cond.shape
                        cond = torch.nn.functional.interpolate(
                            cond, z.shape[2:], 
                            mode=self.args.model.condition.resize_mode, 
                            antialias=True
                        )
                        self._log_tensor_stats(cond, f"cond_after_interpolate from {old_shape} to {cond.shape}")
                    elif cond_mode == "crossattn":
                        old_shape = cond.shape
                        cond = cond.flatten(start_dim=2)
                        self._log_tensor_stats(cond, f"cond_after_flatten from {old_shape} to {cond.shape}")

            # Generate random timesteps
            timesteps = torch.randint(0, self.inferer.scheduler.num_train_timesteps, (z.shape[0],), device=z.device).long()
            self._log_tensor_stats(timesteps, "timesteps")
            
            # Call inferer
            self.debug_log.write(f"Calling inferer with mode={cond_mode}\n")
            self.debug_log.flush()
            
            # Use original implementation but with explicit gradient tracking
            noise_pred = self.inferer(
                inputs=input,
                noise=noise,
                timesteps=timesteps,
                condition=cond,
                mode=cond_mode,
                diffusion_model=self.model,
                autoencoder_model=self.autoencoder
            )
            self._log_tensor_stats(noise_pred, "noise_pred")
            
            # Check if the noise_pred has gradients
            self.debug_log.write(f"noise_pred.requires_grad = {noise_pred.requires_grad}\n")
            self.debug_log.write(f"noise.requires_grad = {noise.requires_grad}\n")
            
            # Compute training loss
            loss = self.recon_loss_fn(noise_pred.float(), noise.float())
            self._log_tensor_stats(loss, "loss")
            
            # Make sure loss is scalar and requires grad
            if loss.numel() > 1:
                loss = loss.mean()
                self.debug_log.write(f"Reduced loss: {loss.item()}, requires_grad={loss.requires_grad}\n")
            
            # Ensure loss requires grad (shouldn't be needed but just in case)
            if not loss.requires_grad:
                self.debug_log.write("WARNING: Loss doesn't require grad! This will cause backward() to fail.\n")
                # Try to use an alternative method
                # Here we're falling back to the regular diffusion runner
                self.debug_log.write("Falling back to regular DiffusionRunner's train_step...\n")
                self.debug_log.flush()
                
                # Call the parent's train_step to avoid gradient issues
                return super().train_step(input, condition=cond)
            
            # Zero grad and back propagation
            self.optimiser.zero_grad(set_to_none=True)
            
            # Log that we're proceeding with backward
            self.debug_log.write("Starting backpropagation...\n")
            self.debug_log.flush()
            
            # Backward pass
            loss.backward()
            
            # Check gradients for NaN
            for i, (name, param) in enumerate(self.model.named_parameters()):
                if param.grad is not None and i < 5:
                    self._log_tensor_stats(param.grad, f"grad_{name}")
                elif param.grad is None and i < 5:
                    self.debug_log.write(f"WARNING: No gradient for parameter {name}\n")
            
            # Apply gradients
            self.debug_log.write("Applying gradients...\n")
            self.debug_log.flush()
            
            # Apply gradient clipping if configured
            if self.args.optim.grad_clip:
                self.debug_log.write(f"Applying gradient clipping with max_norm={self.args.optim.grad_clip}\n")
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.optim.grad_clip)
            
            self.optimiser.step()
            
            # Output dictionary update
            output.update({
                "loss": loss,
            })
            return output
                
        except Exception as e:
            self.debug_log.write(f"Exception in train_step: {str(e)}\n")
            import traceback
            self.debug_log.write(traceback.format_exc())
            self.debug_log.flush()
            raise e
            
    def __del__(self):
        """
        Close log file when runner is destroyed
        """
        if hasattr(self, 'debug_log'):
            self.debug_log.close() 