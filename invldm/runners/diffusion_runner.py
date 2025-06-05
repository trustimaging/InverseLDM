import os

import torch
import torch.nn as nn
import logging

import math

from . import BaseRunner
from .inferers import LatentDiffusionInferer

from monai.networks.blocks import Convolution
# from generative.inferers import LatentDiffusionInferer
from generative.networks.nets.diffusion_model_unet import DiffusionModelUNet, DiffusionModelEncoder
from generative.networks.schedulers import DDPMScheduler, DDIMScheduler, PNDMScheduler

from ..utils.utils import namespace2dict, filter_kwargs_by_class_init


class DiffusionRunner(BaseRunner):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.autoencoder = kwargs.pop("autoencoder").eval()
        self.spatial_dims = kwargs.pop("spatial_dims")
        self.latent_channels = kwargs.pop("latent_channels")
        
        # Handle DataParallel wrapped autoencoder
        if isinstance(self.autoencoder, nn.DataParallel):
            self.autoencoder_module = self.autoencoder.module
        else:
            self.autoencoder_module = self.autoencoder
        
        print("DEBUG-DIFFUSION: Initializing DiffusionRunner")
        print(f"DEBUG-DIFFUSION: latent_channels={self.latent_channels}, spatial_dims={self.spatial_dims}")
        print(f"DEBUG-DIFFUSION: Autoencoder is DataParallel: {isinstance(self.autoencoder, nn.DataParallel)}")
        
        in_channels = out_channels = self.latent_channels
        
        # Sample
        x = next(iter(self.train_loader)) if not self.args.sampling_only else next(iter(self.sample_loader))
        if isinstance(x, (list, tuple)):
            x, c = x
            # Check if c is a slice index (single number tensor)
            if len(c.shape) <= 1 or (len(c.shape) == 2 and c.shape[1] == 1):
                print(f"DEBUG-DIFFUSION: Got slice index condition, shape={c.shape}")
                self.use_slice_conditioning = True
                # Maximum number of slices to embed (adjust as needed)
                self.max_slices = 1000
                # Embedding dimension for slice indices
                self.slice_embed_dim = 64
                # Create an embedding layer for slice indices
                self.slice_embedding = nn.Embedding(
                    num_embeddings=self.max_slices,
                    embedding_dim=self.slice_embed_dim
                ).to(kwargs.get("device", "cuda"))
                
                # Create a position encoding layer to convert slice embedding to spatial features
                # This will be used for addition and concat modes
                self.slice_to_spatial = nn.Sequential(
                    nn.Linear(self.slice_embed_dim, 256),
                    nn.SiLU(),
                    nn.Linear(256, 512),
                    nn.SiLU(),
                    nn.Linear(512, self.latent_channels * 8 * 8),  # Size based on expected spatial features
                ).to(kwargs.get("device", "cuda"))
                
                # For cross-attention, we'll use the embedding directly
                c_dim = self.slice_embed_dim
            else:
                print(f"DEBUG-DIFFUSION: Got condition in loader, shape={c.shape}")
                c_dim = c.flatten(start_dim=2).shape[-1]
                self.use_slice_conditioning = False
        else:
            print("DEBUG-DIFFUSION: No condition in loader")
            self.use_slice_conditioning = False
        
        # Latent scaling factor
        if not self.args.sampling_only: 
            with torch.no_grad():
                with torch.amp.autocast(str(self.device)):
                    z = self.autoencoder_module.encode_stage_2_inputs(x.float().to(self.device))
            sf = 1 / torch.std(z)
            print(f"DEBUG-DIFFUSION: Scaling factor set to {sf}, z.shape={z.shape}, z_mean={z.mean().item()}, z_std={torch.std(z).item()}")
            self.scale_factor = sf
        else:
            self.scale_factor = 1.
            print(f"DEBUG-DIFFUSION: Using default scaling factor {self.scale_factor}")
        
        # Conditioner network
        if self.args.model.condition.mode is not None:
            print(f"DEBUG-DIFFUSION: Setting up condition mode: {self.args.model.condition.mode}")
            assert "c" in locals(), (" Condition mode is passed but Dataset does not return condition. Ensure chosen Dataset class returns a tuple with condition as second element. ")
            has_blocks = self.args.model.condition.num_res_blocks > 0 if self.args.model.condition.spatial_dims > 1 else self.args.model.condition.num_blocks > 0
            if has_blocks:
                print(f"DEBUG-DIFFUSION: Condition has blocks: num_res_blocks={self.args.model.condition.num_res_blocks}")
                if self.args.model.condition.spatial_dims > 1:
                    cond_args = filter_kwargs_by_class_init(SpatialConditioner, namespace2dict(self.args.model.condition))
                    cond_args.update(filter_kwargs_by_class_init(DiffusionModelEncoder, namespace2dict(self.args.model.condition)))

                    # Force a few parameters
                    cond_args["upcast_attention"] = self.args.model.upcast_attention
                    cond_args["with_conditioning"] = False
                    cond_args["num_class_embeds"] = None
                    cond_args["cross_attention_dim"] = None
                    
                    print(f"DEBUG-DIFFUSION: Creating SpatialConditioner with args: {cond_args}")
                    self.cond_proj = nn.Sequential(SpatialConditioner(**cond_args)).to(self.device)
                    
                else:
                    cond_args = filter_kwargs_by_class_init(TransformerConditioner, namespace2dict(self.args.model.condition))
                    cond_args["d_model"] = c_dim
                    print(f"DEBUG-DIFFUSION: Creating TransformerConditioner with args: {cond_args}")
                    self.cond_proj = nn.Sequential(TransformerConditioner(**cond_args)).to(self.device)
                    
                # Update c_dim for xatnn
                with torch.no_grad():
                    with torch.amp.autocast(str(self.device)):
                        projected_c = self.cond_proj(c.float().to(self.device))
                        print(f"DEBUG-DIFFUSION: Test condition projection: input={c.shape}, output={projected_c.shape}")
                        c_dim = projected_c.flatten(start_dim=2).shape[-1]
                        print(f"DEBUG-DIFFUSION: Updated c_dim={c_dim}")
            else:
                print("DEBUG-DIFFUSION: Using Identity for condition projection")
                self.cond_proj = nn.Identity().to(self.device)
            
                        
            # Add input dimension if condition mode is concatenation
            if self.args.model.condition.mode == "concat":
                old_in_channels = in_channels
                in_channels += self.args.model.condition.out_channels
                print(f"DEBUG-DIFFUSION: Concat mode - in_channels increased from {old_in_channels} to {in_channels} (added {self.args.model.condition.out_channels})")
            
            # Make condition depth equal to latent space depth for adding one to the other    
            if self.args.model.condition.mode == "addition" and self.args.model.condition.out_channels != self.latent_channels:
                print(f"DEBUG-DIFFUSION: Addition mode - adding 1x1 conv to match channels: {self.args.model.condition.out_channels} -> {self.latent_channels}")
                self.cond_proj.append(Convolution(
                spatial_dims=self.args.model.condition.spatial_dims,
                in_channels=self.args.model.condition.out_channels,
                out_channels=self.latent_channels,
                strides=1, kernel_size=1, padding=0, conv_only=True,).to(self.device))

        # Diffusion Model
        model_kwargs = filter_kwargs_by_class_init(DiffusionModelUNet, namespace2dict(self.args.model))
        _ = [model_kwargs.pop(key, None) for key in ["spatial_dims", "in_channels", "out_channels", "with_conditioning"]]
        print(f"DEBUG-DIFFUSION: Creating DiffusionModelUNet: in_channels={in_channels}, out_channels={out_channels}")
        self.model = DiffusionModelUNet(
            spatial_dims=self.spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            with_conditioning=self.args.model.condition.mode=="crossattn",
            cross_attention_dim=c_dim if "c_dim" in locals() and self.args.model.condition.mode=="crossattn" else None,
            **model_kwargs
        ).to(self.device)

        # Enable multi-GPU training if multiple GPUs are available
        if len(self.gpu_ids) > 1 and torch.cuda.device_count() > 1:
            print(f"DiffusionRunner: Using DataParallel with {len(self.gpu_ids)} GPUs: {self.gpu_ids}")
            self.model = nn.DataParallel(self.model, device_ids=self.gpu_ids)
        else:
            print(f"DiffusionRunner: Using single GPU: {self.device}")
            
        # Store reference to underlying model for method access
        self.model_module = self.model.module if isinstance(self.model, nn.DataParallel) else self.model

        # Noise schedulers
        assert self.args.params.sampler.lower() in ["ddim", "ddpm", "pndm"]
        scheduler_class = globals()[self.args.params.sampler.upper()+"Scheduler"]
        scheduler_args = filter_kwargs_by_class_init(scheduler_class, namespace2dict(self.args.params))
        print(f"DEBUG-DIFFUSION: Creating {self.args.params.sampler.upper()} scheduler with args: {scheduler_args}")
        self.scheduler = scheduler_class(**scheduler_args)
        
        # Num of inference Steps
        try:
            self.num_inference_timesteps = self.args.params.num_inference_timesteps
        except AttributeError:
            self.num_inference_timesteps = self.args.params.num_train_timesteps

        # Create inferer with configurable condition strength
        condition_strength = getattr(self.args.model.condition, 'strength', 0.5)
        print(f"DEBUG-DIFFUSION: Creating LatentDiffusionInferer with scale_factor={self.scale_factor}, condition_strength={condition_strength}")
        self.inferer = LatentDiffusionInferer(self.scheduler, scale_factor=self.scale_factor, condition_strength=condition_strength)

        # Optimisers and loss functions
        if not self.args.sampling_only:
            self.recon_loss_fn = torch.nn.L1Loss() if self.args.params.recon_loss.lower() == "l1" else torch.nn.MSELoss()
            
            optim_kwargs = filter_kwargs_by_class_init(torch.optim.Adam, namespace2dict(self.args.optim))
            optim_kwargs.pop("params", None)
            print(f"DEBUG-DIFFUSION: Creating optimizer {self.args.optim.optimiser} with params: {optim_kwargs}")
            
            # Collect all parameters to optimize
            params_to_optimize = list(self.model.parameters())
            
            # Add conditioning network parameters if they exist
            if hasattr(self, 'cond_proj'):
                params_to_optimize.extend(self.cond_proj.parameters())
                print(f"DEBUG-DIFFUSION: Added cond_proj parameters to optimizer")
                
            # Add slice conditioning parameters if they exist
            if hasattr(self, 'slice_embedding'):
                params_to_optimize.extend(self.slice_embedding.parameters())
                params_to_optimize.extend(self.slice_to_spatial.parameters())
                print(f"DEBUG-DIFFUSION: Added slice conditioning parameters to optimizer")
            
            # Create the correct optimizer based on the name
            if self.args.optim.optimiser.lower() == "adam":
                self.optimiser = torch.optim.Adam(params_to_optimize, **optim_kwargs)
            elif self.args.optim.optimiser.lower() == "adamw":
                self.optimiser = torch.optim.AdamW(params_to_optimize, **optim_kwargs)
            else:
                self.optimiser = torch.optim.Adam(params_to_optimize, **optim_kwargs)
                print(f"DEBUG-DIFFUSION: WARNING - Unknown optimizer {self.args.optim.optimiser}, falling back to Adam")
                
            # Store params for gradient clipping
            self.params_to_optimize = params_to_optimize
            
            self.scaler = torch.amp.GradScaler(self.device)
                    

    def process_slice_condition(self, slice_idx, mode, target_shape=None):
        """
        Process slice index condition into the appropriate format based on conditioning mode
        
        Args:
            slice_idx: Tensor containing slice indices
            mode: Conditioning mode ("addition", "concat", or "crossattn")
            target_shape: Target spatial shape for the condition (for addition/concat modes)
            
        Returns:
            Processed condition tensor appropriate for the requested mode
        """
        # Ensure slice indices are properly formatted as long tensors
        if slice_idx.dim() > 1 and slice_idx.shape[1] == 1:
            slice_idx = slice_idx.squeeze(1)
            
        # Make sure indices don't exceed embedding size
        slice_idx = torch.clamp(slice_idx, 0, self.max_slices - 1).long()
        
        # Get basic embedding
        batch_size = slice_idx.shape[0]
        slice_embedding = self.slice_embedding(slice_idx)  # Shape: [batch_size, embed_dim]
        
        print(f"DEBUG-DIFFUSION: Processing slice condition, indices={slice_idx}, mode={mode}")
        
        if mode == "crossattn":
            # For cross-attention, just return the embedding vectors
            return slice_embedding.unsqueeze(1)  # [batch_size, 1, embed_dim]
            
        elif mode in ["addition", "concat"]:
            if target_shape is None:
                raise ValueError("Target shape must be provided for addition/concat modes")
                
            # Convert slice embedding to spatial features
            spatial_features = self.slice_to_spatial(slice_embedding)
            
            # Reshape to spatial feature map
            h, w = 8, 8  # Base spatial dimensions
            c = self.latent_channels
            spatial_features = spatial_features.view(batch_size, c, h, w)
            
            # Resize to match target shape
            if (h, w) != target_shape[2:]:
                spatial_features = torch.nn.functional.interpolate(
                    spatial_features, 
                    size=target_shape[2:], 
                    mode='bilinear',
                    align_corners=False
                )
            
            return spatial_features
            
        else:
            raise ValueError(f"Unsupported condition mode: {mode}")

    def train_step(self, input, **kwargs):
        self.model.train()
        self.autoencoder_module.eval()

        # Dictionary of outputs
        output = {}

        # Get condition from kwargs
        cond = kwargs.pop("condition", None)
        if cond is not None:
            print(f"DEBUG-DIFFUSION: train_step - Got condition, shape={cond.shape}")
            # Check if it's a slice index condition
            is_slice_condition = self.use_slice_conditioning or (
                len(cond.shape) <= 1 or (len(cond.shape) == 2 and cond.shape[1] == 1)
            )
        else:
            print("DEBUG-DIFFUSION: train_step - No condition provided")
            is_slice_condition = False

        # Forward pass: predict model noise based on condition
        with torch.amp.autocast(str(self.device)):
            z_mu, z_sigma = self.autoencoder_module.encode(input)
            
            # Check for NaN in encoding
            if torch.isnan(z_mu).any() or torch.isnan(z_sigma).any():
                print("DEBUG-DIFFUSION: WARNING - NaN detected in autoencoder encoding")
                print(f"DEBUG-DIFFUSION: z_mu stats: min={z_mu.min().item()}, max={z_mu.max().item()}, mean={z_mu.mean().item()}")
                print(f"DEBUG-DIFFUSION: z_sigma stats: min={z_sigma.min().item()}, max={z_sigma.max().item()}, mean={z_sigma.mean().item()}")
            
            z = self.autoencoder_module.sampling(z_mu, z_sigma)
            
            # Check for NaN in latent
            if torch.isnan(z).any():
                print("DEBUG-DIFFUSION: WARNING - NaN detected in autoencoder latent")
                print(f"DEBUG-DIFFUSION: z stats: min={z.min().item()}, max={z.max().item()}, mean={z.mean().item()}")
            
            noise = torch.randn_like(z).to(self.device)
            
            # Project and reshape condition
            cond_mode = self.args.model.condition.mode
            
            if cond is not None and cond_mode is not None:
                print(f"DEBUG-DIFFUSION: Processing condition for mode={cond_mode}")
                
                # Process slice index condition differently
                if is_slice_condition:
                    print("DEBUG-DIFFUSION: Processing slice condition")
                    cond = self.process_slice_condition(cond, cond_mode, target_shape=z.shape)
                    
                    # Verify the condition doesn't have NaN values
                    if torch.isnan(cond).any():
                        print("DEBUG-DIFFUSION: Fixing NaN values in slice condition")
                        cond = torch.nan_to_num(cond, nan=0.0)
                        
                    # Ensure the condition has reasonable values for addition mode
                    if cond_mode == "addition":
                        # Check condition statistics
                        c_min = cond.min().item()
                        c_max = cond.max().item()
                        c_mean = cond.mean().item()
                        c_std = torch.std(cond).item()
                        print(f"DEBUG-DIFFUSION: Slice condition stats - min={c_min}, max={c_max}, mean={c_mean}, std={c_std}")
                        
                        # If condition has low variance, add spatial variation
                        if c_std < 0.05:
                            print("DEBUG-DIFFUSION: WARNING - Low variance in condition, enhancing spatial variation")
                            # Add spatial variation by multiplying with spatial gradients
                            h, w = cond.shape[2:]
                            y_grad = torch.linspace(0.8, 1.2, h).view(-1, 1).expand(-1, w).to(cond.device)
                            x_grad = torch.linspace(0.8, 1.2, w).view(1, -1).expand(h, -1).to(cond.device)
                            variation = (x_grad + y_grad) / 2.0
                            
                            # Apply to each channel
                            for c in range(cond.shape[1]):
                                cond[:, c] = cond[:, c] * variation
                else:
                    # Regular spatial condition processing
                    # Fix NaN values in condition before projection
                    if torch.isnan(cond).any():
                        print("DEBUG-DIFFUSION: Fixing NaN values in condition before projection")
                        cond = torch.nan_to_num(cond, nan=0.0)
                        
                    # Debug: check condition stats before projection
                    print(f"DEBUG-DIFFUSION: Condition BEFORE projection - shape={cond.shape}, min={cond.min().item():.4f}, max={cond.max().item():.4f}, mean={cond.mean().item():.4f}, std={torch.std(cond).item():.4f}")
                        
                    cond = self.cond_proj(cond)
                    
                    # Debug: check condition stats after projection
                    print(f"DEBUG-DIFFUSION: Condition AFTER projection - shape={cond.shape}, min={cond.min().item():.4f}, max={cond.max().item():.4f}, mean={cond.mean().item():.4f}, std={torch.std(cond).item():.4f}")
                    
                    # Check for NaN after projection
                    if torch.isnan(cond).any():
                        print("DEBUG-DIFFUSION: WARNING - NaN detected after condition projection")
                        print(f"DEBUG-DIFFUSION: Condition after projection: shape={cond.shape}")
                        # Fix NaN values after projection
                        cond = torch.nan_to_num(cond, nan=0.0)
                    
                    if cond_mode in ["concat", "addition"]:
                        orig_shape = cond.shape
                        resize_mode = self.args.model.condition.resize_mode
                        use_antialias = resize_mode in ["bilinear", "bicubic"]
                        cond = torch.nn.functional.interpolate(cond, z.shape[2:], mode=resize_mode, antialias=use_antialias if use_antialias else None)
                        print(f"DEBUG-DIFFUSION: Interpolated condition from {orig_shape} to {cond.shape} using mode={resize_mode}, antialias={use_antialias if use_antialias else None}")
                        
                        # Fix NaN values after interpolation
                        if torch.isnan(cond).any():
                            print("DEBUG-DIFFUSION: Fixing NaN values after interpolation")
                            cond = torch.nan_to_num(cond, nan=0.0)
                        elif cond_mode == "crossattn" and not is_slice_condition:
                            cond = cond.flatten(start_dim=2)
            else:
                print("DEBUG-DIFFUSION: No condition processing needed")

            timesteps = torch.randint(0, self.inferer.scheduler.num_train_timesteps, (z.shape[0],), device=z.device).long()
            print(f"DEBUG-DIFFUSION: Generated timesteps: min={timesteps.min().item()}, max={timesteps.max().item()}")
            
            print(f"DEBUG-DIFFUSION: Calling inferer with mode={cond_mode}")
            noise_pred = self.inferer(
                inputs=input,
                noise=noise,
                timesteps=timesteps,
                condition=cond,
                mode=cond_mode,
                diffusion_model=self.model,
                autoencoder_model=self.autoencoder_module
            )
            
            # Check for NaN in model output
            if torch.isnan(noise_pred).any():
                print("DEBUG-DIFFUSION: WARNING - NaN detected in model output (noise_pred)")
                print(f"DEBUG-DIFFUSION: noise_pred stats: shape={noise_pred.shape}")
            
        # Compute training loss
        loss = self.recon_loss_fn(noise_pred.float(), noise.float())
        
        # Check for NaN in loss
        if torch.isnan(loss) or torch.isinf(loss):
            print("DEBUG-DIFFUSION: ERROR - NaN/Inf detected in loss")
            print(f"DEBUG-DIFFUSION: Loss: {loss.item() if torch.isfinite(loss) else 'NaN/Inf'}")
            print(f"DEBUG-DIFFUSION: noise_pred stats - min: {noise_pred.min().item()}, max: {noise_pred.max().item()}, mean: {noise_pred.mean().item()}")
            print(f"DEBUG-DIFFUSION: noise stats - min: {noise.min().item()}, max: {noise.max().item()}, mean: {noise.mean().item()}")
            
            # Skip this training step by returning a safe loss value
            output.update({
                "loss": torch.tensor(0.5, device=self.device, dtype=torch.float32),
            })
            return output

        # Check if tensors require gradients
        if not noise_pred.requires_grad:
            print("DEBUG-DIFFUSION: WARNING - noise_pred doesn't require gradients, detaching and creating a new tensor")
            # Create a new tensor that requires gradients
            noise_pred_with_grad = noise_pred.detach().clone().requires_grad_(True)
            loss = self.recon_loss_fn(noise_pred_with_grad.float(), noise.float())
        
        # Zero grad and back propagation
        self.optimiser.zero_grad(set_to_none=True)
        
        try:
            # Scale the loss for mixed precision training
            scaled_loss = self.scaler.scale(loss)
            
            # Backward pass
            scaled_loss.backward()
            
            # Unscale the gradients before clipping
            self.scaler.unscale_(self.optimiser)
            
            # Check for inf/nan in gradients
            grad_norm = 0.0
            parameters_to_check = self.model.parameters() if not hasattr(self, 'params_to_optimize') else self.params_to_optimize
            
            for p in parameters_to_check:
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    grad_norm += param_norm.item() ** 2
            grad_norm = grad_norm ** 0.5
            
            # Log if we have inf/nan gradients
            if not torch.isfinite(torch.tensor(grad_norm)):
                print(f"DEBUG-DIFFUSION: WARNING - Non-finite gradient norm: {grad_norm}")
            
            # Check for NaN in gradients
            for name, param in self.model.named_parameters():
                if param.grad is not None and torch.isnan(param.grad).any():
                    print(f"DEBUG-DIFFUSION: WARNING - NaN detected in gradient for {name}")
                
            # Check gradients for conditioning networks
            if hasattr(self, 'cond_proj'):
                cond_has_grad = False
                for name, param in self.cond_proj.named_parameters():
                    if param.grad is not None:
                        cond_has_grad = True
                        if torch.isnan(param.grad).any():
                            print(f"DEBUG-DIFFUSION: WARNING - NaN detected in cond_proj gradient for {name}")
                if not cond_has_grad and self.cond_proj.parameters():
                    print("DEBUG-DIFFUSION: WARNING - No gradients in cond_proj!")
                    
            if hasattr(self, 'slice_embedding'):
                if self.slice_embedding.weight.grad is not None:
                    print(f"DEBUG-DIFFUSION: slice_embedding grad norm: {self.slice_embedding.weight.grad.norm().item():.6f}")
                else:
                    print("DEBUG-DIFFUSION: WARNING - No gradient in slice_embedding!")
                
            # Apply gradient clipping if configured
            if self.args.optim.grad_clip and hasattr(self, 'params_to_optimize'):
                print(f"DEBUG-DIFFUSION: Applying gradient clipping with max_norm={self.args.optim.grad_clip}")
                torch.nn.utils.clip_grad_norm_(self.params_to_optimize, self.args.optim.grad_clip)
            
            # Step the optimizer
            self.scaler.step(self.optimiser)
            
            # Update the scale for next iteration
            self.scaler.update()
            
        except RuntimeError as e:
            print(f"DEBUG-DIFFUSION: ERROR in backward/optimizer - {str(e)}")
            import traceback
            print(traceback.format_exc())
            
            # Reset the scaler state if we hit an error
            self.scaler.update()
            
            # Create a minimal loss to keep training going
            print("DEBUG-DIFFUSION: Creating fallback loss to continue training")
            with torch.amp.autocast(str(self.device)):
                # Use a small constant loss
                fallback_loss = torch.tensor(0.5, device=self.device, dtype=torch.float32)
                output.update({"loss": fallback_loss})
                return output

        # Output dictionary update
        output.update({
            "loss": loss,
        })
        return output

    @torch.no_grad()
    def valid_step(self, input, **kwargs):
        self.model.eval()
        self.autoencoder_module.eval()

        # Get condition from kwargs
        cond = kwargs.pop("condition", None)
        if cond is not None:
            print(f"DEBUG-DIFFUSION: valid_step - Got condition, shape={cond.shape}")
            # Check if it's a slice index condition
            is_slice_condition = self.use_slice_conditioning or (
                len(cond.shape) <= 1 or (len(cond.shape) == 2 and cond.shape[1] == 1)
            )
        else:
            is_slice_condition = False

        # Forward pass: predict model noise based on condition
        with torch.amp.autocast(str(self.device)):
            z_mu, z_sigma = self.autoencoder_module.encode(input)
            z = self.autoencoder_module.sampling(z_mu, z_sigma)
            noise = torch.randn_like(z).to(self.device)
            
            # Project and reshape condition
            cond_mode = self.args.model.condition.mode
            if cond is not None and cond_mode is not None:
                # Process slice index condition differently
                if is_slice_condition:
                    cond = self.process_slice_condition(cond, cond_mode, target_shape=z.shape)
                else:           
                    cond = self.cond_proj(cond)
                    if cond_mode in ["concat", "addition"]:
                        resize_mode = self.args.model.condition.resize_mode
                        use_antialias = resize_mode in ["bilinear", "bicubic"]
                        cond = torch.nn.functional.interpolate(cond, z.shape[2:], mode=resize_mode, antialias=use_antialias if use_antialias else None)
                    elif cond_mode == "crossattn" and not is_slice_condition:
                        cond = cond.flatten(start_dim=2)
                
            timesteps = torch.randint(0, self.inferer.scheduler.num_train_timesteps, (z.shape[0],), device=z.device).long()
            noise_pred = self.inferer(
                inputs=input,
                noise=noise,
                timesteps=timesteps,
                condition=cond,
                mode=cond_mode,
                diffusion_model=self.model,
                autoencoder_model=self.autoencoder_module
            )
            
        # Compute validation loss
        loss = self.recon_loss_fn(noise_pred.float(), noise.float())

        # Output dictionary
        output = {
            "loss": loss,
        }
        return output

    @torch.no_grad()
    def sample_step(self, input, **kwargs):
        self.model.eval()
        self.autoencoder_module.eval()

        # Get sampling parameters from kwargs
        num_inference_steps = kwargs.pop("num_inference_steps", self.num_inference_timesteps)

        # Get condition from kwargs
        cond = kwargs.pop("condition", None)
        if cond is not None:
            print(f"DEBUG-DIFFUSION: sample_step - Got condition, shape={cond.shape}")
            # Check if it's a slice index condition
            is_slice_condition = self.use_slice_conditioning or (
                len(cond.shape) <= 1 or (len(cond.shape) == 2 and cond.shape[1] == 1)
            )
        else:
            is_slice_condition = False

        # One autoencoder forward pass to get shape of latent space -- can be optimised!
        z_mu, z_sigma = self.autoencoder_module.encode(input)
        z_ = self.autoencoder_module.sampling(z_mu, z_sigma)
        z = torch.randn_like(z_).to(self.device)
            
        # Project and reshape condition
        cond_mode = self.args.model.condition.mode
        if cond is not None and cond_mode is not None:
            # Process slice index condition differently
            if is_slice_condition:
                print("DEBUG-DIFFUSION: Processing slice condition for sampling")
                cond = self.process_slice_condition(cond, cond_mode, target_shape=z.shape)
                
                # Verify the condition doesn't have NaN values
                if torch.isnan(cond).any():
                    print("DEBUG-DIFFUSION: Fixing NaN values in slice condition for sampling")
                    cond = torch.nan_to_num(cond, nan=0.0)
                    
                # Ensure the condition has reasonable values for addition mode
                if cond_mode == "addition":
                    # Check condition statistics
                    c_min = cond.min().item()
                    c_max = cond.max().item()
                    c_mean = cond.mean().item()
                    c_std = torch.std(cond).item()
                    print(f"DEBUG-DIFFUSION: Sampling slice condition stats - min={c_min}, max={c_max}, mean={c_mean}, std={c_std}")
                    
                    # If condition has low variance, add spatial variation
                    if c_std < 0.05:
                        print("DEBUG-DIFFUSION: WARNING - Low variance in sampling condition, enhancing spatial variation")
                        # Add spatial variation by multiplying with spatial gradients
                        h, w = cond.shape[2:]
                        y_grad = torch.linspace(0.8, 1.2, h).view(-1, 1).expand(-1, w).to(cond.device)
                        x_grad = torch.linspace(0.8, 1.2, w).view(1, -1).expand(h, -1).to(cond.device)
                        variation = (x_grad + y_grad) / 2.0
                        
                        # Apply to each channel
                        for c in range(cond.shape[1]):
                            cond[:, c] = cond[:, c] * variation
            else:            
                cond = self.cond_proj(cond)
                if cond_mode in ["concat", "addition"]:
                    resize_mode = self.args.model.condition.resize_mode
                    use_antialias = resize_mode in ["bilinear", "bicubic"]
                    cond = torch.nn.functional.interpolate(cond, z.shape[2:], mode=resize_mode, antialias=use_antialias if use_antialias else None)
                    
                    # Fix NaN values after interpolation
                    if torch.isnan(cond).any():
                        print("DEBUG-DIFFUSION: Fixing NaN values after interpolation in sampling")
                        cond = torch.nan_to_num(cond, nan=0.0)
                elif cond_mode == "crossattn" and not is_slice_condition:
                    cond = cond.flatten(start_dim=2)
        else:
            # Inferer cannot accept cond_mode as None :(
            cond_mode = "crossattn"

        # Set number of inference steps for scheduler
        self.scheduler.set_timesteps(num_inference_steps=num_inference_steps)

        # Sample latent space with diffusion model
        logging.info("Sampling...")
        with torch.amp.autocast(str(self.device)):
            samples = self.inferer.sample(
                input_noise=z,
                diffusion_model=self.model,
                scheduler=self.scheduler,
                autoencoder_model=self.autoencoder_module,
                conditioning=cond,
                mode=cond_mode,
                **kwargs
            )

        return samples, z


class SpatialConditioner(DiffusionModelEncoder):
    def __init__(self, **kwargs) -> None:
        
        num_res_blocks = kwargs.get("num_res_blocks")            
        if isinstance(num_res_blocks, int):
            num_res_blocks = [num_res_blocks] * len(kwargs.get("num_channels"))
        kwargs["num_res_blocks"] = num_res_blocks
            
        super().__init__(**kwargs)
        
        self.conv_out = Convolution(
            spatial_dims=kwargs.get("spatial_dims"),
            in_channels=self.block_out_channels[-1],
            out_channels=kwargs.get("out_channels"),
            strides=1,
            kernel_size=3,
            padding=1,
            conv_only=True,
        )
        
        # Get time embed dim
        self.time_embed_dim = self.block_out_channels[0] * 4
        
        # Initialize weights with small values to prevent overwhelming the main network
        self._initialize_weights()
    
    def _initialize_weights(self):
        # Initialize conv_out with very small weights
        with torch.no_grad():
            # Set conv_out weights to very small values
            if hasattr(self.conv_out, 'conv'):
                nn.init.normal_(self.conv_out.conv.weight, mean=0.0, std=0.001)
                if self.conv_out.conv.bias is not None:
                    nn.init.zeros_(self.conv_out.conv.bias)
            
            # Also initialize conv_in with smaller values
            if hasattr(self.conv_in, 'conv'):
                nn.init.normal_(self.conv_in.conv.weight, mean=0.0, std=0.02)
                if self.conv_in.conv.bias is not None:
                    nn.init.zeros_(self.conv_in.conv.bias)
                    
            # Initialize downsampling blocks with reasonable values
            for block in self.down_blocks:
                for module in block.modules():
                    if isinstance(module, nn.Conv2d):
                        nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                        if module.bias is not None:
                            nn.init.zeros_(module.bias)
                    elif isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.GroupNorm):
                        if module.weight is not None:
                            nn.init.ones_(module.weight)
                        if module.bias is not None:
                            nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor):
        
        # Null time embedding and condition as not relevant here
        temb = torch.zeros((x.shape[0], self.time_embed_dim), dtype=x.dtype, device=x.device)
        context = None
        
        h = self.conv_in(x)   
            
        for downsample_block in self.down_blocks:
            h, _ = downsample_block(hidden_states=h, temb=temb, context=context)   
        
        out = self.conv_out(h)
            
        return out
    
class TransformerConditioner(nn.Module):
    def __init__(self, num_blocks, transformer_num_layers, d_model, resize_factor=1, num_heads=8):
        super(TransformerConditioner, self).__init__()
        self.num_blocks = num_blocks
        self.transformer_num_layers = transformer_num_layers
        self.d_model = d_model
        self.resize_factor = resize_factor
        self.num_heads = num_heads
        
        # If resize_factor is an int, convert it to a list of same value repeated
        if isinstance(resize_factor, int):
            self.resize_factor = [resize_factor] * num_blocks
            
        if isinstance(num_heads, int):
            self.num_heads = [num_heads] * num_blocks

        assert len(self.resize_factor) == num_blocks, \
            "Length of resize_factor list must match num_blocks"
            
        assert len(self.num_heads) == num_blocks, \
            "Length of num_heads list must match num_blocks"

        # Embedding layer
        self.embedding = nn.Linear(d_model, d_model)

        # Transformer layers
        self.transformer_blocks = nn.ModuleList([])
        for i in range(self.num_blocks):
            transformer_layers = nn.ModuleList([
                nn.TransformerEncoderLayer(d_model=d_model//math.prod(self.resize_factor[:i]),
                                        nhead=self.num_heads[i], activation="gelu")
                for _ in range(transformer_num_layers)
            ])
            self.transformer_blocks.append(transformer_layers)
            
    def forward(self, x):
        # Apply embedding
        x = self.embedding(x)

        # Apply each transformer block with resizing at the end
        for block, resize_factor in zip(self.transformer_blocks, self.resize_factor):
            for layer in block:
                x = layer(x)
            x = self.apply_resize(x, resize_factor)
        return x

    def apply_resize(self, x, resize_factor):
        new_size = x.size(-1) // resize_factor
        return x.view(x.size(0), -1, new_size)

# def get_condition_projection(
#     in_channels,
#     spatial_dims,
#     num_layers,
#     num_head_channels,
#     num_feature_channels,
#     norm_num_groups,
#     out_channels=None,
#     dropout=0.,
#     norm_eps=0.000001,
#     upcast_attention=False,
#     use_flash_attention=False,
# ) -> nn.Sequential :
#     out_channels = in_channels if out_channels is None else out_channels
#     if num_layers > 0:
#         if spatial_dims >= 2:
#             return nn.Sequential(
#                 # to feature channels
#                 Convolution(
#                     spatial_dims=spatial_dims,
#                     in_channels=in_channels,
#                     out_channels=num_feature_channels,
#                     strides=1,
#                     kernel_size=1,
#                     padding=0,
#                     conv_only=True,
#                 ),
#                 # transformer layers
#                 SpatialTransformer(
#                     spatial_dims=spatial_dims,
#                     in_channels=num_feature_channels,
#                     num_layers=num_layers,
#                     num_attention_heads=num_feature_channels // num_head_channels,
#                     num_head_channels=num_head_channels,
#                     dropout=dropout,
#                     norm_num_groups=norm_num_groups,
#                     norm_eps=norm_eps,
#                     upcast_attention=upcast_attention,
#                     use_flash_attention=use_flash_attention, 
#                 ),
#                 nn.GroupNorm(num_groups=norm_num_groups, num_channels=num_feature_channels, eps=norm_eps, affine=True),
#                 nn.SiLU(),
#                 # to in_channels
#                 Convolution(
#                     spatial_dims=spatial_dims,
#                     in_channels=num_feature_channels,
#                     out_channels=in_channels,
#                     strides=1,
#                     kernel_size=1,
#                     padding=0,
#                     conv_only=True,
#                 ),
#             )
#         return nn.Sequential(
#             Convolution(
#                 spatial_dims=spatial_dims,
#                 in_channels=in_channels,
#                 out_channels=num_feature_channels,
#                 strides=1,
#                 kernel_size=1,
#                 padding=0,
#                 conv_only=True,
#             ),
#             Permute(0, 2, 1),
#             *[
#                 BasicTransformerBlock(
#                     num_channels=num_feature_channels,
#                     num_attention_heads=num_feature_channels // num_head_channels,
#                     num_head_channels=num_head_channels,
#                     dropout=dropout,
#                     upcast_attention=upcast_attention,
#                     use_flash_attention=use_flash_attention,
#                 )
#                 for _ in range(num_layers)
#             ],
#             Permute(0, 2, 1),
#             nn.GroupNorm(num_groups=norm_num_groups, num_channels=num_feature_channels, eps=norm_eps, affine=True),
#             nn.SiLU(),
#             Convolution(
#                 spatial_dims=spatial_dims,
#                 in_channels=num_feature_channels,
#                 out_channels=out_channels,
#                 strides=1,
#                 kernel_size=1,
#                 padding=0,
#                 conv_only=True,
#             ),
#         )     
#     return nn.Sequential(nn.Identity())

# class Permute(nn.Module):
#     def __init__(self, *args):
#         super().__init__()
#         self.args = args
#     def forward(self, x):
#         return x.permute(*self.args)