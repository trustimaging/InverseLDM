import os

import torch
import torch.nn as nn
import logging

from typing import Optional

from . import BaseRunner

from monai.networks.blocks import Convolution
from generative.inferers import LatentDiffusionInferer
from generative.networks.nets.diffusion_model_unet import DiffusionModelUNet, SpatialTransformer, BasicTransformerBlock
from generative.networks.schedulers import DDPMScheduler, DDIMScheduler, PNDMScheduler

from ..utils.utils import namespace2dict, filter_kwargs_by_class_init


class DiffusionRunner(BaseRunner):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.autoencoder = kwargs.pop("autoencoder").eval()
        self.spatial_dims = kwargs.pop("spatial_dims")
        self.latent_channels = kwargs.pop("latent_channels")
        
        in_channels = out_channels = self.latent_channels
        
        # Sample
        x = next(iter(self.train_loader)) if not self.args.sampling_only else next(iter(self.sample_loader))
        if isinstance(x, (list, tuple)):
            x, c = x
            c_dim = c.flatten(start_dim=2).shape[-1]
        
        # Latent scaling factor
        if not self.args.sampling_only: 
            with torch.no_grad():
                with torch.amp.autocast(str(self.device)):
                    z = self.autoencoder.encode_stage_2_inputs(x.float().to(self.device))
            logging.info(f"Scaling factor set to {1/torch.std(z)}")
            self.scale_factor = 1 / torch.std(z)
        else:
            self.scale_factor = 1.
        
        # Condition embedding layers for concatenation
        if self.args.model.condition.mode is not None:
            assert "c" in locals(), (" Condition mode is passed but Dataset does not return condition. Ensure chosen Dataset class returns a tuple with condition as second element. ")
            self.cond_proj = get_condition_projection(
                in_channels=self.args.model.condition.in_channels,
                spatial_dims=self.args.model.condition.spatial_dims,
                num_layers=self.args.model.condition.num_proj_layers,
                num_head_channels=self.args.model.condition.num_proj_head_channels,
                num_feature_channels=self.args.model.condition.num_proj_feature_channels,
                norm_num_groups=self.args.model.condition.norm_proj_num_groups,
                dropout=self.args.model.condition.proj_dropout,
                norm_eps=self.args.model.condition.proj_norm_eps,
                upcast_attention=self.args.model.upcast_attention,
                use_flash_attention=self.args.model.use_flash_attention,
            ).to(self.device)
                        
            # Add input dimension if condition mode is concatenation
            if self.args.model.condition.mode == "concat":
                in_channels += self.args.model.condition.in_channels

        # Diffusion Model
        model_kwargs = filter_kwargs_by_class_init(DiffusionModelUNet, namespace2dict(self.args.model))
        _ = [model_kwargs.pop(key, None) for key in ["spatial_dims", "in_channels", "out_channels", "with_conditioning"]]
        self.model = DiffusionModelUNet(
            spatial_dims=self.spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            with_conditioning=self.args.model.condition.mode=="crossattn",
            cross_attention_dim=c_dim if "c_dim" in locals() and self.args.model.condition.mode=="crossattn" else None,
            **model_kwargs
        ).to(self.device)

        # Noise schedulers
        assert self.args.params.sampler.lower() in ["ddim", "ddpm", "pndm"]
        scheduler_class = globals()[self.args.params.sampler.upper()+"Scheduler"]
        scheduler_args = filter_kwargs_by_class_init(scheduler_class, namespace2dict(self.args.params))
        self.scheduler = scheduler_class(**scheduler_args)
        
        # Num of inference Steps
        try:
            self.num_inference_timesteps = self.args.params.num_inference_timesteps
        except AttributeError:
            self.num_inference_timesteps = self.args.params.num_train_timesteps

        # Inferer
        self.inferer = LatentDiffusionInferer(self.scheduler, scale_factor=self.scale_factor)

        # Optimisers and loss functions
        if not self.args.sampling_only:
            self.recon_loss_fn = torch.nn.L1Loss() if self.args.params.recon_loss.lower() == "l1" else torch.nn.MSELoss()
            
            optim_kwargs = filter_kwargs_by_class_init(torch.optim.Adam, namespace2dict(self.args.optim))
            optim_kwargs.pop("params", None)
            self.optimiser = torch.optim.Adam(self.model.parameters(), **optim_kwargs)
            self.scaler = torch.amp.GradScaler(self.device)
                    

    def train_step(self, input, **kwargs):
        self.model.train()
        self.autoencoder.eval()

        # Dictionary of outputs
        output = {}

        # Get condition from kwargs
        cond = kwargs.pop("condition", None)

        # Forward pass: predict model noise based on condition
        with torch.amp.autocast(str(self.device)):
            z_mu, z_sigma = self.autoencoder.encode(input)
            z = self.autoencoder.sampling(z_mu, z_sigma)
            noise = torch.randn_like(z).to(self.device)
            
            # Project and reshape condition
            cond_mode = self.args.model.condition.mode
            if cond is not None and cond_mode is not None:     
                cond = self.cond_proj(cond)
                if cond_mode == "concat":
                    cond = torch.nn.functional.interpolate(cond, z.shape[2:], mode=self.args.model.condition.resize_mode, antialias=True)
                elif cond_mode == "crossattn":
                    cond = cond.flatten(start_dim=2)
            else:
                # Inferer cannot accept cond_mode as None :(
                cond_mode = "crossattn"

            timesteps = torch.randint(0, self.inferer.scheduler.num_train_timesteps, (z.shape[0],), device=z.device).long()
            noise_pred = self.inferer(
                inputs=input,
                noise=noise,
                timesteps=timesteps,
                condition=cond,
                mode=cond_mode,
                diffusion_model=self.model,
                autoencoder_model=self.autoencoder
            )
            
        # Compute training loss
        loss = self.recon_loss_fn(noise_pred.float(), noise.float())
        
        # Zero grad and back propagation
        self.optimiser.zero_grad(set_to_none=True)
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimiser)
        self.scaler.update()

        # Gradient Clipping
        # if self.args.optim.grad_clip:
        #     torch.nn.utils.clip_grad_norm_(self.model.parameters(),
        #                                    self.args.optim.grad_clip)

        # Output dictionary update
        output.update({
            "loss": loss,
        })
        return output

    @torch.no_grad()
    def valid_step(self, input, **kwargs):
        self.model.eval()
        self.autoencoder.eval()

        # Get condition from kwargs
        cond = kwargs.pop("condition", None)

        # Forward pass: predict model noise based on condition
        with torch.amp.autocast(str(self.device)):
            z_mu, z_sigma = self.autoencoder.encode(input)
            z = self.autoencoder.sampling(z_mu, z_sigma)
            noise = torch.randn_like(z).to(self.device)
            
            # Project and reshape condition
            cond_mode = self.args.model.condition.mode
            if cond is not None and cond_mode is not None:             
                cond = self.cond_proj(cond)
                if cond_mode == "concat":
                    cond = torch.nn.functional.interpolate(cond, z.shape[2:], mode=self.args.model.condition.resize_mode, antialias=True)
                elif cond_mode == "crossattn":
                    cond = cond.flatten(start_dim=2)
            else:
                # Inferer cannot accept cond_mode as None :(
                cond_mode = "crossattn"
                
            timesteps = torch.randint(0, self.inferer.scheduler.num_train_timesteps, (z.shape[0],), device=z.device).long()
            noise_pred = self.inferer(
                inputs=input,
                noise=noise,
                timesteps=timesteps,
                condition=cond,
                mode=cond_mode,
                diffusion_model=self.model,
                autoencoder_model=self.autoencoder
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
        self.autoencoder.eval()

        # Get sampling parameters from kwargs
        num_inference_steps = kwargs.pop("num_inference_steps", self.num_inference_timesteps)

        # Get condition from kwargs
        cond = kwargs.pop("condition", None)

        # One autoencoder forward pass to get shape of latent space -- can be optimised!
        z_mu, z_sigma = self.autoencoder.encode(input)
        z_ = self.autoencoder.sampling(z_mu, z_sigma)
        z = torch.randn_like(z_).to(self.device)
            
        # Project and reshape condition
        cond_mode = self.args.model.condition.mode
        if cond is not None and cond_mode is not None:             
            cond = self.cond_proj(cond)
            if cond_mode == "concat":
                cond = torch.nn.functional.interpolate(cond, z.shape[2:], mode=self.args.model.condition.resize_mode, antialias=True)
            elif cond_mode == "crossattn":
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
                autoencoder_model=self.autoencoder,
                conditioning=cond,
                mode=cond_mode,
                **kwargs
            )

        return samples, z


def get_condition_projection(
    in_channels,
    spatial_dims,
    num_layers,
    num_head_channels,
    num_feature_channels,
    norm_num_groups,
    dropout=0.,
    norm_eps=0.000001,
    upcast_attention=False,
    use_flash_attention=False,
) -> nn.Sequential :
    if num_layers > 0:
        if spatial_dims >= 2:
            return nn.Sequential(
                # to feature channels
                Convolution(
                    spatial_dims=spatial_dims,
                    in_channels=in_channels,
                    out_channels=num_feature_channels,
                    strides=1,
                    kernel_size=1,
                    padding=0,
                    conv_only=True,
                ),
                # transformer layers
                SpatialTransformer(
                    spatial_dims=spatial_dims,
                    in_channels=num_feature_channels,
                    num_layers=num_layers,
                    num_attention_heads=num_feature_channels // num_head_channels,
                    num_head_channels=num_head_channels,
                    dropout=dropout,
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                    upcast_attention=upcast_attention,
                    use_flash_attention=use_flash_attention, 
                ),
                nn.GroupNorm(num_groups=norm_num_groups, num_channels=num_feature_channels, eps=norm_eps, affine=True),
                nn.SiLU(),
                # to in_channels
                Convolution(
                    spatial_dims=spatial_dims,
                    in_channels=num_feature_channels,
                    out_channels=in_channels,
                    strides=1,
                    kernel_size=1,
                    padding=0,
                    conv_only=True,
                ),
            )
        return nn.Sequential(
            Convolution(
                spatial_dims=spatial_dims,
                in_channels=in_channels,
                out_channels=num_feature_channels,
                strides=1,
                kernel_size=1,
                padding=0,
                conv_only=True,
            ),
            Permute(0, 2, 1),
            *[
                BasicTransformerBlock(
                    num_channels=num_feature_channels,
                    num_attention_heads=num_feature_channels // num_head_channels,
                    num_head_channels=num_head_channels,
                    dropout=dropout,
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                    upcast_attention=upcast_attention,
                    use_flash_attention=use_flash_attention,
                )
                for _ in range(num_layers)
            ],
            Permute(0, 2, 1),
            nn.GroupNorm(num_groups=norm_num_groups, num_channels=num_feature_channels, eps=norm_eps, affine=True),
            nn.SiLU(),
            Convolution(
                spatial_dims=spatial_dims,
                in_channels=num_feature_channels,
                out_channels=in_channels,
                strides=1,
                kernel_size=1,
                padding=0,
                conv_only=True,
            ),
        )     
    return nn.Identity()

class Permute(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.args = args
    def forward(self, x):
        return x.permute(*self.args)