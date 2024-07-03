import os

import torch
import torch.nn as nn
import logging

from typing import Optional

from . import BaseRunner

from generative.inferers import LatentDiffusionInferer
from generative.networks.nets import DiffusionModelUNet
from generative.networks.schedulers import DDPMScheduler, DDIMScheduler, PNDMScheduler

from ..utils.utils import namespace2dict, filter_kwargs_by_class_init


class DiffusionRunner(BaseRunner):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.autoencoder = kwargs.pop("autoencoder").eval()
        self.spatial_dims = kwargs.pop("spatial_dims")
        self.latent_channels = kwargs.pop("latent_channels")
        
        try:
            self.num_inference_timesteps = self.args.params.num_inference_timesteps
        except AttributeError:
            self.num_inference_timesteps = self.args.params.num_train_timesteps

        model_kwargs = filter_kwargs_by_class_init(DiffusionModelUNet, namespace2dict(self.args.model))
        _ = [model_kwargs.pop(key, None) for key in ["spatial_dims", "in_channels", "out_channels"]]
        self.model = DiffusionModelUNet(
            spatial_dims=self.spatial_dims,
            in_channels=self.latent_channels,
            out_channels=self.latent_channels,
            **model_kwargs
        ).to(self.device)

        if self.args.params.sampler == "ddim":
            scheduler_args = filter_kwargs_by_class_init(DDIMScheduler, namespace2dict(self.args.params))
            self.scheduler = DDIMScheduler(**scheduler_args)
        elif self.args.params.sampler == "ddpm":
            scheduler_args = filter_kwargs_by_class_init(DDPMScheduler, namespace2dict(self.args.params))
            self.scheduler = DDPMScheduler(**scheduler_args)
        elif self.args.params.sampler == "pndm":
            scheduler_args = filter_kwargs_by_class_init(PNDMScheduler, namespace2dict(self.args.params))
            self.scheduler = PNDMScheduler(**scheduler_args)

        if not self.args.sampling_only: 
            with torch.no_grad():
                with torch.amp.autocast(str(self.device)):
                    x = next(iter(self.train_loader))
                    z = self.autoencoder.encode_stage_2_inputs(x.float().to(self.device))
            logging.info(f"Scaling factor set to {1/torch.std(z)}")
            self.scale_factor = 1 / torch.std(z)
        else:
            self.scale_factor = 1.

        self.inferer = LatentDiffusionInferer(self.scheduler, scale_factor=self.scale_factor)

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

            timesteps = torch.randint(0, self.inferer.scheduler.num_train_timesteps, (z.shape[0],), device=z.device).long()
            noise_pred = self.inferer(inputs=input, diffusion_model=self.model, noise=noise, timesteps=timesteps, autoencoder_model=self.autoencoder)
            
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

            timesteps = torch.randint(0, self.inferer.scheduler.num_train_timesteps, (z.shape[0],), device=z.device).long()
            noise_pred = self.inferer(inputs=input, diffusion_model=self.model, noise=noise, timesteps=timesteps, autoencoder_model=self.autoencoder)
            
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

        # Set number of inference steps for scheduler
        self.scheduler.set_timesteps(num_inference_steps=num_inference_steps)

        # Sample latent space with diffusion model
        logging.info("Sampling...")
        with torch.amp.autocast(str(self.device)):
            samples = self.inferer.sample(
                input_noise=z, diffusion_model=self.model, scheduler=self.scheduler, autoencoder_model=self.autoencoder,
                **kwargs
            )

        return samples, z
