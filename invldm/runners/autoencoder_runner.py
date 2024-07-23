import os

import torch
import torch.nn as nn
import logging

from functools import partial
from typing import Optional

from . import BaseRunner
from .utils import _instance_optimiser, _instance_lr_scheduler, set_requires_grad, _instance_autoencoder_loss_fn, _instance_discriminator_loss_fn

from .losses import FocalFrequencyLoss

from wiener_loss import WienerLoss

from generative.losses.perceptual import PerceptualLoss
from generative.losses.adversarial_loss import PatchAdversarialLoss
from generative.networks.nets import AutoencoderKL, PatchDiscriminator

from ..utils.utils import namespace2dict, filter_kwargs_by_class_init, scale2range


class AutoencoderRunner(BaseRunner):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        model_kwargs = filter_kwargs_by_class_init(AutoencoderKL, namespace2dict(self.args.model))
        self.model = AutoencoderKL(**model_kwargs).to(self.device)

        if not self.args.sampling_only:

            # Optimiser and scaler
            self.optimiser = torch.optim.Adam(self.model.parameters(), self.args.optim.lr, weight_decay=self.args.optim.weight_decay, betas=self.args.optim.betas)
            self.scaler = torch.amp.GradScaler(self.device)

            # Loss weights
            self.recon_weight = self.args.params.recon_weight
            self.kl_weight = self.args.params.kl_weight
            self.perceptual_weight = self.args.params.perceptual_weight
            self.wiener_weight=self.args.params.wiener_weight
            self.adversarial_weight = self.args.params.adversarial_weight

            # Loss instances
            if self.recon_weight > 0.:
                self.recon_loss_fn = torch.nn.L1Loss() if self.args.params.recon_loss.lower() == "l1" else torch.nn.MSELoss()

            if self.perceptual_weight > 0.:
                if self.args.params.perceptual_loss == "lpips":
                    self.perceptual_loss_fn = PerceptualLoss(
                        spatial_dims=self.args.model.spatial_dims,
                        network_type=self.args.params.lpips_model
                    ).to(self.device)
                if self.args.params.perceptual_loss == "ffl":
                    self.perceptual_loss_fn = FocalFrequencyLoss()
            
            if self.wiener_weight > 0:
                if self.args.params.wiener_penalty == "laplace":
                    penalty_function = partial(laplace2D, alpha=self.args.params.wiener_laplace_alpha, beta=self.args.params.wiener_laplace_beta)
                elif self.args.params.wiener_penalty in ["identity", "trainable"]:
                    penalty_function = self.args.params.wiener_penalty
                else:
                    raise AttributeError(f"Penalty function {self.args.params.wiener_penalty} is not avaiable. [laplace, identity]")
                
                x = next(iter(self.train_loader))
                if isinstance(x, (list, tuple)):
                    x, _ = x

                self.wiener_loss_fn = WienerLoss(
                    method="fft",
                    filter_dim=self.args.params.wiener_filter_dim,
                    filter_scale=self.args.params.wiener_filter_scale,
                    mode=self.args.params.wiener_mode,
                    epsilon=self.args.params.wiener_epsilon,
                    penalty_function=penalty_function,
                    store_filters=False,
                    clamp_min=self.args.params.wiener_clamp_min,
                    corr_norm=self.args.params.wiener_corr_norm,
                    rel_epsilon=self.args.params.wiener_rel_epsilon,
                    input_shape=x[0].shape,
                )
                if penalty_function == "trainable":
                    self.optimiser.add_param_group(dict(params=self.wiener_loss_fn.parameters(), lr=100 / (2*(self.args.training.n_epochs) * len(self.train_loader))))
            
            if self.adversarial_weight > 0:
                self.discriminator = PatchDiscriminator(
                    spatial_dims=self.args.model.spatial_dims,
                    num_layers_d=self.args.params.disc_n_layers,
                    num_channels=self.args.params.disc_feature_channels,
                    in_channels=self.args.model.in_channels,
                    out_channels=self.args.model.out_channels,
                ).to(self.device)

                self.adversarial_loss_fn = PatchAdversarialLoss(criterion=self.args.params.adversarial_mode)
            
                # Discriminator optimser and scaler
                self.optimiser_d = torch.optim.Adam(self.discriminator.parameters(), self.args.optim.d_lr, weight_decay=self.args.optim.weight_decay, betas=self.args.optim.betas)
                self.scaler_d = torch.amp.GradScaler(self.device)

    def train_step(self, input, **kwargs):
        self.model.train()

        if "discriminator" in self.__dict__:
            self.discriminator.train()

        self.optimiser.zero_grad(set_to_none=True)

        # Train G
        with torch.amp.autocast(str(self.device)):
            loss = 0.
            recon, z_mu, z_sigma = self.model(input)

            if self.recon_weight > 0.:
                r_loss = self.recon_loss_fn(recon.float(), input.float())
                loss += self.recon_weight * r_loss 

            if self.perceptual_weight > 0.:
                p_loss = self.perceptual_loss_fn(recon.float(), input.float())
                loss += self.perceptual_weight * p_loss

            if self.kl_weight > 0.:
                kl_loss = 0.5 * torch.sum(z_mu.pow(2) + z_sigma.pow(2) - torch.log(z_sigma.pow(2)) - 1, dim=[1, 2, 3])
                kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]
                loss += self.kl_weight * kl_loss

            if self.wiener_weight > 0:
                w_loss = self.wiener_loss_fn(input.float(), recon.float(), gamma=self.args.params.wiener_gamma)
                loss += self.wiener_weight * w_loss

            if (self.adversarial_weight) > 0. and (self.epoch > self.args.training.warm_up_epochs):
                logits_fake = self.discriminator(recon.contiguous().float())[-1]
                generator_loss = self.adversarial_loss_fn(logits_fake, target_is_real=True, for_discriminator=False)
                loss += self.adversarial_weight * generator_loss

        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimiser)
        self.scaler.update()

        # Train D
        if (self.adversarial_weight) > 0. and (self.epoch > self.args.training.warm_up_epochs):
            with torch.amp.autocast(str(self.device)):
                self.optimiser_d.zero_grad(set_to_none=True)

                logits_fake = self.discriminator(recon.contiguous().detach())[-1]
                loss_d_fake = self.adversarial_loss_fn(logits_fake, target_is_real=False, for_discriminator=True)
                logits_real = self.discriminator(input.contiguous().detach())[-1]
                loss_d_real = self.adversarial_loss_fn(logits_real, target_is_real=True, for_discriminator=True)
                discriminator_loss = (loss_d_fake + loss_d_real) * 0.5

                loss_d = self.adversarial_weight * discriminator_loss

            self.scaler_d.scale(loss_d).backward()
            self.scaler_d.step(self.optimiser_d)
            self.scaler_d.update()
        else:
            loss_d = torch.zeros_like(loss)

        # Output dictionary
        output = {
            "loss": loss,
            "recon": recon,
            "mean": z_mu,
            "log_var": z_sigma,
            "loss_d": loss_d,
        }
        return output
    
    @torch.no_grad()
    def valid_step(self, input, **kwargs):
        # Forward pass: recon and the statistical posterior
        self.model.eval()

        if "discriminator" in self.__dict__:
            self.discriminator.eval()

        with torch.amp.autocast(str(self.device)):
            loss = 0.
            recon, z_mu, z_sigma = self.model(input)

            if self.recon_weight > 0.:
                r_loss = self.recon_loss_fn(recon.float(), input.float())
                loss += self.recon_weight * r_loss 

            if self.perceptual_weight > 0.:
                p_loss = self.perceptual_loss_fn(recon.float(), input.float())
                loss += self.perceptual_weight * p_loss

            if self.kl_weight > 0.:
                kl_loss = 0.5 * torch.sum(z_mu.pow(2) + z_sigma.pow(2) - torch.log(z_sigma.pow(2)) - 1, dim=[1, 2, 3])
                kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]
                loss += self.kl_weight * kl_loss

            if self.wiener_weight > 0:
                w_loss = self.wiener_loss_fn(input.float(), recon.float(), gamma=self.args.params.wiener_gamma)
                loss += self.wiener_weight * w_loss

            if self.adversarial_weight > 0.:
                logits_fake = self.discriminator(recon.contiguous().float())[-1]
                generator_loss = self.adversarial_loss_fn(logits_fake, target_is_real=True, for_discriminator=False)
                loss += self.adversarial_weight * generator_loss

                logits_fake = self.discriminator(recon.contiguous().detach())[-1]
                loss_d_fake = self.adversarial_loss_fn(logits_fake, target_is_real=False, for_discriminator=True)
                logits_real = self.discriminator(input.contiguous().detach())[-1]
                loss_d_real = self.adversarial_loss_fn(logits_real, target_is_real=True, for_discriminator=True)
                discriminator_loss = (loss_d_fake + loss_d_real) * 0.5

                loss_d = self.adversarial_weight * discriminator_loss
            else:
                loss_d = torch.zeros_like(loss)

        # Output dictionary
        output = {
            "loss": loss,
            "recon": recon,
            "mean": z_mu,
            "log_var": z_sigma,
            "loss_d": loss_d,
        }
        return output
    
    @torch.no_grad()
    def sample_step(self, input, **kwargs):
        mu = torch.zeros_like(input)
        sigma = torch.ones_like(input)
        z = self.model.sampling(mu, sigma)
        sample = self.model.decode(z)
        return sample, z



def laplace2D(mesh, alpha=-0.2, beta=1.5):
    """ Helper function for AWLoss """
    xx, yy = mesh[:,:,0], mesh[:,:,1]
    x = torch.sqrt(xx**2 + yy**2) 
    T = 1 - torch.exp(-torch.abs(x) ** alpha) ** beta
    T = scale2range(T, [0.25, 1.])
    return T