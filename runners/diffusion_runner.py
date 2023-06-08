import torch
import torch.nn as nn

from . import BaseRunner

from models.utils import (_instance_diffusion_model, _instance_optimiser,
                          _instance_diffusion_loss_fn, _instance_lr_scheduler,
                          data_parallel_wrapper)


class DiffusionRunner(BaseRunner):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        self.data_conditioner = kwargs.pop("data_conditioner", None)

        autoencoder = kwargs.pop("autoencoder").eval()
        self.model = _instance_diffusion_model(autoencoder,
                                               self.args,
                                               self.device)
        self.model = data_parallel_wrapper(module=self.model,
                                           device=self.device,
                                           device_ids=self.gpu_ids)

        self.optimiser = _instance_optimiser(self.args, self.model)
        self.lr_scheduler = _instance_lr_scheduler(self.args, self.optimiser)
        self.loss_fn = _instance_diffusion_loss_fn(self.args)

    def train_step(self, input, **kwargs):
        # Forward pass: predict model noise based on condition
        noise, noise_pred = self.model(input)

        # Compute training loss
        loss = self.loss_fn(noise, noise_pred)

        # Zero grad and back propagation
        self.optimiser.zero_grad()
        loss.backward()

        # Gradient Clipping
        if self.args.optim.grad_clip:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                           self.args.optim.grad_clip)

        # Update gradients
        self.optimiser.step()

        # Update lr scheduler
        if self.lr_scheduler:
            self.lr_scheduler.step()

        # Output dictionary
        output = {
            "loss": loss,
        }
        return output

    def valid_step(self, input, **kwargs):
        # Forward pass: predict model noise based on condition
        noise, noise_pred = self.model(input)

        # Compute validation loss
        loss = self.loss_fn(noise, noise_pred)

        # Output dictionary
        output = {
            "loss": loss,
        }
        return output
    
    # @torch.no_grad()
    # def sample(self, n_samples=None, **kwargs):
    #     # One autoencoder forward pass to get shape of latent space -- can be optimised!
    #     ch, h, w = self.train_loader.dataset.dataset[0][0].shape
    #     z = self.model.module.ldm.autoencoder_encode(torch.randn(n_samples, ch, h, w,
    #                                                       device=self.model.module.device))

    #     # Sample latent space with diffusion model and decode to reconstruct image
    #     z_sample = self.model.module.sampler.sample(shape=z.shape,
    #                                                 cond=None)
    #     sample = self.model.module.ldm.autoencoder_decode(z_sample)
    #     return sample

    @torch.no_grad()
    def sample_step(self, input, **kwargs):
        # One autoencoder forward pass to get shape of latent space -- can be optimised!
        z = self.model.module.ldm.autoencoder_encode(input)

        # Sample latent space with diffusion model and decode to reconstruct image
        z_sample = self.model.module.sampler.sample(shape=z.shape,
                                                    cond=None)
        sample = self.model.module.ldm.autoencoder_decode(z_sample)
        return sample
