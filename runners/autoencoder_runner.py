import torch
import torch.nn as nn

from . import BaseRunner
from models.utils import (_instance_autoencoder_model, _instance_optimiser,
                          _instance_autoencoder_loss_fn, _instance_lr_scheduler,
                          data_parallel_wrapper)


class AutoencoderRunner(BaseRunner):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.model = _instance_autoencoder_model(self.args, self.device)
        self.model = data_parallel_wrapper(module=self.model,
                                           device=self.device,
                                           device_ids=self.gpu_ids)

        # If not in sampling only mode, instantiate optimising objects
        if not self.args.sampling.sampling_only: 
            assert (self.train_loader is not None), " Train data loader is required in training mode, but got None"
            self.optimiser = _instance_optimiser(self.args, self.model)
            self.lr_scheduler = _instance_lr_scheduler(self.args, self.optimiser)
            self.loss_fn = _instance_autoencoder_loss_fn(self.args)

    def train_step(self, input, **kwargs):
        # Forward pass: recon and the statistical posterior
        recon, mean, log_var = self.model(input)

        # Compute training loss
        loss = self.loss_fn(input, recon, mean, log_var)

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
            "recon": recon,
            "mean": mean,
            "log_var": log_var
        }
        return output

    def valid_step(self, input, **kwargs):
        # Forward pass: recon and the statistical posterior
        recon, mean, log_var = self.model(input)

        # Compute validation loss
        loss = self.loss_fn(input, recon, mean, log_var)

        # Output dictionary
        output = {
            "loss": loss,
            "recon": recon,
            "mean": mean,
            "log_var": log_var
        }
        return output
    
    # def sample(self, n_samples=None, **kwargs):
    #      # One autoencoder forward pass to get shape of latent space -- can be optimised!
    #     ch, h, w = self.train_loader.dataset.dataset[0][0].shape
    #     _, _ = self.model.module.model.encode(torch.randn(n_samples, ch, h, w,
    #                                                    device=self.model.module.device))
    #     z = self.model.module.model.sample()
        
    #     # Sample N(0, I) in latent space and decode
    #     sample = self.model.module.model.decode(torch.randn_like(z))
    #     return sample
    
    def sample_step(self, input, **kwargs):
         # One autoencoder forward pass to get shape of latent space -- can be optimised!
        _, _ = self.model.module.model.encode(input)
        z = self.model.module.model.sample()
        
        # Sample N(0, I) in latent space and decode
        sample = self.model.module.model.decode(torch.randn_like(z))
        return sample    


