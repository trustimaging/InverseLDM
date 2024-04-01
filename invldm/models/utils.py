import os
import inspect
import torch
import torch.nn as nn
import torch.optim as optim

from ..utils.utils import scale2range

from .diffusion_model import DiffusionWrapper
from .autoencoder_model import AutoencoderWrapper, GaussianDistribution
from .losses import _divergence_fn, _perceptual_fn, _reconstruction_fn


def _instance_autoencoder_model(args, device="cpu"):
    return AutoencoderWrapper(args, device)


def _instance_diffusion_model(autoencoder, args, device="cpu"):
    return DiffusionWrapper(autoencoder, args, device)


def _instance_optimiser(args, model):
    # Get optim class
    cls = getattr(optim, args.optim.optimiser)

    # Select only valid kwargs
    valid_kwargs = inspect.getfullargspec(cls).args
    kwargs = {}
    for vk in valid_kwargs:
        try:
            kwargs[vk] = args.optim.__dict__[vk]
        except KeyError:
            pass

    # change weight decay of biases to zero if prompted
    if not args.optim.bias_weight_decay:
        decay_params, no_decay_params = [], []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue           
            elif len(param.shape) == 1 or name.endswith(".bias"):
                no_decay_params.append(param)
            else: decay_params.append(param)
        params = [{'params': no_decay_params, 'weight_decay': 0.}, {'params': decay_params, 'weight_decay': kwargs.pop("weight_decay")}]        
    else:
        params=model.parameters()

    # instance optimiser
    optimiser = cls(params, **kwargs)
    return optimiser


def _instance_lr_scheduler(args, optimiser):
    if args.optim.lr_scheduler.scheduler:
        cls = getattr(optim.lr_scheduler,
                      args.optim.lr_scheduler.scheduler)

        # Select only valid kwargs
        valid_kwargs = inspect.getfullargspec(cls).args
        kwargs = {}
        for vk in valid_kwargs:
            try:
                kwargs[vk] = args.optim.lr_scheduler.__dict__[vk]
            except KeyError:
                pass

        scheduler = cls(optimiser, **kwargs)
        return scheduler
    else:
        return None


def _instance_autoencoder_loss_fn(args):
    # Get correct loss functions from args
    div_fn = _divergence_fn(args)
    recon_fn = _reconstruction_fn(args)
    percep_fn = _perceptual_fn(args)

    # Function to combine all losses with their respective weights 
    def autoencoder_loss_fn(input: torch.Tensor,
                            recon: torch.Tensor,
                            mean: torch.Tensor,
                            log_var: torch.Tensor) -> torch.Tensor:
        
        # Evaluate divergence and reconstruction loss functions
        loss_div = div_fn(mean, log_var).mean()
        loss_recon = recon_fn(input, recon).mean()

        # LPIPS loss requires inputs in range [-1, 1]
        if args.model.perceptual_loss == "lpips":
            input, recon = scale2range(input, [-1, 1]), scale2range(recon, [-1, 1])
        
        # Evaluate perceptual loss
        loss_percep = percep_fn.to(input.device)(input, recon).mean()

        # Combine losses
        loss = args.params.recon_weight * loss_recon + \
            args.params.div_weight * loss_div + \
            args.params.perceptual_weight * loss_percep
        return loss
    return autoencoder_loss_fn


def _instance_diffusion_loss_fn(args):
    diffusion_loss_fn = _reconstruction_fn(args)    
    return diffusion_loss_fn


def data_parallel_wrapper(module, device, **kwargs): 
    """
    Ensures code consistency for DataParallel when cpu is used.
    """
    if device == "cpu":
        return DataParallelCPU(module)
    else:
        return nn.DataParallel(module, **kwargs)
    

class DataParallelCPU(nn.Module):
    def __init__(self, module, **kwargs):
        super().__init__()
        self.module = module

    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)
