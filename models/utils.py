import os
import inspect
import torch
import torch.nn as nn
import torch.optim as optim

from .diffusion_model import DiffusionWrapper
from .autoencoder_model import AutoencoderWrapper, GaussianDistribution
from .losses import divergence, perceptual, reconstruction


def _instance_autoencoder_model(args, device):
    return AutoencoderWrapper(args, device)


def _instance_diffusion_model(autoencoder, args, device):
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
    if args.optim.bias_weight_decay is not None:
        decay_params, no_decay_params = [], []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue           
            elif len(param.shape) == 1 or name.endswith(".bias"):
                no_decay_params.append(param)
            else: decay_params.append(param)
        params = [{'params': no_decay_params, 'weight_decay': args.optim.bias_weight_decay}, {'params': decay_params, 'weight_decay': kwargs.pop("weight_decay")}]        
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
    def autoencoder_loss_fn(input: torch.Tensor,
                            recon: torch.Tensor,
                            mean: torch.Tensor,
                            log_var: torch.Tensor) -> torch.Tensor:
        loss_div = divergence(args, mean, log_var).mean()
        loss_recon = reconstruction(args, input, recon).mean()
        loss_percep = perceptual(args, input, recon).mean()
        loss = args.params.recon_weight * loss_recon + \
            args.params.div_weight * loss_div + \
            args.params.perceptual_weight * loss_percep
        return loss
    return autoencoder_loss_fn


def _instance_diffusion_loss_fn(args):
    def diffusion_loss_fn(target: torch.Tensor,
                          pred: torch.Tensor) -> torch.Tensor:
        loss = reconstruction(args, target, pred)
        return loss
    return diffusion_loss_fn


class DataParallelCPU(nn.Module):
    def __init__(self, module, **kwargs):
        super().__init__()
        self.module = module

    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)


def data_parallel_wrapper(module, device, **kwargs): 
    """
    Ensures code consistency for DataParallel when cpu is used.
    """
    if device == "cpu":
        return DataParallelCPU(module)
    else:
        return nn.DataParallel(module, **kwargs)

        
        

# def _instance_diffusion_sampler(ldm, args):
#     if args.params.sampler.lower() == "ddim":
#         return DDIMSampler(model=ldm,
#                            n_steps=args.params.num_diffusion_timesteps,
#                            ddim_discretize="uniform",
#                            ddim_eta=0.)
#     else:
#         raise NotImplementedError(f"{args.params.sampler} not implemented.")
