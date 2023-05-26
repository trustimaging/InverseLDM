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

    # instance optimiser
    optimiser = cls(model.parameters(), **kwargs)
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
    def diffusion_loss_fn(targe_noise: torch.Tensor,
                          pred_noise: torch.Tensor) -> torch.Tensor:
        loss = reconstruction(args, targe_noise, pred_noise)
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
        # os.environ['MASTER_ADDR'] = 'localhost'
        # os.environ['MASTER_PORT'] = '12355'
        # torch.distributed.init_process_group(rank=0, world_size=2)
        # return nn.parallel.DistributedDataParallel(module, **kwargs)
        return nn.DataParallel(module, **kwargs)

        
        

# def _instance_diffusion_sampler(ldm, args):
#     if args.params.sampler.lower() == "ddim":
#         return DDIMSampler(model=ldm,
#                            n_steps=args.params.num_diffusion_timesteps,
#                            ddim_discretize="uniform",
#                            ddim_eta=0.)
#     else:
#         raise NotImplementedError(f"{args.params.sampler} not implemented.")
