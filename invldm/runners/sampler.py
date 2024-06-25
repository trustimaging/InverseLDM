import torch
import logging

from functools import partial
from torchsummary import summary

from ..datasets.utils import (_instance_dataset, _instance_dataloader)

from ..runners import AutoencoderRunner, DiffusionRunner


class Sampler():
    def __init__(self, args):
        self.args = args

        # Datasets
        self.autoencoder_sampling_dataset = _instance_dataset(
           self.args.data, n_samples=self.args.autoencoder.sampling.n_samples,
        )
        self.diffusion_sampling_dataset = _instance_dataset(
           self.args.data, n_samples=self.args.diffusion.sampling.n_samples,
        )

        # Dataloaders
        self.autoencoder_sample_dataloader = _instance_dataloader(
            self.args.autoencoder.sampling, self.autoencoder_sampling_dataset
        )
        self.diffusion_sample_dataloader = _instance_dataloader(
            self.args.diffusion.sampling, self.diffusion_sampling_dataset
        )

        # Autoencoder runner, load pre-trained, eval mode
        assert args.autoencoder.sampling_only
        self.autoencoder_runner = AutoencoderRunner(
            args=args.autoencoder,
            args_run=args.run,
            args_logging=args.logging,
            sample_loader=self.autoencoder_sample_dataloader
        )
        self.autoencoder_runner.load_checkpoint(
            self.autoencoder_runner.get_checkpoint_path(),
            model_only=True
        )
        self.autoencoder_runner.model.eval()

        # Diffusion runner, load pre-trained, eval mode
        assert args.diffusion.sampling_only
        self.diffusion_runner = DiffusionRunner(
            args=args.diffusion,
            args_run=args.run,
            args_logging=args.logging,
            autoencoder=self.autoencoder_runner.model,
            spatial_dims=args.autoencoder.model.spatial_dims,
            latent_channels=args.autoencoder.model.latent_channels,
            sample_loader=self.diffusion_sample_dataloader
        )
        self.diffusion_runner.load_checkpoint(
            self.diffusion_runner.get_checkpoint_path(),
            model_only=True
        )
        self.diffusion_runner.model.eval()


    def sample(self):
        if self.args.autoencoder.sampling.enable:
            logging.info(" ---- Autoencoder Sampling ---- ")
            self.autoencoder_runner.sample()

        if self.args.diffusion.sampling.enable:
            logging.info(" ---- Diffusion Sampling ---- ")
            self.diffusion_runner.sample()
        logging.info(" ---- Sampling Concluded without Errors ---- ")

