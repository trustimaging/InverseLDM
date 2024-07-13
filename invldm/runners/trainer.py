import torch
import logging

from functools import partial
from torchsummary import summary

from ..datasets.utils import (_instance_dataset, _instance_dataloader,
                            _split_valid_dataset)

from ..runners import AutoencoderRunner, DiffusionRunner


class Trainer():
    def __init__(self, args):
        self.args = args

        # Datasets
        self.dataset = _instance_dataset(self.args.data)
        self.autoencoder_train_dataset, self.autoencoder_valid_dataset = \
            _split_valid_dataset(args.autoencoder, self.dataset)
        self.diffusion_train_dataset, self.diffusion_valid_dataset = \
            _split_valid_dataset(args.diffusion, self.dataset)

        # Dataloaders
        self.autoencoder_train_dataloader = _instance_dataloader(
            self.args.autoencoder.training, self.autoencoder_train_dataset
        )
        self.autoencoder_valid_dataloader = _instance_dataloader(
            self.args.autoencoder.validation, self.autoencoder_valid_dataset
        )
        self.diffusion_train_dataloader = _instance_dataloader(
            self.args.diffusion.training, self.diffusion_train_dataset
        )
        self.diffusion_valid_dataloader = _instance_dataloader(
            self.args.diffusion.validation, self.diffusion_valid_dataset
        )
        

        # Model trainers
        self.autoencoder_runner = AutoencoderRunner(
            args=args.autoencoder,
            args_run=args.run,
            args_logging=args.logging,
            train_loader=self.autoencoder_train_dataloader,
            valid_loader=self.autoencoder_valid_dataloader,
        )

        self.diffusion_runner = DiffusionRunner(
            args=args.diffusion,
            args_run=args.run,
            args_logging=args.logging,
            autoencoder=self.autoencoder_runner.model,
            spatial_dims=args.autoencoder.model.spatial_dims,
            latent_channels=args.autoencoder.model.latent_channels,
            train_loader=self.diffusion_train_dataloader,
            valid_loader=self.diffusion_valid_dataloader,
        )


    def train(self):
        logging.info(" ---- Dataset ---- ")
        logging.info(self.dataset)

        logging.info(" ---- Model - Autoencoder ----")
        sample = self.dataset[0]
        if isinstance(sample, tuple):
            sample, cond = sample
        sample = sample.to(self.autoencoder_runner.device)
        logging.info(summary(model=self.autoencoder_runner.model, input_data=sample.shape, device=self.autoencoder_runner.device))

        logging.info(" ---- Model - Diffusion ----")
        if self.args.diffusion.training.n_epochs > 0:
            with torch.no_grad():
                mu, sigma = self.diffusion_runner.autoencoder.encode(sample.unsqueeze(0).float())
                z = self.diffusion_runner.autoencoder.sampling(mu, sigma)
                t = torch.tensor([0]).repeat(z.shape[0])                

                if cond is not None:
                    c = self.diffusion_runner.cond_proj(cond.unsqueeze(0).float().to(z.device))
                    if self.args.diffusion.model.condition.mode == "concat":
                        c = torch.nn.functional.interpolate(c, z.shape[2:])
                        z = torch.concat([z, c], dim=1)
                        input_data = (z, t)
                    elif self.args.diffusion.model.condition.mode == "crossattn":
                        c = c.flatten(start_dim=2)
                        input_data = (z, t, c)
                    else:
                        input_data = (z, t)

            logging.info(summary(model=self.diffusion_runner.model, input_data=input_data, device=self.diffusion_runner.device))
            logging.info(f"\n\nLatent size: {z.shape[1:]}")
            if "c" in locals():
                logging.info(f"\nCondition (latent) size: {c.shape[1:]}\n\n")
            
        
        logging.info(" ---- Autoencoder Training ---- ")
        self.autoencoder_runner.train()
        
        del self.autoencoder_runner.discriminator
        del self.autoencoder_runner.perceptual_loss_fn

        if self.args.diffusion.training.n_epochs > 0:
            logging.info(" ---- Diffusion Training ---- ")
            self.diffusion_runner.train()

        logging.info(" ---- Training Concluded without Errors ---- ")

