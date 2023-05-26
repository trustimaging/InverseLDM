import logging

from .autoencoder_trainer import AutoencoderTrainer
from .diffusion_trainer import DiffusionTrainer

from seismic.utils import _instance_conditioner
from datasets.utils import (_instance_dataset, _instance_dataloader,
                            _split_valid_dataset)


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

        # Data conditioning
        try:
            self.data_conditioner = _instance_conditioner(args.seismic)
        except KeyError:
            self.data_conditioner = None
        if self.data_conditioner:
            self.data_conditioner.process()

        # Model trainers
        self.autoencoder = AutoencoderTrainer(
            args=args.autoencoder,
            args_run=args.run,
            args_logging=args.logging,
            train_loader=self.autoencoder_train_dataloader,
            valid_loader=self.autoencoder_valid_dataloader,
        )

        self.diffusion = DiffusionTrainer(
            autoencoder=self.autoencoder.model.module.model,
            args=args.diffusion,
            args_run=args.run,
            args_logging=args.logging,
            data_conditioner=self.data_conditioner,
            train_loader=self.diffusion_train_dataloader,
            valid_loader=self.diffusion_valid_dataloader,
        )

    def train(self):
        logging.info(" ---- Autoencoder Training ---- ")
        self.autoencoder.train()
        logging.info(" ---- Diffusion Training ---- ")
        self.diffusion.train()
