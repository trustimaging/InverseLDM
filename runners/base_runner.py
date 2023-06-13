import os
import time
import fnmatch

import torch
import numpy as np
import logging
import matplotlib.pyplot as plt

from abc import ABC, abstractmethod
from typing import Optional

from utils.utils import namespace2dict, gpu_diagnostics
from utils.visualisation import visualise_samples

torch.set_printoptions(sci_mode=False)


class BaseRunner(ABC):
    def __init__(self, **kwargs) -> None:

        self.args = kwargs.pop("args")
        self.logging_args = kwargs.pop("args_logging")
        self.run_args = kwargs.pop("args_run")

        self.train_loader = kwargs.pop("train_loader", None)
        self.valid_loader = kwargs.pop("valid_loader", None)
        self.sample_loader = kwargs.pop("sample_loader", None)

        self.device = self.run_args.device
        self.gpu_ids = self.run_args.gpu_ids

        self.model = None
        self.optimiser = None
        self.loss_fn = None
        self.lr_scheduler = None

        self.epoch = 1
        self.steps = 1

        self.init_steps = 0
        self.n_steps = self.get_total_round_steps()

        self.hparam_dict = {}

        self.completed = False

    @abstractmethod
    def train_step(self, input: torch.Tensor, **kwargs) -> dict:
        """ Function performs one batch training step. Input is a batch for training """
        return NotImplementedError

    @abstractmethod
    @torch.no_grad()
    def valid_step(self, input: torch.Tensor, **kwargs) -> dict:
        """ Function performs one batch validation step. Input is a batch for validating """
        return NotImplementedError
    
    # @torch.no_grad()
    # def sample(self, n_samples=None, **kwargs) -> dict:
    #     """ Function performs sampling of self.model. n_samples is the number of samples to generate"""
    #     return NotImplementedError

    @torch.no_grad()
    def sample_step(self, input: torch.Tensor, **kwargs) -> torch.Tensor:
        """ Function performs sampling of self.model. n_samples is the number of samples to generate"""
        return NotImplementedError

    def _update_hparam_dict(self) -> dict:
        hparam_dict = {
            "name": self.args.name,
            "dataset": str(self.train_loader.dataset.dataset),
            "device": self.device,
            "gpus": torch.cuda.get_device_name(self.device) if "cuda" in self.device else None
        }
        hparam_dict.update(namespace2dict(self.args, flatten=True))
        self.hparam_dict = hparam_dict
        return None

    def save_checkpoint(self, path: str = "") -> None:
        states = {
                "model_state_dict": self.model.state_dict(),
                "optimiser_state_dict": self.optimiser.state_dict(),
                "lr_scheduler": self.lr_scheduler,
                "epoch": self.epoch,
                "step": self.steps,
            }
        if not path:
            latest_ckpt_name = f"{self.args.name.lower()}_ckpt_latest.pth"
            path = os.path.join(self.args.ckpt_path, latest_ckpt_name)

        torch.save(states, path)
        logging.info(f"Saved {self.args.name.lower()} checkpoint {path}")
        return None

    def load_checkpoint(self, path: str = "", model_only: Optional[bool] = False) -> None:
        if not path:
            try:
                latest_ckpt_name = [f for f in os.listdir(self.args.ckpt_path) if fnmatch.fnmatch(f, "*latest*")][0]
                path = os.path.join(self.args.ckpt_path, latest_ckpt_name)
            except IndexError as e:
                if self.args.sampling.sampling_only:
                    logging.critical(f"Could not find latest {self.args.name.lower()} model. Please specify checkpoint path to load.")
                    raise IndexError(e)
                else:
                    if self.run_args.y:
                        return None
                    else:
                        logging.critical(f"Could not find latest {self.args.name.lower()} model.")
                        user_input = input("\tProceed from scratch? (Y/N): ")
                        if user_input.lower() == "y" or user_input.lower() == "yes":
                            return None
                        else:
                            logging.critical(f"Aborting...")
                            raise IndexError(e)

        logging.info(f"Loading {self.args.name} checkpoint {path} ...")

        try:
            states = torch.load(path)
        except RuntimeError:
            states = torch.load(path, map_location="cpu")
        self.model.load_state_dict(states["model_state_dict"])
        if not model_only:
            self.optimiser.load_state_dict(states["optimiser_state_dict"])
            self.lr_scheduler = states["lr_scheduler"]
            self.epoch = states["epoch"]
            self.steps = states["step"]

        logging.info(f"{self.args.name.lower().capitalize()} checkpoint successfully loaded.")
        return None
    
    def checkpoint_path(self) -> str:
        # Initialise checkpoint path as None (find latest model)
        ckpt = None

        # Try to find checkpoint file if it's passed as a step integer
        if self.args.model.checkpoint is not None:
            try:
                ckpt_no = int(self.args.model.checkpoint)
                try:
                    ckpt_file = [f for f in os.listdir(self.args.ckpt_path) if fnmatch.fnmatch(f, f"*_step_{ckpt_no}*")][0]
                    ckpt = os.path.join(self.args.ckpt_path, ckpt_file)
                except IndexError:
                    logging.critical(f"Tried to load step {ckpt_no} from {self.args.ckpt_path} but found no file. Loading latest model...")
            except ValueError:
                ckpt = self.args.model.checkpoint
        return ckpt
    
    def resume_training(self) -> None:
        # Load the checkpoint
        ckpt = self.checkpoint_path()
        self.load_checkpoint(ckpt)

        # Prevent running train if it's complete
        if self.epoch >= self.args.training.n_epochs:
            logging.info(f"{self.args.name.lower().capitalize()} training already completed with {self.epoch} epochs")
            self.completed = True

        # Resume training at beginning of current epoch
        self.steps = int((self.epoch - 1) * len(self.train_loader)) + 1

        # Store the initial steps (used for progression logs)
        self.init_steps = self.steps
        self.n_steps = self.get_total_round_steps()
        return None

    def save_figure(self, x: torch.Tensor, mode: str, fig_type: str, save_tensor: Optional[bool] = False) -> None:
        assert mode in ["training", "validation", "sampling", ""]
        assert fig_type in ["input", "recon", "sample", ""]

        if self.args.sampling.sampling_only:
            file_name = f"{self.args.name.lower()}_{mode}_{fig_type}_batch_{self.steps}.png"
            path = os.path.join(self.run_args.samples_folder, file_name)

        else:
            file_name = f"{self.args.name.lower()}_{mode}_{fig_type}_epoch_{self.epoch}_step_{self.steps}.png"
            if fig_type == "recon" or fig_type == "input":
                path = os.path.join(self.args.recon_path, file_name)
            elif fig_type == "sample":
                path = os.path.join(self.args.samples_path, file_name)
            else:
                path = os.path.join(self.args.log_path, file_name)
            
        fig = visualise_samples(x, scale=True)
        plt.savefig(path)
        plt.close(fig)
        logging.info(f"Saved {self.args.name} {mode} {fig_type} figure in {path}")

        if save_tensor:
            path = path[:-3] + "pt"
            torch.save(x, path)
            logging.info(f"Saved {self.args.name} {mode} {fig_type} tensor in {path}")
        return None

    def get_total_round_steps(self) -> int:
        if self.args.sampling.sampling_only:
            if self.sample_loader is not None:
                return len(self.sample_loader)
            else:
                return -1
        else:
            return int((self.args.training.n_epochs) * len(self.train_loader)) - self.init_steps

    def progress(self) -> float:
        return (self.steps - self.init_steps) / self.n_steps

    def eta(self, start_time: float) -> str:
        try:
            progress = self.progress()
            current_time = np.round((time.time() - start_time) / 60, 2)
            expected_total_time = np.round(current_time * (self.n_steps / (self.steps - self.init_steps)), 2)
            if self.n_steps >= 10:
                eta = np.round(expected_total_time - current_time, 2)
                h = int(np.floor(eta / 60))
                m = int(np.floor(eta - h*60))
                s = int(np.floor((eta - h*60 - m)*60))
            else:
                h, m, s = 99, 99, 99
            s = "Progress: {:.2f}%  ETA: {:02d}:{:02d}:{:02d}".format(progress*100, h, m, s)
            return s
        except (RuntimeWarning, ZeroDivisionError, OverflowError, ValueError):
            return ""

    def train(self) -> None:
        assert (not self.args.sampling.sampling_only), " Sampling only flags cannot be passed for training experiment. "

        start_time = time.time()

        # Prepare iterator for validation
        try:
            valid_iterator = iter(self.valid_loader)
        except TypeError:
            valid_iterator = None

        # Adjust start epoch, load checkpoint
        if self.run_args.resume_training:
            self.resume_training()
        start_epoch = self.epoch

        # Fetch tracking logger
        logger = self.logging_args.tracker

        # Update hyperparameter dictionary
        self._update_hparam_dict()

        # Training mode
        self.model.train()

        # Loop through batch and perform training and validation steps
        if not self.completed:
            try:
                for epoch in range(start_epoch, self.args.training.n_epochs + 1):
                    for i, (input, condition) in enumerate(self.train_loader):

                        # Send batch to device
                        input = input.float().to(self.device)

                        # Train batch
                        output = self.train_step(input, condition=condition)
                        try:
                            loss = output["loss"]
                        except KeyError as e:
                            logging.critical("Train step output must be a dictionary containing the key 'loss'")
                            raise KeyError(e)

                        # Log loss
                        logging.info(f"{self.args.name.lower().capitalize()} Epoch {self.epoch} Step {self.steps} Training Loss: {loss.item()} {self.eta(start_time)}")
                        if logger:
                            logger.log_scalar(
                                tag=f"{self.args.name.lower()}_training_loss",
                                val=loss.item(),
                                step=self.steps
                            )
                            logger.log_hparams(
                                hparam_dict=self.hparam_dict,
                                metric_dict={'hparam/train_loss': loss.item()}
                            )
                            if self.lr_scheduler:
                                logger.log_scalar(
                                    tag=f"{self.args.name.lower()}_optim_lr",
                                    val=torch.tensor(self.lr_scheduler.get_last_lr()).mean().item(),
                                    step=self.steps
                                )

                        # Save training recon fig
                        if self.args.training.save_recon_freq > 0 and self.steps % self.args.training.save_recon_freq == 0:
                            try:
                                recon = output["recon"]
                                self.save_figure(input, "training", "input")
                                self.save_figure(recon, "training", "recon")
                                if logger:
                                    logger.log_figure(
                                        tag=f"{self.args.name.lower()}_training_input",
                                        fig=visualise_samples(input, scale=True),
                                        step=self.steps
                                    )
                                    logger.log_figure(
                                        tag=f"{self.args.name.lower()}_training_recon",
                                        fig=visualise_samples(recon, scale=True),
                                        step=self.steps
                                    )
                            except KeyError:
                                pass


                        # Memory diagnostics:
                        # gpu_diagnostics()

                        # Save a sample batch fig
                        if self.args.sampling.freq > 0 and self.steps % self.args.sampling.freq == 0:
                            try:
                                sample = self.sample_step(
                                    torch.randn((
                                    self.args.sampling.batch_size,
                                    input.shape[1],
                                    input.shape[2],
                                    input.shape[3]
                                    ), device=self.device)
                                )
                                # gpu_diagnostics()
                                self.save_figure(sample, "training", "sample")
                                if logger:
                                    logger.log_figure(
                                        tag=f"{self.args.name.lower()}_training_sample",
                                        fig=visualise_samples(sample, scale=True),
                                        step=self.steps
                                    )
                            except NotImplementedError:
                                pass


                        # Validate batch
                        if valid_iterator:
                            if (self.args.validation.freq > 0 and self.steps%self.args.validation.freq == 0) or \
                               (self.args.validation.save_recon_freq > 0 and self.steps%self.args.validation.save_recon_freq == 0):
                                try:
                                    val_input, val_condition = next(valid_iterator)
                                    val_input = val_input.float().to(self.device)
                                except StopIteration:
                                    valid_iterator = iter(self.valid_loader)  # restart iterator
                                    val_input, val_condition = next(valid_iterator)
                                    val_input = val_input.float().to(self.device)

                                # Validate batch
                                val_output = self.valid_step(val_input, condition=val_condition)
                                try:
                                    val_loss = val_output["loss"]
                                except KeyError as e:
                                    logging.critical("Validation step output must be a dictionary containing the key 'loss'")
                                    raise KeyError(e)

                                # Log validation
                                logging.info(f"{self.args.name.lower().capitalize()} Epoch {self.epoch} Step {self.steps} Validation Loss: {val_loss.item()}")
                                if logger:
                                    logger.log_scalar(
                                        tag=f"{self.args.name.lower()}_valid_loss",
                                        val=val_loss.item(),
                                        step=self.steps
                                    )
                                    logger.log_hparams(
                                        hparam_dict=self.hparam_dict,
                                        metric_dict={'hparam/valid_loss': val_loss.item()}
                                    )

                                # Save validation recon figure
                                if self.args.validation.save_recon_freq > 0 and self.steps%self.args.validation.save_recon_freq == 0:
                                    try:
                                        val_recon = val_output["recon"]
                                        self.save_figure(val_input, "validation", "input")
                                        self.save_figure(val_recon, "validation", "recon")
                                        logger.log_figure(
                                            tag=f"{self.args.name.lower()}_valid_input",
                                            fig=visualise_samples(val_input, scale=True),
                                            step=self.steps
                                        )
                                        logger.log_figure(
                                            tag=f"{self.args.name.lower()}_valid_recon",
                                            fig=visualise_samples(val_recon, scale=True),
                                            step=self.steps
                                        )
                                    except KeyError:
                                        pass
                                # Memory diagnostics
                                # gpu_diagnostics()

                        # Save training checkpoints
                        if (self.args.training.ckpt_freq > 0 and self.steps % self.args.training.ckpt_freq == 0) or epoch == self.args.training.n_epochs:
                            if not self.args.training.ckpt_last_only:
                                ckpt_name = f"{self.args.name.lower()}_ckpt_epoch_{self.epoch}_step_{self.steps}.pth"
                                ckpt_path = os.path.join(self.args.ckpt_path, ckpt_name)
                                self.save_checkpoint(ckpt_path)

                            # Update latest checkpoint
                            self.save_checkpoint()

                        self.steps += 1

                    self.epoch += 1

            except KeyboardInterrupt:
                self.save_checkpoint()
                raise (KeyboardInterrupt)
        self.completed = True
        return None

    @torch.no_grad()
    def validate(self) -> None:
        if self.valid_loader:
            total_loss = 0.
            for i, (input, _) in enumerate(self.valid_loader):
                input = input.float().to(self.device)
                output = self.valid_step(input)
                try:
                    loss = output["loss"]
                except KeyError as e:
                    logging.critical("Validation step output must be a dictionary containing the key 'loss'")
                    raise KeyError(e)
                total_loss += loss * input.shape[0]
            avg_loss = total_loss / len(self.valid_loader.dataset)
            logging.info(f"{self.args.name.lower().capitalize()} Average validation loss: {avg_loss.item()}")
        return None

    @torch.no_grad()
    def sample(self) -> None:
        start_time = time.time()
        if self.sample_loader:
            for i, (input, _) in enumerate(self.sample_loader):
                input = input.float().to(self.device)
                sample = self.sample_step(input)
                self.save_figure(sample, "", "sample", save_tensor=True)

                logging.info(f"{self.args.name.lower().capitalize()} Sampling batch {self.steps} / {len(self.sample_loader)} concluded. {self.eta(start_time)}")

                if self.args.sampling.sampling_only:
                    self.steps += 1
        return None

