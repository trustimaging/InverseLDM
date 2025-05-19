import os
import yaml
import shutil
import argparse
import random
import numpy as np
import torch
import sys
import logging
import datetime
import warnings

from monai.config import print_config as print_monai_config
from monai.utils import set_determinism

from .utils import dict2namespace, namespace2dict, namespcae_summary_ticket
from ..loggers.utils import _instance_logger

from pathlib import Path
from glob import glob
from argparse import Namespace
from dotmap import DotMap


def setup_train():
    # getting cl and yml args
    args = parse_args()

    # add default args to missing args
    default_missing_args(args)

    # sampling params not relevant for training
    # delattr(args.data, "sampling")
    # delattr(args.autoencoder, "sampling")
    # delattr(args.diffusion, "sampling")

    # check devices
    check_devices(args)

    # check overwrite:
    check_overwrite(args)

    # check and adjust dataset files
    check_dataset_file(args)

    # checking tracking tool
    check_tracking_tool(args)

    # create relevant folders
    create_training_experiment_folders(args)

    # setup seeds and devices
    set_seed(args.run.seed)
    set_determinism(args.run.seed)

    # set up loggers
    setup_logger(args)

    # if condition requested but path not passed, look at working directory
    if args.data.condition.mode is not None and args.data.condition.path is None:
        args.data.condition.path = os.getcwd()

    # for training we must make sure the sampling_only flags are off
    args.data.sampling_only = False
    args.autoencoder.sampling_only = False
    args.diffusion.sampling_only = False

    # save updated args as config in log folder
    with open(os.path.join(args.run.log_folder, "config_training.yml"), "w") as f:
        yaml.dump(namespace2dict(args), f, default_flow_style=False, sort_keys=False, indent=4)

    # Set up tracker -- note this should come AFTER the yaml dump
    setup_model_tracker(args)

    # std out experiment summary ticket
    if args.run.resume_training:
        config = "\n\n ------------------------------ RESUMED CONFIGURATION ------------------------------ " + namespcae_summary_ticket(args)
    else:
        config = "\n\n ---------------------------------- CONFIGURATION ---------------------------------- " + namespcae_summary_ticket(args)
    
    print_monai_config()
    logging.info(config)

    return args


def setup_sampling():
    # getting cl and yml args
    args = parse_args()
    print(args.autoencoder)

    # add default args to missing args
    default_missing_args(args)

    # training and validation params not relevant for sampling
    # delattr(args.autoencoder, "training")
    # delattr(args.autoencoder, "validation")
    # delattr(args.diffusion, "training")
    # delattr(args.diffusion, "validation")

    # check devices
    check_devices(args)

    # check and adjust dataset files, only required if conditioned
    if args.data.condition.mode is not None:
        check_dataset_file(args)

    # create relevant folders
    create_sampling_experiment_folders(args)

    # setup seeds and devices
    set_seed(args.run.seed)

    # set up loggers
    setup_logger(args)

    # for sampling, we must make sure the sampling_only flags are on
    args.data.sampling_only = True
    args.autoencoder.sampling_only = True
    args.diffusion.sampling_only = True

    # save updated args as config in log folder
    with open(os.path.join(args.run.log_folder, "config_sampling.yml"), "w") as f:
        yaml.dump(namespace2dict(args), f, default_flow_style=False, sort_keys=False, indent=4)

    # Set up tracker -- note this should come AFTER the yaml dump
    setup_model_tracker(args)

    # std out experiment summary ticket
    config = "\n\n ---------------------------------- SAMPLING CONFIGURATION ---------------------------------- " + namespcae_summary_ticket(args)
    logging.info(config)

    return args


def set_seed(seed):
    """
    Use this to set ALL the random seeds to a fixed value and take out any
    randomness from cuda kernels
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

    return True


def parse_args():
    # parse command line arguments
    parser = argparse.ArgumentParser(description=globals()["__doc__"])
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the config file for training or sampling"
    )
    parser.add_argument(
        "--logdir",
        type=str,
        default="exps",
        help="Path for saving running related data."
    )
    parser.add_argument(
        "--name",
        type=str,
        required=True,
        help="Name of experiment and also the log folder.",
    )
    parser.add_argument(
        "--comment",
        type=str,
        default="",
        help="A string for experiment comment"
    )
    parser.add_argument(
        "--jobid",
        type=int,
        default=-1,
        help="Store job ID for hpc systems"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to which run. Options: cuda or cpu. If cuda, specify\
            gpu_ids with flag"
    )
    parser.add_argument(
        "--gpu_ids",
        type=list,
        default=None,
        help="List of gpu ids to be used in DataParallel. If None, all will be\
            used"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--resume_training",
        action="store_true",
        help="Whether to resume training. Used in conjunction\
            with checkpoint flag in yaml config. If checkpoint not specified, will use latest model."
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Whether to \
            overwrite experiment of same name"
    )
    parser.add_argument(
        "-y", action="store_true", help="Automatic 'yes' to user input\
            questions"
    )
    parser.add_argument(
        "--verbose",
        type=str,
        default="info",
        help="Verbose level: info | debug | warning | critical",
    )
    run_args = parser.parse_args()
    run_args.exp_folder = os.path.join(run_args.logdir,
                                       run_args.name)
    run_args.log_folder = os.path.join(run_args.exp_folder,
                                       "logs")

    # read config, add run args to namespace
    args = parse_config(run_args.config)
    args.run = run_args

    return args


def parse_config(config_file):
    try:
        with open(config_file, "r") as f:
            config = dict2namespace(yaml.safe_load(f))
    except yaml.constructor.ConstructorError:
        logging.warn(f"Loading {config_file} unsafely with YML.")
        with open(config_file, "r") as f:
            config = yaml.unsafe_load(f)
        try:
            config = dict2namespace(config)
        except AttributeError:
            pass
    
    return config


def check_dataset_file(args):
    # Check dataset_path:
    try:
        dataset_path = args.data.dataset_path
        if dataset_path is None or not dataset_path:
            raise AttributeError
    except AttributeError:
        dataset_path = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                    '..', "datasets"))
        args.data.dataset_path = dataset_path

    # Check dataset file
    try:
        open(os.path.join(args.data.dataset_path,
                          args.data.dataset.lower() + "_dataset.py"), "r")
    except AttributeError:
        raise Exception(AttributeError, "Dataset name must be passed in config\
                        file as 'dataset'. See config example. ")
    except FileNotFoundError:
        msg = f"File {args.data.dataset.lower()}_dataset.py must exist in\
                            {args.data.dataset_path}"
        logging.exception(msg)
        raise Exception(FileNotFoundError, msg)
    return args


def check_devices(args):
    if args.run.device == "cpu" or "cuda" in args.run.device:
        pass
    else:
        new_device = "cuda" if torch.cuda.is_available() else "cpu"
        logging.warn(f"Device '{args.run.device}' can't be recognised. Changing devide to {new_device}.")
        args.run.device = new_device

    gpu_ids = []
    if args.run.gpu_ids:
        for id in args.run.gpu_ids:
            try:
                gpu_ids.append(int(id))
            except ValueError:
                pass
            args.run.gpu_ids = gpu_ids
    return args


def check_tracking_tool(args):
    if args.logging.tool:
        track_file = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                  '..', "loggers", args.logging.tool.lower() + "_logger.py"))
        try:
            open(track_file, "r")
        except FileNotFoundError:
            raise Exception(FileNotFoundError,
                            f"Expecting file '{track_file}' to exist, but not found")
    return args


def default_missing_args(args):
    default_args = parse_config(os.path.join(os.path.dirname(
        os.path.abspath(__file__)), "default_config.yml"))
    
    def default(args, default_args):
        for key in default_args.__dict__:
            default_value = default_args.__dict__[key]
            try:
                args.__dict__[key]
            except KeyError:
                setattr(args, key, default_value)

            if isinstance(default_value, argparse.Namespace):
                _ = default(args.__dict__[key],
                            default_args.__dict__[key])
        return args
    
    args = default(args, default_args)

    args.autoencoder.name = "autoencoder"
    args.diffusion.name = "diffusion"

    return args


def check_overwrite(args):
    if args.run.overwrite:
        if args.run.resume_training:
            args.run.overwrite = False
            logging.warn("Both --resume_training and --overwrite flags passed, turning off overwrite")
        elif os.path.exists(args.run.exp_folder):
            if args.run.y:
                user_input = "y"
            else:
                user_input = input(f"The current directory '{args.run.exp_folder}' is about to be rewritten. Do you wish to proceed (YES/NO) : ")
            
            if user_input.lower() in ['y', 'yes']:
                pass
            elif user_input.lower() in ['n', 'no']:
                logging.info("No overwriting, aborting script.")
                sys.exit()
            else:
                check_overwrite(args)
    return args


def create_folder(folder, overwrite=False, exception=True):
    try:
        os.makedirs(folder)
    except FileExistsError as e:
        if overwrite:
            logging.info(f"Overwriting {folder} ...")
            shutil.rmtree(folder)
            os.makedirs(folder)
            logging.info(f"Done!")
        else:
            if exception:
                raise Exception(FileExistsError, f"Experiment folder '{folder}' already exists. Use the --overwrite flag if you wish to overwrite or --resume_training flag to continue training from a checkpoint.")
            else:
                pass
    return None


def create_training_experiment_folders(args):
    # Whether to raise overwriting exception when folder already exists
    if args.run.resume_training:
        exception = False
    else:
        exception = True

    # Create experiment folder
    create_folder(args.run.exp_folder, args.run.overwrite, exception)

    # Create logging folder
    create_folder(args.run.log_folder, args.run.overwrite, exception)

    # Create logging folders for autoencoder and diffusion
    autoencoder_log_path = os.path.join(args.run.log_folder, "autoencoder")
    args.autoencoder.log_path = autoencoder_log_path
    create_folder(autoencoder_log_path, args.run.overwrite, exception)

    diffusion_log_path = os.path.join(args.run.log_folder, "diffusion")
    args.diffusion.log_path = diffusion_log_path
    create_folder(diffusion_log_path, args.run.overwrite, exception)

    # Create and checkpoint folder for autoencoder and diffusion
    autoencoder_ckpt_path = os.path.join(autoencoder_log_path, "checkpoints")
    args.autoencoder.ckpt_path = autoencoder_ckpt_path
    create_folder(autoencoder_ckpt_path, args.run.overwrite, exception)

    diffusion_ckpt_path = os.path.join(diffusion_log_path, "checkpoints")
    args.diffusion.ckpt_path = diffusion_ckpt_path
    create_folder(diffusion_ckpt_path, args.run.overwrite, exception)

    # Create model tracking management folder
    if args.logging.tool:
        track_folder = os.path.join(args.run.exp_folder, args.logging.tool)
        args.logging.track_folder = track_folder
        create_folder(track_folder, args.run.overwrite, exception)

    # Create samples folders
    if args.diffusion.training.sampling_freq > 0:
        samples_folder = os.path.join(diffusion_log_path, "samples")
        args.diffusion.samples_path = samples_folder
        create_folder(samples_folder, args.run.overwrite, exception)
    if args.autoencoder.training.sampling_freq > 0:
        samples_folder = os.path.join(autoencoder_log_path, "samples")
        args.autoencoder.samples_path = samples_folder
        create_folder(samples_folder, args.run.overwrite, exception)

    # Create recon folders
    if (args.autoencoder.training.save_recon_freq > 0 or args.autoencoder.validation.save_recon_freq > 0):
        recon_folder = os.path.join(autoencoder_log_path, "recons")
        args.autoencoder.recon_path = recon_folder
        create_folder(recon_folder, args.run.overwrite, exception)

    return None


def create_sampling_experiment_folders(args):

    # Check experiment exists
    assert os.path.exists(args.run.exp_folder), f"Experiment folder '{args.run.exp_folder}' not found. "
    assert os.path.exists(args.run.log_folder), f"Experiment logging folder '{args.run.log_folder}' not found. "

    autoencoder_log_path = os.path.join(args.run.log_folder, "autoencoder")
    args.autoencoder.log_path = autoencoder_log_path
    assert os.path.exists(autoencoder_log_path), f" Autoencoder logging folder '{autoencoder_log_path}' not found. " 

    diffusion_log_path = os.path.join(args.run.log_folder, "diffusion")
    args.diffusion.log_path = diffusion_log_path
    assert os.path.exists(diffusion_log_path), f" Diffusion logging folder '{diffusion_log_path}' not found. " 

    autoencoder_ckpt_path = os.path.join(autoencoder_log_path, "checkpoints")
    args.autoencoder.ckpt_path = autoencoder_ckpt_path
    assert os.path.exists(autoencoder_ckpt_path), f" Autoencoder checkpoint folder '{autoencoder_ckpt_path}' not found. " 

    diffusion_ckpt_path = os.path.join(diffusion_log_path, "checkpoints")
    args.diffusion.ckpt_path = diffusion_ckpt_path
    assert os.path.exists(diffusion_ckpt_path), f" Autoencoder checkpoint folder '{diffusion_ckpt_path}' not found. " 

    # Create samples folders
    samples_folder = os.path.join(args.run.exp_folder, "samples")
    args.run.samples_folder = samples_folder
    create_folder(samples_folder, args.run.overwrite, exception=False)

    return None


def setup_logger(args):
    level = getattr(logging, args.run.verbose.upper(), None)
    if not isinstance(level, int):
        raise ValueError("Logger level {} not supported".format(args.run.verbose))

    handler1 = logging.StreamHandler()
    handler2 = logging.FileHandler(os.path.join(args.run.log_folder, "stdout.txt"))
    formatter = logging.Formatter(
        "%(levelname)s - %(filename)s - %(asctime)s - %(message)s"
    )
    handler1.setFormatter(formatter)
    handler2.setFormatter(formatter)

    logger = logging.getLogger()
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    logger.setLevel(level)

    args.logging.logger = logger
    args.run.pid = os.getpid()

    logging.info("Writing log file to {}".format(args.run.log_folder))
    logging.info("Exp instance id = {}".format(args.run.pid))
    logging.info("Exp comment = {}".format(args.run.comment))

    return args


def setup_model_tracker(args):
    if args.logging.tool:
        args.logging.tracker = _instance_logger(args)
    else:
        args.logging.tracker = None
    return args


def _load_config(yaml_path, exp_name=None):
    """Load yaml configuration settings."""
    with open(yaml_path, 'r') as yaml_file:
        try:
            yaml_config = yaml.safe_load(yaml_file)
        except yaml.YAMLError as exc:
            print(exc)

    # Return a Python object instead of a dictionary for cleaner access to params
    config = DotMap(yaml_config, _dynamic=False)

    # Add default params if not provided
    config = _add_default_params(config, exp_name)

    # Override some config values with gpu device and experiment folder
    config = _override_values(config)

    # Set correct ckpt paths for training/sampling
    config = _set_ckpt_paths(config)

    return config

def _add_default_params(config, exp_name=None):
    """Fill in default values not passed in yaml config."""

    # Add defaults of sampling
    config.sampling_only = False

    # Add defaults for condition config
    config.data.setdefault("condition", {})
    config.data.condition.setdefault("mode", None)
    config.data.condition.setdefault("path", None)
    config.data.condition.setdefault("resize_mode", "bilinear")

    # if condition requested but path not passed, look at working directory
    if config.data.condition.mode is not None and config.data.condition.path is None:
        config.data.condition.path = os.getcwd()

    # add experiment (defaults to time)
    if exp_name is None:
        curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        exp_name = curr_time
    config.experiment = exp_name

    # Device settings
    if not hasattr(config, "device"):
        if torch.cuda.is_available():
            config.device = "cuda:0"
        else:
            config.device = "cpu"
    os.environ["CUDA_VISIBLE_DEVICES"] = config.device.split(":")[-1]
    config.device = "cuda:0" if "cuda" in config.device else "cpu"

    if not hasattr(config, "run"):
        config.run = {}
    if not hasattr(config.run, "deterministic"):
        config.run.deterministic = True
    if not hasattr(config.run, "seed"):
        config.run.seed = 42
    
    # Add LR schedulers
    for model_name in ["autoencoder", "diffusion"]:
        # Skip if model section doesn't exist
        if model_name not in config:
            continue
            
        if not hasattr(config[model_name].model, "condition"):
            config[model_name].model.condition = {}
        if not hasattr(config[model_name].model.condition, "mode"):
            config[model_name].model.condition.mode = None
        if not hasattr(config[model_name].model.condition, "resize_mode"):
            config[model_name].model.condition.resize_mode = "bilinear"

        # For the diffusion model, add setup for more complex conditioning
        if model_name == "diffusion" and config[model_name].model.condition.mode == "crossattn":
            # Set defaults for cross-attention conditioning
            if not hasattr(config[model_name].model.condition, "num_res_blocks"):
                config[model_name].model.condition.num_res_blocks = 2
            if not hasattr(config[model_name].model.condition, "spatial_dims"):
                config[model_name].model.condition.spatial_dims = config[model_name].model.spatial_dims
            if not hasattr(config[model_name].model.condition, "in_channels"):
                config[model_name].model.condition.in_channels = 1  # Default for slice number
            if not hasattr(config[model_name].model.condition, "out_channels"):
                config[model_name].model.condition.out_channels = 64  # Default embedding size
            if not hasattr(config[model_name].model.condition, "num_channels"):
                config[model_name].model.condition.num_channels = [64, 128, 128]
            if not hasattr(config[model_name].model.condition, "attention_levels"):
                config[model_name].model.condition.attention_levels = [False, False, True]
                
        # check and adjust dataset files, only required if conditioned
        if config.data.condition.mode is not None:
            pass

        # check lr scheduler has seed
        if "optim" in config[model_name]:
            if "lr_scheduler" not in config[model_name].optim:
                config[model_name].optim.lr_scheduler = {}
            if "seed" not in config[model_name].optim.lr_scheduler:
                config[model_name].optim.lr_scheduler.seed = config.run.seed
            if "scheduler" not in config[model_name].optim.lr_scheduler:
                config[model_name].optim.lr_scheduler.scheduler = None

    return config

def _override_values(config):
    """Override config values when needed."""
    if hasattr(config, "device"):
        # Device settings
        if torch.cuda.is_available():
            device_idx = "0" if ":" not in config.device else config.device.split(":")[-1]
            os.environ["CUDA_VISIBLE_DEVICES"] = device_idx
            config.device = "cuda:0"  # Always use first enumerated device
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
            config.device = "cpu"
            warnings.warn("No GPU detected. Using CPU...")

    # Override submodels
    for model_name in ["autoencoder", "diffusion"]:
        # Skip if model section doesn't exist
        if model_name not in config:
            continue
            
        # Set model name
        config[model_name].name = model_name
            
    return config

def _set_ckpt_paths(config):
    """Set checkpoint paths for training/sampling."""   

    # set absolute path to file via experiment folder
    experiment_folder = str(os.path.join(os.getcwd(), "exps", 
                            config.experiment))
    for model_name in ["autoencoder", "diffusion"]:
        # Skip if model section doesn't exist
        if model_name not in config:
            continue
                
        # Set paths for training/sampling
        if config.sampling_only:
            # check if checkpoint provided in YAML file and if it exists
            if hasattr(config[model_name].model, "checkpoint") and \
                    config[model_name].model.checkpoint is not None:
                chkpt_file = config[model_name].model.checkpoint
                if not os.path.isfile(chkpt_file):
                    # Check if file exists with cwd as prefix
                    tmp_chkpt_file = os.path.join(
                        os.getcwd(), config[model_name].model.checkpoint)
                    if os.path.isfile(tmp_chkpt_file):
                        chkpt_file = tmp_chkpt_file
                    else:
                        # Check all "*model_name*/checkpoints/" files
                        glob_pattern = os.path.join(
                            "**", model_name, "checkpoints", "*.pth")
                        all_ckpt_files = list(
                            Path(os.getcwd()).rglob(glob_pattern[-5:]))
                        if all_ckpt_files:
                            # Take newest (sorted by name == time encoded)
                            chkpt_file = sorted(all_ckpt_files)[-1]
                        else:
                            # Default, model to not found later
                            chkpt_file = None
                # TODO: change code below with elif.
                # Should be:
                # if invalid_path and valid_path_with_getcwd:
                #     ...
                # elif invalid_path and valid_glob_pattern:
                #     ...
                # else:
                #     chkpt_file = None
                
                config[model_name].model.checkpoint = chkpt_file
        else: # not config.sampling_only
            # Get autencoder checkpoint from experiments
            config[model_name].experiment_folder = str(os.path.join(
                experiment_folder, model_name))
        
    return config


def _get_config(yaml_path=None, exp_name=None, config_dict=None):
    """Get configuration settings from configuration yaml file."""
    if config_dict is not None:
        config = DotMap(config_dict, _dynamic=False)
        config = _add_default_params(config, exp_name)
        config = _override_values(config)
        config = _set_ckpt_paths(config)
        return config

    # Need to provide either the dict or a file
    assert yaml_path is not None, "Need to provide either " \
                                  "yaml path or config dict"

    # Return a Python object instead of a dictionary for cleaner access to params
    config = _load_config(yaml_path, exp_name)
    return config
