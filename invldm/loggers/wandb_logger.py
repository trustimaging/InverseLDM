import os
import wandb
import torch
import numpy as np
import matplotlib.pyplot as plt
from . import BaseLogger
import nvidia_smi
import logging


class WandbLogger(BaseLogger):
    def __init__(self, args):
        super().__init__(args)
        
        # Get WandB configuration from environment variables
        api_key = os.environ.get('WANDB_API_KEY')
        project = os.environ.get('WANDB_PROJECT', 'conditioning')
        name = os.environ.get('WANDB_NAME', f"{args.name}_{args.run.run_name}")
        
        # Initialize WandB
        wandb.login(key=api_key)
        
        # Extract config from args
        config = self._extract_config(args)
        
        # Initialize WandB run
        self.run = wandb.init(
            project=project,
            name=name,
            config=config,
            dir=args.run.exp_folder
        )
        
        # Initialize nvidia-ml-py for GPU monitoring
        try:
            nvidia_smi.nvmlInit()
            self.gpu_available = True
            self.device_count = nvidia_smi.nvmlDeviceGetCount()
        except Exception as e:
            logging.warning(f"GPU monitoring not available: {e}")
            self.gpu_available = False
        
        # Track last GPU log step to reduce frequency
        self.last_gpu_log_step = 0
        # Allow configuring GPU log frequency via environment variable
        self.gpu_log_frequency = int(os.environ.get('WANDB_GPU_LOG_FREQ', '100'))
        
        # Track if hparams have been logged
        self.hparams_logged = False
        
        # Disable GPU logging if requested
        if os.environ.get('WANDB_DISABLE_GPU_LOGGING', '').lower() == 'true':
            self.gpu_available = False
            logging.info("GPU logging disabled via WANDB_DISABLE_GPU_LOGGING")
            
        return None
    
    def _extract_config(self, args):
        """Extract configuration from args namespace"""
        config = {}
        for key, value in args.__dict__.items():
            if hasattr(value, '__dict__'):
                # Recursively extract nested namespaces
                config[key] = self._namespace_to_dict(value)
            else:
                config[key] = value
        return config
    
    def _namespace_to_dict(self, namespace):
        """Convert namespace to dictionary recursively"""
        result = {}
        for key, value in namespace.__dict__.items():
            if hasattr(value, '__dict__') and not isinstance(value, torch.nn.Module):
                result[key] = self._namespace_to_dict(value)
            else:
                result[key] = str(value) if not isinstance(value, (int, float, str, bool, list, dict, type(None))) else value
        return result
    
    def log_scalar(self, tag, val, step, **kwargs):
        """Log scalar values to WandB"""
        # Use commit=False to batch logs together for better performance
        wandb.log({tag: val}, step=step, commit=False)
        
        # Log GPU metrics less frequently (every N steps)
        if "loss" in tag.lower() and self.gpu_available:
            if step - self.last_gpu_log_step >= self.gpu_log_frequency:
                self._log_gpu_metrics(step)
                self.last_gpu_log_step = step
        
        # Commit the logs
        wandb.log({}, commit=True)
        
        return None
    
    def log_figure(self, tag, fig, step, **kwargs):
        """Log matplotlib figures to WandB"""
        # Convert matplotlib figure to image
        wandb.log({tag: wandb.Image(fig)}, step=step)
        plt.close(fig)
        return None
    
    def log_hparams(self, hparam_dict, metric_dict, **kwargs):
        """Log hyperparameters - WandB handles this through config"""
        # Only log hparams once at the beginning
        if not self.hparams_logged:
            wandb.config.update(hparam_dict)
            self.hparams_logged = True
        
        # Skip logging metrics from hparams to avoid duplication
        # These metrics are already logged via log_scalar
        return None
    
    def _log_gpu_metrics(self, step):
        """Log GPU utilization and memory metrics"""
        if not self.gpu_available:
            return
        
        gpu_metrics = {}
        
        try:
            for i in range(self.device_count):
                handle = nvidia_smi.nvmlDeviceGetHandleByIndex(i)
                
                # Memory info
                mem_info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
                gpu_metrics[f'gpu_{i}/memory_used_mb'] = mem_info.used / 1024 / 1024
                gpu_metrics[f'gpu_{i}/memory_total_mb'] = mem_info.total / 1024 / 1024
                gpu_metrics[f'gpu_{i}/memory_percent'] = (mem_info.used / mem_info.total) * 100
                
                # Utilization
                util = nvidia_smi.nvmlDeviceGetUtilizationRates(handle)
                gpu_metrics[f'gpu_{i}/utilization_percent'] = util.gpu
                
                # Power
                try:
                    power = nvidia_smi.nvmlDeviceGetPowerUsage(handle) / 1000  # Convert to watts
                    power_limit = nvidia_smi.nvmlDeviceGetPowerManagementLimit(handle) / 1000
                    gpu_metrics[f'gpu_{i}/power_watts'] = power
                    gpu_metrics[f'gpu_{i}/power_limit_watts'] = power_limit
                    gpu_metrics[f'gpu_{i}/power_percent'] = (power / power_limit) * 100
                except nvidia_smi.NVMLError:
                    pass
                
                # Temperature
                try:
                    temp = nvidia_smi.nvmlDeviceGetTemperature(handle, nvidia_smi.NVML_TEMPERATURE_GPU)
                    gpu_metrics[f'gpu_{i}/temperature_c'] = temp
                except nvidia_smi.NVMLError:
                    pass
            
            wandb.log(gpu_metrics, step=step)
            
        except Exception as e:
            logging.warning(f"Error logging GPU metrics: {e}")
    
    def __del__(self):
        """Cleanup when logger is destroyed"""
        if hasattr(self, 'gpu_available') and self.gpu_available:
            try:
                nvidia_smi.nvmlShutdown()
            except:
                pass
        
        if hasattr(self, 'run'):
            self.run.finish() 