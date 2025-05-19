import os
import gzip
import numpy as np
import torch
from . import BaseDataset
from ..utils.slice_condition import create_slice_condition


class Brain2DSamplingDataset(BaseDataset):
    def __init__(self, args,  **kwargs):
        BaseDataset.__init__(self, args)

        assert self.args.sampling_only, "Brain2DSamplingDataset is designed for sampling. Please see Brain2DDataset"

        # Save args in object
        self.args = args

        # Check maxsamples
        try:
            maxsamples = self.args.maxsamples
        except (AttributeError, KeyError):
            maxsamples = None

        self.n_samples = kwargs.pop("n_samples", 64)
        self.cond_paths = []

        # For slice conditioning, we'll generate synthetic slice numbers
        self.slice_numbers = None
        if self.args.condition.mode == "slice":
            # Generate evenly spaced slice numbers within a reasonable range (e.g., 0-255)
            # This can be customized based on the data
            slice_range = kwargs.pop("slice_range", (0, 255))
            min_slice, max_slice = slice_range
            
            if hasattr(self.args.sampling, "slice_values") and self.args.sampling.slice_values:
                # Use specific slice values if provided
                self.slice_numbers = self.args.sampling.slice_values
            else:
                # Generate evenly distributed slices
                self.slice_numbers = torch.linspace(min_slice, max_slice, self.n_samples).int().tolist()
        # Regular condition path handling
        elif self.args.condition.mode and self.args.condition.path:
            include = self.args.condition.mode
            self._get_image_paths(self.args.condition.path, include, maxsamples)

        return None

    def _get_image_paths(self, path, include="", maxsamples=None):
        suffix = (".npy", ".npy.gz")
        # Loop through folders and subfolders
        for subdir, _, files in os.walk(path):
            for filename in files:
                if include in filename.lower() and \
                   filename.lower().endswith(suffix):
                    self.cond_paths.append(os.path.join(subdir, filename))
        self.cond_paths = self.cond_paths[:maxsamples]
        assert len(self.cond_paths) > 0, f" Found no data samples to load in {path} with that includes {include} and has suffixes {suffix}"
        return None

    
    def _read_npy_data(self, data_path):
        # Load .npy or .npy.gz data into torch tensor
        if data_path.endswith(".npy.gz"):
            with gzip.open(data_path, 'rb') as f:
                y = torch.from_numpy(np.load(f))
        elif data_path.endswith(".npy"):
            y = torch.from_numpy(np.load(data_path))
        else:
            raise TypeError

        # Check channel dimension exists
        if len(y.shape) < 3:
            y = y.unsqueeze(0)
        return y

    def __getitem__(self, index):
        # Input for sampling
        y = torch.rand((
            self.args.sampling.in_channels,
            self.args.sampling.in_size[0],
            self.args.sampling.in_size[1]
        ))

        # Handle slice conditioning
        if self.args.condition.mode == "slice" and self.slice_numbers is not None:
            # Create a filename-like string to pass to create_slice_condition
            slice_num = self.slice_numbers[index % len(self.slice_numbers)]
            dummy_path = f"dummy_vp_123456_sagittal_{slice_num}.npy"
            
            # Create condition tensor with the slice number
            cond_shape = (1, y.shape[1], y.shape[2])
            cond = create_slice_condition(dummy_path, cond_shape)
            
            if self.cond_transform:
                cond = self.cond_transform(cond)
            return y, cond
        # Regular file-based conditioning
        elif self.args.condition.mode is not None and self.args.condition.path is not None:
            cond_path = self.cond_paths[index]
            cond = self._read_npy_data(cond_path)
            cond = cond / torch.abs(torch.max(cond))
            if self.cond_transform:
                cond = self.cond_transform(cond)
            return y, cond
        return y


    def __len__(self):
        if self.args.condition.mode == "slice" and self.slice_numbers is not None:
            return len(self.slice_numbers)
        elif self.args.condition.mode and self.cond_paths:
            return len(self.cond_paths)
        else:
            return self.n_samples

    def __str__(self):
        dic = {}
        dic["name"] = self.__class__.__name__
        dic.update(self.__dict__)
        if hasattr(self, "cond_paths") and self.cond_paths:
            dic.pop("cond_paths")
        dic["len"] = self.__len__()
        return "{}".format(dic)

    def info(self, nsamples=None):
        nsamples = self.__len__() if nsamples is None else nsamples
        sample = self.__getitem__(0)[0]
        idx = torch.randint(0, self.__len__(), [nsamples])
        arr = torch.empty_like(sample).unsqueeze(0).repeat(nsamples, 1, 1, 1)
        for i in range(nsamples):
            arr[i] = self.__getitem__(idx[i])[0]
        stats = {"max": arr.max().item(), "min": arr.min().item(),
                 "mean": arr.mean().item(), "std": arr.std().item(),
                 "shape": sample.shape}
        return stats
