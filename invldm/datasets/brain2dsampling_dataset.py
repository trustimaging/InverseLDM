import os
import gzip
import numpy as np
import torch
from . import BaseDataset


class Brain2DSamplingDataset(BaseDataset):
    def __init__(self, args,  **kwargs):
        BaseDataset.__init__(self, args)

        assert not self.args.sampling_only, "Brain2DSamplingDataset is designed for sampling. Please see Brain2DDataset"

        # Save args in object
        self.args = args

        # Check maxsamples
        try:
            maxsamples = self.args.maxsamples
        except KeyError:
            maxsamples = None

        self.n_samples = kwargs.pop("n_samples", 64)
        self.cond_paths = []

        if self.condition.mode:
            prefix = self.args.condition.mode
            self._get_image_paths(self.args.condition.path, prefix, maxsamples)

        return None

    def _get_image_paths(self, path, prefix="", maxsamples=None):
        suffix = (".npy", ".npy.gz")
        # Loop through folders and subfolders
        for subdir, _, files in os.walk(path):
            for filename in files:
                if filename.lower().startswith(prefix) and \
                   filename.lower().endswith(suffix):
                    self.cond_paths.append(os.path.join(subdir, filename))
        self.cond_paths = self.cond_paths[:maxsamples]
        assert len(self.cond_paths) > 0, f" Found no data samples to load in {path} with prefix {prefix} and suffixes {suffix}"
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

        # Get condition, apply steps above
        if self.args.condition.mode is not None and self.args.condition.path is not None:
            cond_path = self.cond_paths[index]
            cond = self._read_npy_data(cond_path)
            cond = cond / torch.abs(torch.max(cond))
            if self.cond_transform:
                cond = self.cond_transform(cond)
            return y, cond
        return y


    def __len__(self):
        if self.args.condition.mode:
            return len(self.cond_paths)
        else:
            return self.n_samples

    def __str__(self):
        dic = {}
        dic["name"] = self.__class__.__name__
        dic.update(self.__dict__)
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
