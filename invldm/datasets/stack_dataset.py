import os
import gzip
import numpy as np
import torch
from . import BaseDataset


class StackDataset(BaseDataset):
    def __init__(self, args):
        BaseDataset.__init__(self, args)

        # Save args in object
        self.args = args
        
        # Check maxsamples
        try:
            maxsamples = self.args.maxsamples
        except KeyError:
            maxsamples = None

        # Get image paths
        self.data_paths = []
        self._get_image_paths(self.args.data_path, maxsamples)
        return None

    def _get_image_paths(self, path, maxsamples=None):
        prefix = "vp"
        suffix = ("-stack.npy", "-stack.npy.gz")

        # Loop through folders and subfolders
        for subdir, _, files in os.walk(path):
            for filename in files:
                if filename.lower().startswith(prefix) and \
                   filename.lower().endswith(suffix):
                    self.data_paths.append(os.path.join(subdir, filename))
        self.data_paths = self.data_paths[:maxsamples]
        assert len(self.data_paths) > 0, " Found no data samples to load"
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
        # Data item path
        y_path = self.data_paths[index]

        # Read data
        y = self._read_npy_data(y_path)
        
        # Apply transform
        if self.transform:
            y = self.transform(y)
        return y

    def __len__(self):
        return len(self.data_paths)

    def __str__(self):
        dic = {}
        dic["name"] = self.__class__.__name__
        dic.update(self.__dict__)
        dic.pop("data_paths")
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
