import os
import gzip
import numpy as np
import torch
from . import BaseDataset


class RandomDataset(BaseDataset):
    def __init__(self, args,  **kwargs):
        BaseDataset.__init__(self, args)
        self.n_samples = kwargs.pop("n_samples", 64)
        self.w, self.h = self.args.sampling.in_size
        self.nc = self.args.sampling.in_channels

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        if self.args.sampling_only:
            if self.args.condition.mode is None:
                return torch.rand((self.nc, self.h, self.w))
            else:
                raise NotImplementedError ("This class does not support conditional sampling. Please pass custom dataset or select from existing ones in invldm/datasets/")
        else:
            raise NotImplementedError ("This class does not support training mode. Please pass custom dataset or select from existing ones in invldm/datasets/")