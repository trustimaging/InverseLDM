import os
import gzip
import numpy as np
import torch
from . import BaseDataset

from scipy.ndimage import rotate


class Brain2DMaskedDataset(BaseDataset):
    def __init__(self, args):
        BaseDataset.__init__(self, args)

        # Save args in object
        self.args = args
        self.args.slowness = args.slowness if hasattr(args, "slowness") else False
        self.args.log = args.log if hasattr(args, "log") else False
        # self.p = args.mask_probability if hasattr(args, "mask_probability") else 0.5
        self.width = args.mask.width if hasattr(args.mask, "width") else 4
        self.spacing = args.mask.spacing if hasattr(args.mask, "spacing") else 10
        self.rotation = args.mask.rotation if hasattr(args.mask, "rotation") else 10
        self.mask = None

        # Checking mode
        try:
            mode = self.args.mode
            if mode.lower() in ["mri", "vp"]:
                self.mode = mode
            else:
                raise ValueError("mode must be 'mri' or 'vp', \
                                but got '{}'".format(mode))

        except (AttributeError, KeyError):
            self.mode = "vp"

        # Check maxsamples
        try:
            maxsamples = self.args.maxsamples
        except (AttributeError, KeyError):
            maxsamples = None

        # Get image paths
        self.data_paths = []
        self._get_image_paths(self.args.data_path, maxsamples)
        return None

    def _get_image_paths(self, path, maxsamples=None):
        prefix = "m" if self.mode == "mri" else "vp"
        suffix = (".npy", ".npy.gz")

        # Loop through folders and subfolders
        for filename in os.listdir(path):
            if filename.lower().startswith(prefix) and \
                filename.lower().endswith(suffix):
                self.data_paths.append(os.path.join(path, filename))
        self.data_paths = self.data_paths[:maxsamples]
        assert len(self.data_paths) > 0, f" Found no data samples to load in {path} with prefix {prefix} and suffixes {suffix}"
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
        if self.args.mode == "vp":
            y = y / 3000.

            # Slowness if prompted
            if self.args.slowness:
                y = 1. / y

            # Log if prompted
            if self.args.log:
                y = torch.log(y) + 1
        else:
            y = y / torch.max(torch.abs(y))
        
        # Apply transform
        if self.transform:
            y = self.transform(y)

        # Apply mask
        if self.mask is None:
            self.mask = create_mask(y.shape, self.spacing, self.width, self.rotation)

        x = y.clone()* self.mask
        return x, y

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


def flatten(xss):
    return [x for xs in xss for x in xs]

def create_mask(shape, spacing, width, rotation=0., orientation="vertical"):
    mask = np.ones(shape[-2:])
    if orientation == "horizontal":
        idxs = [[i + j for j in range(width)] for i in (range(0, shape[-2], spacing))]
        idxs = flatten(idxs)
        idxs = [idx for idx in idxs if idx < shape[-2]]
        mask[idxs] = 0.
    elif orientation == "vertical":
        idxs = [[i + j for j in range(width)] for i in (range(0, shape[-1], spacing))]
        idxs = flatten(idxs)
        idxs = [idx for idx in idxs if idx < shape[-1]]
        mask[:, idxs] = 0.

    mask = rotate(mask, rotation, reshape=False, mode="mirror")
    mask = torch.from_numpy(mask)
    mask.unsqueeze(0)
    repeat = list(shape)
    repeat[-1] = repeat[-2] = 1
    mask = mask.repeat(*repeat)
    return mask