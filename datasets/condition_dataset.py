# import os
# import gzip
# import numpy as np
# import torch
# from . import BaseDataset


# class ConditionDataset(BaseDataset):
#     def __init__(self, args):
#         BaseDataset.__init__(self, args)

#         self.condition_paths = []
#         self._get_condition_paths(self.args.data_path)

#     def _get_condition_paths(self, path, maxsamples=None):
#         suffix = (".npy", ".npy.gz", ".pt")

#         # Loop through folders and subfolders
#         for subdir, _, files in os.walk(path):
#             for filename in files:
#                 if filename.lower().endswith(suffix):
#                     self.condition_paths.append(os.path.join(subdir, filename))
#         self.condition_paths = self.condition_paths[:maxsamples]

#     def _read_npy_data(self, data_path):
#         # Load .npy, .npy.gz or .pt data into torch tensor
#         if data_path.endswith(".npy.gz"):
#             with gzip.open(data_path, 'rb') as f:
#                 y = torch.from_numpy(np.load(f))
#         elif data_path.endswith(".npy"):
#             y = torch.from_numpy(np.load(data_path))
#         elif data_path.endswith(".pt"):
#             y = torch.load(data_path)
#         else:
#             raise TypeError

#         # Ensure channel dimension exists
#         if len(y.shape) < 3:
#             y = y.unsqueeze(0)
#         return y

#     def __getitem__(self, index):
#         # Data item path
#         y_path = self.condition_paths[index]

#         # Read data
#         y = self._read_npy_data(y_path)
        
#         # Apply transform
#         if self.transform:
#             y = self.transform(y)

#         # Get condition, apply steps above
#         if self.args.condition.mode is not None and self.args.condition.path is not None:
#             cond_path = self._get_condition_path(y_path)
#             cond = self._read_npy_data(cond_path)
#             if self.cond_transform:
#                 cond = self.cond_transform(cond)
#             return y, cond
#         return y

#     def __len__(self):
#         return len(self.condition_paths)

#     def __str__(self):
#         dic = {}
#         dic["name"] = self.__class__.__name__
#         dic.update(self.__dict__)
#         dic.pop("data_paths")
#         dic["len"] = self.__len__()
#         return "{}".format(dic)

#     def info(self, nsamples=None):
#         nsamples = self.__len__() if nsamples is None else nsamples
#         sample = self.__getitem__(0)[0]
#         idx = torch.randint(0, self.__len__(), [nsamples])
#         arr = torch.empty_like(sample).unsqueeze(0).repeat(nsamples, 1, 1, 1)
#         for i in range(nsamples):
#             arr[i] = self.__getitem__(idx[i])[0]
#         stats = {"max": arr.max().item(), "min": arr.min().item(),
#                  "mean": arr.mean().item(), "std": arr.std().item(),
#                  "shape": sample.shape}
#         return stats
