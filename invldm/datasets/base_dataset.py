from abc import abstractmethod
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Resize, Lambda, ToTensor, Normalize

from torchio.transforms import Resize as IOResize

from ..utils.utils import scale2range, clip_outliers, namespace2dict


class BaseDataset(Dataset):
    def __init__(self, args, **kwargs):
        self.args = args
        self.transform = self._get_transform(**namespace2dict(self.args))
        self.cond_transform = self._get_transform(**namespace2dict(self.args.condition))

    @abstractmethod
    def __len__(self):
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, index):
        raise NotImplementedError

    def _get_transform(self, **kwargs):
        resize = kwargs.pop("resize", None)
        clip_min = kwargs.pop("clip_min", None)
        clip_max = kwargs.pop("clip_max", None)
        resize3d = kwargs.pop("resize", None)
        antialias = kwargs.pop("antialias", True)
        to_tensor = kwargs.pop("to_tensor", False)
        outliers = kwargs.pop("clip_outliers", False)
        scale = kwargs.pop("scale", False)
        normalise = kwargs.pop("normalise", False)

        transform_list = []
        if resize:
            transform_list.append(Resize(resize, antialias=antialias))
        if resize3d:
            transform_list.append(IOResize(resize, "sitkBSpline"))
        if to_tensor:
            transform_list.append(ToTensor())
        if outliers:
            transform_list.append(ClipOutliers(fence=outliers))
        if clip_min or clip_max:
            transform_list.append(Clip(clip_min, clip_max))
        if scale:
            scale = scale
            transform_list.append(Scale(scale=scale))
        if normalise:
            transform_list.append(Normalize(normalise[0],
                                            normalise[1]))

        return Compose(transform_list)
    

class Scale(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        return scale2range(x, self.scale)
    
    def __repr__(self) -> str:
        detail = f"(scale={self.scale})"
        return f"{self.__class__.__name__}{detail}"
    

class ClipOutliers(nn.Module):
    def __init__(self, fence):
        super().__init__()
        self.fence = fence

    def forward(self, x):
        return clip_outliers(x, self.fence)
    
    def __repr__(self) -> str:
        detail = f"(fence={self.fence})"
        return f"{self.__class__.__name__}{detail}"
    
class Clip(nn.Module):
    def __init__(self, clip_min=None, clip_max=None):
        super().__init__()
        self.clip_min = clip_min
        self.clip_max = clip_max

    def forward(self, x):
        return torch.clip(x, min=self.clip_min, max=self.clip_max)
    
    def __repr__(self) -> str:
        detail = f"(min={self.clip_min}, max={self.clip_max})"
        return f"{self.__class__.__name__}{detail}"