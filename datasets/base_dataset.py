from abc import abstractmethod

from torch.utils.data import Dataset
from torchvision.transforms import Compose, Resize, Lambda, ToTensor, Normalize

from utils.utils import scale2range, clip_outliers


class BaseDataset(Dataset):
    def __init__(self, args):
        self.args = args
        self.transform = self._get_transform()

    @staticmethod
    def modify_commandline_options(parser, is_train):
        raise NotImplementedError

    @abstractmethod
    def __len__(self):
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, index):
        raise NotImplementedError

    def _get_transform(self):
        transform_list = []
        if self.args.image_size:
            transform_list.append(Resize(self.args.image_size))
        if self.args.to_tensor:
            transform_list.append(ToTensor())
        if self.args.clip_outliers:
            transform_list.append(
                lambda x: clip_outliers(x, self.args.clip_outliers)
            )
        if self.args.scale:
            scale = self.args.scale
            transform_list.append(Lambda(
                lambda x: scale2range(x, [scale[0], scale[-1]])))
        if self.args.normalise:
            transform_list.append(Normalize(self.args.normalise[0],
                                            self.args.normalise[1]))

        return Compose(transform_list)