import torch
from abc import abstractmethod


class BaseDataConditioner():
    def __init__(self, **kwargs) -> None:
        self.args = kwargs.pop("args")
        self.condition = None

    @abstractmethod
    def process(self) -> None:

        if self.args.save_condition:
            self.args.seismic_folder
            pass
        self.condition = None  ## Update self.condition
        raise NotImplementedError
    
    def reverse(self) -> torch.Tensor:
        raise NotImplementedError

    def get_condition(self) -> torch.Tensor:
        return self.condition


