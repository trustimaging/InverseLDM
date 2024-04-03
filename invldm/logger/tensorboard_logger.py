import os
from . import BaseLogger
import torch.utils.tensorboard as tb


class TensorboardLogger(BaseLogger):
    def __init__(self, args):
        super().__init__(args)

        self.logger = tb.SummaryWriter(
            log_dir=os.path.join(
                self.args.run.exp_folder,
                "tensorboard"
            )
        )
        return None

    def log_scalar(self, tag, val, step, **kwargs):
        self.logger.add_scalar(tag, val, step)
        return None

    def log_figure(self, tag, fig, step, **kwargs):
        self.logger.add_figure(tag, fig, step)
        return None

    # def log_hparams(self, hparam_dict, metric_dict, **kwargs):
    #     # fix types
    #     hparam_dict = self._cast_str_invalid_types(hparam_dict)
    #     metric_dict = self._cast_str_invalid_types(metric_dict)
    #     self.logger.add_hparams(hparam_dict, metric_dict)
    #     return None

    # def _cast_str_invalid_types(self, dictionary):
    #     accepted_types = [int, float, str, bool, torch.Tensor]
    #     for k, v in dictionary.items():
    #         if v and (type(v) in accepted_types):
    #             pass
    #         else:
    #             dictionary[k] = str(v)
    #     return dictionary
