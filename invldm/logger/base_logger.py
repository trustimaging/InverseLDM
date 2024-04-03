from abc import abstractmethod

class BaseLogger():
    def __init__(self, args):
        self.args = args

    def log_scalar(self, tag, val, step, **kwargs):
        pass

    def log_figure(self, tag, fig, step, **kwargs):
        pass

    def log_hparams(self, hparam_dict, metric_dict, **kwargs):
        pass


class NullLogger(BaseLogger):
    def __init__(self, args):
        return None
    def _make_logger(self):
        return None
