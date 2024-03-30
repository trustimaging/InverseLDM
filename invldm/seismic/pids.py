from . import BaseDataConditioner


class PIDS(BaseDataConditioner):
    def __init__(self, **kwargs) -> None:
        super.__init__(**kwargs)

    def process(self) -> None:
        pass
