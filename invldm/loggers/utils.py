import logging
import importlib 
from .base_logger import BaseLogger, NullLogger


def _instance_logger(args):
    if args.logging.tool is not None:
        # mod = __import__(f'.{args.logging.tool.lower()}_logger.py', fromlist=[args.logging.tool], globals=globals())
        # mod_name = f'.{args.logging.tool.lower()}_logger'
        mod = importlib.import_module(f'.{args.logging.tool.lower()}_logger', "invldm.loggers")
        cls_name = args.logging.tool + "Logger"

        
        try:
            cls = getattr(mod, cls_name)
        except AttributeError as e:
            logging.warn(f'Expecting {args.logging.tool.lower()}_logger.py'
                        f' to contain a logger class named {cls_name}. Note that this is case sensitive.')
            raise AttributeError(e)
        assert (issubclass(cls, BaseLogger))
        logger = cls(args)
    else:
        logger = NullLogger()
    return logger
