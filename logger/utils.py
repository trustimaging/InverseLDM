import logging

from .base_logger import BaseLogger, NullLogger


def _instance_logger(args):
    if args.logging.tool is not None:
        mod = __import__(f'logger.{args.logging.tool.lower()}_logger',
                        fromlist=[args.logging.tool])
        cls_name = args.logging.tool.lower().capitalize() + "Logger"
        try:
            cls = getattr(mod, cls_name)
        except AttributeError as e:
            logging.warn(f'Expecting {args.logging.tool.lower()}_logger.py'
                        f' to contain a logger class named {cls_name}')
            raise (AttributeError, e)
        assert (issubclass(cls, BaseLogger))
        logger = cls(args)
    else:
        logger = NullLogger()
    return logger
