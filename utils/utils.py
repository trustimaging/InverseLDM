import argparse
import torch


def dict2namespace(dictionary: dict) -> argparse.Namespace:
    namespace = argparse.Namespace()
    for key, value in dictionary.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def namespace2dict(namespace: argparse.Namespace,
                   flatten: bool = False) -> dict:
    dictionary = {}
    for key, val in namespace.__dict__.items():
        if isinstance(val, argparse.Namespace):
            lower_dictionary = namespace2dict(val, flatten)
            if flatten:
                dictionary.update(lower_dictionary)
            else:
                dictionary[key] = lower_dictionary
        else:
            dictionary[key] = val
    return dictionary


def namespcae_summary_ticket(namespace, rcnt=-1):
    rcnt += 1
    ticket = "\n"
    for key, value in namespace.__dict__.items():
        if isinstance(value, argparse.Namespace):
            ticket += ("\n" + 2*rcnt * "\t" + key + ":")
            lower_ticket = namespcae_summary_ticket(value, rcnt)
            ticket += lower_ticket
        else:
            ticket += (2*rcnt * "\t" + key + ": " + str(value) + "\n")
    return ticket


def scale2range(x, range=[-1, 1]):
    return (x - x.min()) * (max(range) - min(range)) / (x.max() - x.min()) + min(range)


def clip_outliers(x, fence="outer"):
    f = 1.5 if fence == "inner" else 3.0
    q1 = torch.quantile(x, q=0.25)
    q3 = torch.quantile(x, q=0.75)
    iqr = q3 - q1
    lower = q1 - f*iqr
    upper = q3 + f*iqr
    x[torch.where(x < lower)] = lower
    x[torch.where(x > upper)] = upper
    return x