import os
import importlib


def get_cache_dir(config):
    root = os.path.expanduser(os.path.expandvars(config.get('config', 'root')))
    name = config.get('cache', 'name')
    return os.path.join(root, name)


def get_model_dir(config):
    root = os.path.expanduser(os.path.expandvars(config.get('config', 'root')))
    name = config.get('model', 'name')
    dnn = config.get('model', 'dnn')
    return os.path.join(root, name, dnn)


def parse_attr(s):
    m, n = s.rsplit('.', 1)
    m = importlib.import_module(m)
    return getattr(m, n)