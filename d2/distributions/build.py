# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from fvcore.common.registry import Registry

D2DISTRIBUTIONS_REGISTRY = Registry("D2DISTRIBUTIONS_REGISTRY")  # noqa F401 isort:skip
D2DISTRIBUTIONS_REGISTRY.__doc__ = """

"""


def build_d2distributions(cfg, **kwargs):
    """
    Build the whole model architecture, defined by ``cfg.MODEL.META_ARCHITECTURE``.
    Note that it does not load any weights from ``cfg``.
    """
    name = cfg.name
    return D2DISTRIBUTIONS_REGISTRY.get(name)(cfg, **kwargs)

