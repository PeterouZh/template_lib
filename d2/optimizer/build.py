# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from fvcore.common.registry import Registry

OPTIMIZER_REGISTRY = Registry("OPTIMIZER_REGISTRY")  # noqa F401 isort:skip
OPTIMIZER_REGISTRY.__doc__ = """

"""


def build_optimizer(cfg, **kwargs):
    """
    Build the whole model architecture, defined by ``cfg.MODEL.META_ARCHITECTURE``.
    Note that it does not load any weights from ``cfg``.
    """
    name = cfg.name
    return OPTIMIZER_REGISTRY.get(name)(cfg, **kwargs)


