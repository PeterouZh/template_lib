# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from fvcore.common.registry import Registry

GAN_LOSS_REGISTRY = Registry("GAN_LOSS_REGISTRY")  # noqa F401 isort:skip
GAN_LOSS_REGISTRY.__doc__ = """

"""


def build_GAN_loss(cfg,  **kwargs):
    """
    """
    name = cfg.name
    return GAN_LOSS_REGISTRY.get(name)(cfg=cfg, **kwargs)

