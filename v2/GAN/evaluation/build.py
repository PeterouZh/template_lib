from fvcore.common.registry import Registry

GAN_METRIC_REGISTRY = Registry("GAN_METRIC_REGISTRY")  # noqa F401 isort:skip
GAN_METRIC_REGISTRY.__doc__ = """

"""

def build_GAN_metric(cfg, **kwargs):
    """
    Build the whole model architecture, defined by ``cfg.MODEL.META_ARCHITECTURE``.
    Note that it does not load any weights from ``cfg``.
    """

    return GAN_METRIC_REGISTRY.get(cfg.name)(cfg=cfg, **kwargs)

