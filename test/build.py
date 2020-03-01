from fvcore.common.registry import Registry

START_REGISTRY = Registry("START_REGISTRY")  # noqa F401 isort:skip
START_REGISTRY.__doc__ = """

"""


def build_start(cfg, **kwargs):
    """
    Build the whole model architecture, defined by ``cfg.MODEL.META_ARCHITECTURE``.
    Note that it does not load any weights from ``cfg``.
    """
    name = cfg.start.name
    return START_REGISTRY.get(name)(cfg=cfg, **kwargs)
