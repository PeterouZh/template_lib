from fvcore.common.registry import Registry


D2LAYERv2_REGISTRY = Registry("D2LAYERv2_REGISTRY")  # noqa F401 isort:skip
D2LAYERv2_REGISTRY.__doc__ = """
"""
def build_d2layer_v2(cfg, **kwargs):
    """
    Build the whole model architecture, defined by ``cfg.MODEL.META_ARCHITECTURE``.
    Note that it does not load any weights from ``cfg``.
    """
    return D2LAYERv2_REGISTRY.get(cfg.name)(cfg=cfg, **kwargs)