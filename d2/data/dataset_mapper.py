import copy
from .build import DATASET_MAPPER_REGISTRY


@DATASET_MAPPER_REGISTRY.register()
class NoneMapper(object):

  def __init__(self, cfg, **kwargs):
    pass

  def __call__(self, dataset_dict):
    dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
    return dataset_dict
