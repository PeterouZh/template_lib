from .build import build_dataset_mapper

from . import build_cifar10, build_stl10, build_cifar10_per_class
from . import build_cifar100
from .build_ImageNet import ImageNetDatasetMapper
from .dataset_mapper import NoneMapper