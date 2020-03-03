import functools
import os
import numpy as np
import random
import copy
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from template_lib.utils import get_attr_kwargs
from .build import DATASET_MAPPER_REGISTRY


@DATASET_MAPPER_REGISTRY.register()
class CIFAR10DatasetMapper(object):
  """
  A callable which takes a dataset dict in Detectron2 Dataset format,
  and map it into a format used by the model.

  This is the default callable to be used to map your dataset dict into training data.
  You may need to follow it to implement your own one for customized logic.

  The callable currently does the following:

  1. Read the image from "file_name"
  2. Applies cropping/geometric transforms to the image and annotations
  3. Prepare data and annotations to Tensor and :class:`Instances`
  """
  def build_transform(self, img_size):
    transform = transforms.Compose([
      transforms.Resize(img_size),
      transforms.ToTensor(),
      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    return transform

  def __init__(self, cfg, **kwargs):

    self.img_size             = get_attr_kwargs(cfg.dataset, 'img_size', kwargs=kwargs)

    self.transform = self.build_transform(img_size=self.img_size)

  def __call__(self, dataset_dict):
    """
    Args:
        dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

    Returns:
        dict: a format that builtin models in detectron2 accept
    """
    dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
    # USER: Write your own image loading if it's not from a file
    image = dataset_dict['image']
    dataset_dict['image'] = self.transform(image)
    return dataset_dict


def get_dict(name, data_path, subset):
  if subset.lower() == 'train':
    train = True
  elif subset.lower() == 'test':
    train = False
  c10_dataset = datasets.CIFAR10(root=data_path, train=subset, download=True)

  meta_dict = {}
  meta_dict['num_images'] = len(c10_dataset)
  meta_dict['class_to_idx'] = c10_dataset.class_to_idx
  meta_dict['classes'] = c10_dataset.classes
  MetadataCatalog.get(name).set(**meta_dict)

  dataset_dicts = []
  data_iter = iter(c10_dataset)
  for idx, (img, label) in enumerate(data_iter):
    record = {}

    record["image_id"] = idx
    record["height"] = img.height
    record["width"] = img.width
    record["image"] = img
    record["label"] = int(label)
    dataset_dicts.append(record)
  return dataset_dicts


from detectron2.data import DatasetCatalog, MetadataCatalog

data_path = "datasets/cifar10/"
registed_name = ['cifar10_train',
                 'cifar10_test']
subsets = ['train',
           'test']

for name, subset in zip(registed_name, subsets):
  # warning : lambda must specify keyword arguments
  DatasetCatalog.register(
    name, (lambda name=name, data_path=data_path, subset=subset:
           get_dict(name=name, data_path=data_path, subset=subset)))


if __name__ == '__main__':
  import matplotlib.pylab as plt
  dataset_dicts = get_dict(name=registed_name[0], data_path=data_path, subset=subsets[0])
  metadata = MetadataCatalog.get(registed_name[0])
  for d in random.sample(dataset_dicts, 3):
    img = d["image"]
    file_name = str(d['image_id']) + '.jpg'
    saved_dir = 'results/build_cifar10'
    os.makedirs(saved_dir, exist_ok=True)
    img.save(os.path.join(saved_dir, file_name))

    pass
    # plt.imshow(vis.get_image())
    # plt.show()
    # cv2_imshow(vis.get_image()[:, :, ::-1])