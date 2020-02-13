import functools
import tqdm
import os
import numpy as np
import random
import copy
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import pickle
import json

from detectron2.utils.visualizer import Visualizer
from detectron2.data import DatasetCatalog, MetadataCatalog

from template_lib.d2.data.BigGAN import ImageFolder, default_loader, find_classes, is_image_file


class CenterCropLongEdge(object):
  """Crops the given PIL Image on the long edge.
  Args:
      size (sequence or int): Desired output size of the crop. If size is an
          int instead of sequence like (h, w), a square crop (size, size) is
          made.
  """
  def __call__(self, img):
    """
    Args:
        img (PIL Image): Image to be cropped.
    Returns:
        PIL Image: Cropped image.
    """
    return transforms.functional.center_crop(img, min(img.size))

  def __repr__(self):
    return self.__class__.__name__


class DatasetMapper:
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
      CenterCropLongEdge(),
      transforms.Resize(img_size),
      transforms.ToTensor(),
      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    return transform

  def __init__(self, cfg, is_train=True):
    img_size = cfg.dataset.img_size
    self.transform = self.build_transform(img_size=img_size)
    self.is_train = is_train

  def __call__(self, dataset_dict):
    """
    Args:
        dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

    Returns:
        dict: a format that builtin models in detectron2 accept
    """
    dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
    # USER: Write your own image loading if it's not from a file
    image_path = dataset_dict['image_path']
    image = default_loader(image_path)
    dataset_dict['image'] = self.transform(image)
    return dataset_dict


def get_dict(name, data_path, show_bar=False):
  index_filename = '%s_index.json'%name

  if os.path.exists(index_filename):
    print('Loading pre-saved Index file %s...' % index_filename)
    with open(index_filename, 'r') as fp:
      dataset_dicts = json.load(fp)

  else:
    print('Saving Index file %s...' % index_filename)
    dataset_dicts = []
    classes, class_to_idx = find_classes(data_path)
    data_path = os.path.expanduser(data_path)
    pbar = sorted(os.listdir(data_path))
    if show_bar:
      pbar = tqdm.tqdm(pbar, desc='get_dict')

    idx = 0
    for target in pbar:
      d = os.path.join(data_path, target)
      if not os.path.isdir(d):
        continue

      for root, _, fnames in sorted(os.walk(d)):
        for fname in sorted(fnames):
          if is_image_file(fname):
            record = {}

            record["image_id"] = idx
            idx += 1
            # record["height"] = img.height
            # record["width"] = img.width
            image_path = os.path.join(root, fname)
            record["image_path"] = image_path
            record["label"] = class_to_idx[target]

            dataset_dicts.append(record)
    with open(index_filename, 'w') as fp:
      json.dump(dataset_dicts, fp)
    print('Save Index file %s.' % index_filename)

  meta_dict = {}
  meta_dict['num_images'] = len(dataset_dicts)
  MetadataCatalog.get(name).set(**meta_dict)

  return dataset_dicts


registed_names = ['imagenet_train', ]
data_paths = ["datasets/imagenet/train", ]

for name, data_path in zip(registed_names, data_paths):
  # warning : lambda must specify keyword arguments
  DatasetCatalog.register(
    name, (lambda name=name, data_path=data_path:
           get_dict(name=name, data_path=data_path)))


if __name__ == '__main__':
  import matplotlib.pylab as plt
  dataset_dicts = get_dict(name=registed_names[0], data_path=data_paths[0], show_bar=True)

  metadata = MetadataCatalog.get(registed_names[0])
  for d in random.sample(dataset_dicts, 3):
    image = default_loader(d['image_path'])
    img = np.asarray(image)
    visualizer = Visualizer(img, metadata=metadata, scale=0.5)
    vis = visualizer.draw_dataset_dict(d)
    file_name = os.path.basename(d['image_path'])
    saved_dir = 'results/build_ImageNet'
    os.makedirs(saved_dir, exist_ok=True)
    vis.save(os.path.join(saved_dir, file_name))
    # plt.imshow(vis.get_image())
    # plt.show()
    # cv2_imshow(vis.get_image()[:, :, ::-1])