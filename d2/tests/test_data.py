import random
import os, sys
import unittest

import template_lib.utils as utils


class TestingBuildImageNet(unittest.TestCase):

  def test_load_in_memory(self):
    """
    Usage:
        export CUDA_VISIBLE_DEVICES=4,5
        export PORT=6006
        export TIME_STR=0
        export PYTHONPATH=.:./EXPERIMENTS:./detectron2_lib

        python 	EXPERIMENTS/pagan/train_net.py \
          --config EXPERIMENTS/pagan/config/pagan.yaml \
          --command ddp_search_cgan_gen_ImageNet_debug \
          --outdir results/PAGAN_ImageNet/ddp_search_cgan_gen_ImageNet_debug
    :return:
    """
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
      os.environ['CUDA_VISIBLE_DEVICES'] = '4,5'
    if 'PORT' not in os.environ:
      os.environ['PORT'] = '6006'
    if 'TIME_STR' not in os.environ:
      os.environ['TIME_STR'] = '0' if utils.is_debugging() else '0'
    # func name
    assert sys._getframe().f_code.co_name.startswith('test_')
    command = sys._getframe().f_code.co_name[5:]
    class_name = self.__class__.__name__[7:] \
      if self.__class__.__name__.startswith('Testing') \
      else self.__class__.__name__
    outdir = f'results/{class_name}/{command}'

    from template_lib.d2.data.build_ImageNet import get_dict, registed_names, data_paths, images_per_class_list, kwargs_list
    from detectron2.data import MetadataCatalog

    dataset_dicts = get_dict(name=registed_names[-1], data_path=data_paths[-1],
                             images_per_class=images_per_class_list[-1], show_bar=True, **(kwargs_list[-1]))

    metadata = MetadataCatalog.get(registed_names[-1])
    for d in random.sample(dataset_dicts, 3):

      pass

  def test_create_imagenet_train_index(self):
    """
    Usage:
        export CUDA_VISIBLE_DEVICES=4,5
        export PORT=6006
        export TIME_STR=0
        export PYTHONPATH=.:./EXPERIMENTS:./detectron2_lib

        python 	EXPERIMENTS/pagan/train_net.py \
          --config EXPERIMENTS/pagan/config/pagan.yaml \
          --command ddp_search_cgan_gen_ImageNet_debug \
          --outdir results/PAGAN_ImageNet/ddp_search_cgan_gen_ImageNet_debug
    :return:
    """
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
      os.environ['CUDA_VISIBLE_DEVICES'] = '4,5'
    if 'PORT' not in os.environ:
      os.environ['PORT'] = '6006'
    if 'TIME_STR' not in os.environ:
      os.environ['TIME_STR'] = '0' if utils.is_debugging() else '0'
    # func name
    assert sys._getframe().f_code.co_name.startswith('test_')
    command = sys._getframe().f_code.co_name[5:]
    class_name = self.__class__.__name__[7:] \
      if self.__class__.__name__.startswith('Testing') \
      else self.__class__.__name__
    outdir = f'results/{class_name}/{command}'

    from template_lib.d2.data.build_ImageNet import get_dict, registed_names, data_paths, images_per_class_list, \
      kwargs_list
    from detectron2.data import MetadataCatalog

    dataset_dicts = get_dict(name=registed_names[0], data_path=data_paths[0],
                             images_per_class=images_per_class_list[0], show_bar=True, **(kwargs_list[0]))

    metadata = MetadataCatalog.get(registed_names[-1])
    for d in random.sample(dataset_dicts, 3):
      pass

  def test_register_imagenet_per_class(self):
    """
    Usage:
        export CUDA_VISIBLE_DEVICES=4,5
        export PORT=6006
        export TIME_STR=0
        export PYTHONPATH=.:./EXPERIMENTS:./detectron2_lib

        python 	EXPERIMENTS/pagan/train_net.py \
          --config EXPERIMENTS/pagan/config/pagan.yaml \
          --command ddp_search_cgan_gen_ImageNet_debug \
          --outdir results/PAGAN_ImageNet/ddp_search_cgan_gen_ImageNet_debug
    :return:
    """
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
      os.environ['CUDA_VISIBLE_DEVICES'] = '4,5'
    if 'PORT' not in os.environ:
      os.environ['PORT'] = '6006'
    if 'TIME_STR' not in os.environ:
      os.environ['TIME_STR'] = '0' if utils.is_debugging() else '0'
    # func name
    assert sys._getframe().f_code.co_name.startswith('test_')
    command = sys._getframe().f_code.co_name[5:]
    class_name = self.__class__.__name__[7:] \
      if self.__class__.__name__.startswith('Testing') \
      else self.__class__.__name__
    outdir = f'results/{class_name}/{command}'

    from template_lib.d2.data.build_ImageNet_per_class import get_dict, registed_names, data_paths, \
      kwargs_list
    from detectron2.data import MetadataCatalog

    # dataset_dicts = get_dict(name=registed_names[0], data_path=data_paths[0], show_bar=True, **(kwargs_list[0]))
    #
    # metadata = MetadataCatalog.get(registed_names[-1])
    # for d in random.sample(dataset_dicts, 3):
    #   pass


class TestingBuildCIFAR10(unittest.TestCase):

  def test_cifar10_train(self):
    """

    """
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
      os.environ['CUDA_VISIBLE_DEVICES'] = '4,5'
    if 'PORT' not in os.environ:
      os.environ['PORT'] = '6006'
    if 'TIME_STR' not in os.environ:
      os.environ['TIME_STR'] = '0' if utils.is_debugging() else '0'
    # func name
    assert sys._getframe().f_code.co_name.startswith('test_')
    command = sys._getframe().f_code.co_name[5:]
    class_name = self.__class__.__name__[7:] \
      if self.__class__.__name__.startswith('Testing') \
      else self.__class__.__name__
    outdir = f'results/{class_name}/{command}'
    os.makedirs(outdir, exist_ok=True)

    import matplotlib.pylab as plt
    from template_lib.d2.data.build_cifar10 import data_path, registed_name_list, registed_func_list, kwargs_list
    from detectron2.data import MetadataCatalog

    dataset_dicts = registed_func_list[0](name=registed_name_list[0], data_path=data_path, **kwargs_list[0])
    metadata = MetadataCatalog.get(registed_name_list[0])
    for d in random.sample(dataset_dicts, 3):
      img = d["image"]
      file_name = str(d['image_id']) + '.jpg'
      img.save(os.path.join(outdir, file_name))
      pass

  def test_cifar10_train_sampler(self):
    """

    """
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
      os.environ['CUDA_VISIBLE_DEVICES'] = '4,5'
    if 'PORT' not in os.environ:
      os.environ['PORT'] = '6006'
    if 'TIME_STR' not in os.environ:
      os.environ['TIME_STR'] = '0' if utils.is_debugging() else '0'
    # func name
    assert sys._getframe().f_code.co_name.startswith('test_')
    command = sys._getframe().f_code.co_name[5:]
    class_name = self.__class__.__name__[7:] \
      if self.__class__.__name__.startswith('Testing') \
      else self.__class__.__name__
    outdir = f'results/{class_name}/{command}'
    os.makedirs(outdir, exist_ok=True)

    import matplotlib.pylab as plt
    from template_lib.d2.data.build_cifar10 import data_path, registed_name_list, registed_func_list, kwargs_list
    from detectron2.data import MetadataCatalog

    dataset_dicts = registed_func_list[3](name=registed_name_list[3], data_path=data_path, **kwargs_list[3])
    metadata = MetadataCatalog.get(registed_name_list[2])
    for d in random.sample(dataset_dicts, 3):
      img = d["image"]
      file_name = str(d['image_id']) + '.jpg'
      img.save(os.path.join(outdir, file_name))
      pass

  def test_cifar10_per_class(self):
    """

    """
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
      os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    if 'PORT' not in os.environ:
      os.environ['PORT'] = '6006'
    if 'TIME_STR' not in os.environ:
      os.environ['TIME_STR'] = '0' if utils.is_debugging() else '0'
    # func name
    assert sys._getframe().f_code.co_name.startswith('test_')
    command = sys._getframe().f_code.co_name[5:]
    class_name = self.__class__.__name__[7:] \
      if self.__class__.__name__.startswith('Testing') \
      else self.__class__.__name__
    outdir = f'results/{class_name}/{command}'
    os.makedirs(outdir, exist_ok=True)

    import matplotlib.pylab as plt
    from template_lib.d2.data.build_cifar10_per_class import data_path, kwargs_list
    from detectron2.data import MetadataCatalog

    pass


class TestingBuildCIFAR100(unittest.TestCase):

  def test_cifar100_train(self):
    """

    """
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
      os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    if 'PORT' not in os.environ:
      os.environ['PORT'] = '6006'
    if 'TIME_STR' not in os.environ:
      os.environ['TIME_STR'] = '0' if utils.is_debugging() else '0'
    # func name
    assert sys._getframe().f_code.co_name.startswith('test_')
    command = sys._getframe().f_code.co_name[5:]
    class_name = self.__class__.__name__[7:] \
      if self.__class__.__name__.startswith('Testing') \
      else self.__class__.__name__
    outdir = f'results/{class_name}/{command}'
    os.makedirs(outdir, exist_ok=True)

    import matplotlib.pylab as plt
    from template_lib.d2.data.build_cifar100 import data_path, registed_name_list, registed_func_list, kwargs_list
    from detectron2.data import MetadataCatalog

    dataset_dicts = registed_func_list[0](name=registed_name_list[0], data_path=data_path, **kwargs_list[0])
    metadata = MetadataCatalog.get(registed_name_list[0])
    for d in random.sample(dataset_dicts, 3):
      img = d["image"]
      file_name = str(d['image_id']) + '.jpg'
      img.save(os.path.join(outdir, file_name))
      pass