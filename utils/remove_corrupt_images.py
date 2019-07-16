import os
from PIL import Image
import glob
import unittest
import tqdm

from . import logging_utils


def remove_corrupt_images(imgs_dir, ext='png'):
  logger = logging_utils.logging_init()
  imgs_list = sorted(glob.glob(os.path.join(imgs_dir, '**/*.' + ext)))
  for img_path in tqdm.tqdm(imgs_list):
    try:
      img = Image.open(img_path)  # open the image file
      img.verify()  # verify that it is, in fact an image
    except (IOError, SyntaxError) as e:
      logger.info_msg('Bad file:', img_path)  # print out the names of corrupt files


class TestingUnit(unittest.TestCase):

  def test_Case(self):
    imgs_dir = '/media/shhs/Peterou2/user/code/ffhq-dataset/thumbnails128x128'
    remove_corrupt_images(imgs_dir)