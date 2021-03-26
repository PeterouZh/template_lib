import os
import sys
import unittest
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np
import matplotlib.pyplot as plt
import copy

from template_lib import utils


def open_video(video_file):
  cap = cv2.VideoCapture(str(video_file))
  ret_success, frame = cap.read()
  return ret_success, frame


def cv2_to_pil(img):
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  im_pil = Image.fromarray(img)
  return im_pil


class VideoWriter(object):
  def __init__(self, outfile, w, h, fps):
    self.w = w
    self.h = h
    out_size = (w, h)
    self.video = cv2.VideoWriter(outfile, cv2.VideoWriter_fourcc(*'mp4v'), fps, out_size)
    pass

  def write(self, image, is_tensor=True, rgb=True):
    if is_tensor:
      from torchvision.transforms.functional import to_pil_image
      image = to_pil_image(image)
    image = np.array(image)
    if rgb:
      image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    assert image.shape[:2] == (self.h, self.w)
    self.video.write(image)
    pass

  def release(self):
    self.video.release()
    pass


class ImageioVideoWriter(object):
  def __init__(self, outfile, fps, **kwargs):
    """
    pip install imageio-ffmpeg opencv-python

    """
    import imageio
    self.video = imageio.get_writer(outfile, fps=fps)
    pass

  def write(self, image, dst_size=None, **kwargs):
    if dst_size is not None:
      w, h = self._get_size(w=image.size[0], h=image.size[1], dst_size=dst_size, for_min_edge=False)
      image = image.resize((w, h), Image.LANCZOS)
    self.video.append_data(np.array(image))
    pass

  def release(self):
    self.video.close()
    pass

  def _get_size(self, w, h, dst_size, for_min_edge=True):
    if for_min_edge:
      edge = min(w, h)
    else:
      edge = max(w, h)

    w = int(dst_size / edge * w)
    h = int(dst_size / edge * h)
    return w, h


class Testing_cv2_utils(unittest.TestCase):

  def test_zoom_in_video_writer(self, debug=True):
    """
    Usage:
        export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
        export TIME_STR=1
        export PYTHONPATH=./exp:./stylegan2-pytorch:./
        python 	-c "from exp.tests.test_styleganv2 import Testing_stylegan2;\
          Testing_stylegan2().test_train_ffhq_128()"

    :return:
    """
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
      os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    if 'TIME_STR' not in os.environ:
      os.environ['TIME_STR'] = '0' if utils.is_debugging() else '0'
    from template_lib.v2.config_cfgnode.argparser import \
      (get_command_and_outdir, setup_outdir_and_yaml, get_append_cmd_str, start_cmd_run)

    tl_opts = ' '.join(sys.argv[sys.argv.index('--tl_opts') + 1:]) if '--tl_opts' in sys.argv else ''
    print(f'tl_opts:\n {tl_opts}')

    command, outdir = get_command_and_outdir(self, func_name=sys._getframe().f_code.co_name, file=__file__)
    argv_str = f"""
                --tl_config_file none
                --tl_command none
                --tl_outdir {outdir}
                --tl_opts {tl_opts}
                """
    args, cfg = setup_outdir_and_yaml(argv_str, return_cfg=True)

    n_gpus = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
    import tqdm
    from template_lib.proj.pil.pil_utils import get_size

    img_path = "template_lib/datasets/images/zebra_GT_target_origin.png"
    outvid = f"{args.tl_outdir}/test.mp4"

    out_size = 2048
    image = Image.open(img_path)
    w, h = image.size
    max_scale = out_size / min(w, h)
    out_w, out_h = get_size(w=w, h=h, dst_size=out_size)

    out_video = VideoWriter(outfile=outvid, w=out_w, h=out_h, fps=10)

    for scale in tqdm.tqdm(np.arange(1, max_scale, 0.05)):
      out_img = Image.new(mode='RGB', size=(out_w, out_h), color='black')
      cur_w, cur_h = int(w * scale), int(h * scale)
      cur_image = image.resize((cur_w, cur_h), resample=Image.NEAREST)
      xy = (out_w - cur_w) // 2, (out_h - cur_h) // 2
      out_img.paste(cur_image, xy)
      out_video.write(out_img, is_tensor=False, rgb=True)
    out_video.release()
    pass







