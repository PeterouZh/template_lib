import os
import sys
import unittest
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import copy

from template_lib import utils


# __all__ = ['add_text_in_tensor', ]

_font = r'template_lib/datasets/sans-serif.ttf'

def add_text_in_tensor(img_tensor,
                       text,
                       xy=(0, 0),
                       size=16,
                       color=(255, 0, 0)):
  import torchvision
  import torchvision.transforms.functional as tv_trans_f

  unsqueeze = True if img_tensor.dim() == 4 else False

  if img_tensor.dim() == 3:
    img_tensor = img_tensor.unsqueeze(0)
  assert img_tensor.dim() == 4 and img_tensor.size(0) == 1

  img_tensor = torchvision.utils.make_grid(img_tensor.cpu(), nrow=1, padding=0, normalize=True)
  img_pil = tv_trans_f.to_pil_image(img_tensor)

  font = ImageFont.truetype(font=_font, size=size)
  draw = ImageDraw.Draw(img_pil)
  draw.text(xy=xy, text=text, fill=color, font=font)

  img_tensor = tv_trans_f.to_tensor(img_pil)
  if unsqueeze:
    img_tensor = img_tensor.unsqueeze(0)
  return img_tensor


def sr_merge_original_image_and_patches(image,
                                        lefts,
                                        uppers,
                                        w,
                                        h,
                                        pad=3,
                                        width=2,
                                        patch_width=1,
                                        outlines=('#ff0000', '#0000ff')):
  """
  lefts, uppers: []
  """
  image = copy.deepcopy(image)
  im_w, im_h = image.size
  resized_w = im_w // 2 - pad
  resized_h = int((resized_w / w) * h)

  out_size = (im_w, im_h + pad + resized_h)
  out_image = Image.new('RGB', out_size, "white")

  start_w = 0
  for left, upper, outline in zip(lefts, uppers, outlines):

    # Add rectangle on image
    patch = draw_rectangle_and_return_crop(
      image=image, left=left, upper=upper, w=w, h=h, width=width, patch_width=patch_width,
      outline=outline)

    resized_patch = patch.resize((resized_w, resized_h), resample=Image.NEAREST)
    out_image.paste(resized_patch, (start_w, im_h + pad))
    start_w = im_w - resized_w
  out_image.paste(image, (0, 0))
  return out_image


def draw_rectangle_and_return_crop(image, left, upper, w, h, outline='#ff0000', width=2, patch_width=1):
  right = left + w
  lower = upper + h
  bbox = ((left, upper), (right, lower))

  patch = image.crop((left, upper, right, lower))
  draw_patch = ImageDraw.Draw(patch)
  patch_bbox = ((0, 0, patch.size[0]-patch_width, patch.size[1]-patch_width))
  draw_patch.rectangle(patch_bbox, outline=outline, width=patch_width)

  draw = ImageDraw.Draw(image)
  draw.rectangle(bbox, outline=outline, width=width)
  return patch


class Testing_pil_utils(unittest.TestCase):

  def test_draw_text_on_rectangle(self, debug=True):
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

    img_path = "template_lib/datasets/images/zebra_GT_target_origin.png"

    source_img = Image.open(img_path)
    font = ImageFont.truetype(_font)
    text = "very loooooooooooooooooong text"

    # get text size
    text_size = font.getsize(text)
    # set button size + 10px margins
    button_size = (text_size[0] + 20, text_size[1] + 20)
    # create image with correct size and black background
    button_img = Image.new('RGB', button_size, "black")

    # put text on button with 10px margins
    button_draw = ImageDraw.Draw(button_img)
    button_draw.text((10, 10), text, font=font)

    # put button on source image in position (0, 0)
    source_img.paste(button_img, (0, 0))

    # save in new file
    plt.imshow(source_img)
    plt.show()
    # source_img.save("output.jpg", "JPEG")

    pass

  def test_draw_rectangle(self, debug=True):
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

    img_path = "template_lib/datasets/images/zebra_GT_target_origin.png"

    im = Image.open(img_path)
    draw = ImageDraw.Draw(im)
    draw.rectangle(((10, 10), (50, 50)), outline='#ff0000', width=2)

    plt.imshow(im)
    plt.show()
    pass

  def test_sr_merge_original_image_and_patches(self, debug=True):
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

    img_path = "template_lib/datasets/images/zebra_GT_target_origin.png"

    left = [80, 300]
    upper = [120, 140]
    w = 50
    h = 20
    pad = 2

    image = Image.open(img_path)

    out_image = sr_merge_original_image_and_patches(image=image, lefts=left, uppers=upper, w=w, h=h, pad=pad,
                                                    width=3, patch_width=1)

    fig, axes = plt.subplots(1, 1)
    axes.imshow(out_image)
    # axes[1].imshow(patch)
    fig.show()
    pass

  def test_add_text_in_tensor(self, debug=True):
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
    import torchvision.transforms.functional as trans_f

    img_path = "template_lib/datasets/images/zebra_GT_target_origin.png"
    left = [80, 300]
    upper = [120, 140]
    w = 50
    h = 20
    pad = 2

    img = Image.open(img_path)

    img = sr_merge_original_image_and_patches(img, lefts=left, uppers=upper, w=w, h=h, pad=pad)

    img_tensor = trans_f.to_tensor(img)

    img_tensor_text = add_text_in_tensor(img_tensor=img_tensor, text='Image', xy=(5, 5))

    img_pil = trans_f.to_pil_image(img_tensor_text)

    fig, axes = plt.subplots()
    axes.imshow(img_pil)
    fig.show()
    pass
