import torch
import sys

__all__ = ['get_sample_imgs_list', "get_sample_imgs_list_ddp"]

def get_sample_imgs_list(sample_func, num_imgs=50000, as_numpy=True, stdout=sys.stdout):
  """

  :param sample_func: return imgs (b, c, h, w), range [-1, 1]
  :param num_imgs:
  :param stdout:
  :return:
  """

  img_list = list()
  imgs = sample_func()
  bs = imgs.size(0)
  eval_iter = num_imgs // bs
  for iter_idx in range(eval_iter):
    print('\r',
          end='sample images [%d/%d]' % (iter_idx * bs, eval_iter * bs),
          file=stdout, flush=True)

    # Generate a batch of images
    imgs = sample_func()
    gen_imgs = imgs.mul_(127.5).add_(127.5).clamp_(0.0, 255.0) \
      .permute(0, 2, 3, 1).type(torch.uint8)
    if as_numpy:
      gen_imgs = gen_imgs.to('cpu').numpy()
    img_list.extend(list(gen_imgs))
  print('', file=stdout)
  return img_list


def get_sample_imgs_list_ddp(sample_func, num_imgs=50000, stdout=sys.stdout):
  """

  :param sample_func:
  :param num_imgs:
  :param stdout:
  :return:
  """
  import torch
  from template_lib.d2.utils import comm
  ws = torch.distributed.get_world_size()
  num_imgs = num_imgs // ws
  img_list = get_sample_imgs_list(sample_func=sample_func, num_imgs=num_imgs, as_numpy=False, stdout=stdout)
  imgs = torch.stack(img_list)

  imgs_list = comm.gather(imgs)
  ret_img_list = []
  if len(imgs_list) > 0:
    imgs = torch.cat(imgs_list, dim=0).to('cuda')
    ret_img_list.extend(list(imgs.to('cpu').numpy()))
  return ret_img_list