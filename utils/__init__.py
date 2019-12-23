import os
import multiprocessing
import random
import time
from . import config, config_utils, args_parser, modelarts_utils


def is_debugging():
  import sys
  gettrace = getattr(sys, 'gettrace', None)

  if gettrace is None:
    assert 0, ('No sys.gettrace')
  elif gettrace():
    return True
  else:
    return False


class TorchResnetWorker(multiprocessing.Process):
  def run(self):
    bs, gpu = self._args
    print('bs: %s'%bs)
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu
    import torch
    import torch.nn.functional as F
    import torchvision
    net = torchvision.models.resnet152().cuda()
    net = torch.nn.DataParallel(net).cuda()

    rbs = bs
    try:
      while True:
        t = random.random()
        time.sleep(t)

        x = torch.rand(rbs, 3, 299, 299).cuda()
        y = net(x)

        tensor = torch.randint(0, 1000, (bs,))  # tensor([0, 1, 2, 0, 1])
        one_hot = F.one_hot(tensor, num_classes=1000).float().cuda()
        loss = (y - one_hot).mean()
        loss.backward()

        t = random.random()
        time.sleep(t)
        rbs = random.randint(1, bs)
    except RuntimeError:
      pass
