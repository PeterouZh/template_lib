import os
import multiprocessing
import random
import time
from .utils_func import is_debugging, get_prefix_abb, get_eval_attr
from . import config
from . import args_parser


class TorchResnetWorker(multiprocessing.Process):
  def run(self):
    bs, gpu, determine_bs, q = self._args
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu
    import torch
    import torch.nn.functional as F
    import torchvision
    net = torchvision.models.resnet152().cuda()
    net = torch.nn.DataParallel(net).cuda()

    if determine_bs:
      self.determine_bs(net, q)
    else:
      self.train(net, bs)

  def train(self, net, bs):
    try:
      import torch
      import torch.nn.functional as F
      rbs = bs
      print(bs)
      while True:
        t = random.random()
        time.sleep(t)

        x = torch.rand(rbs, 3, 224, 224).cuda()
        y = net(x)

        tensor = torch.randint(0, 1000, (rbs,))  # tensor([0, 1, 2, 0, 1])
        one_hot = F.one_hot(tensor, num_classes=1000).float().cuda()
        loss = (y - one_hot).mean()
        loss.backward()

        t = random.random()
        time.sleep(t)
        rbs = random.randint(1, bs)
    except RuntimeError:
      torch.cuda.empty_cache()
      pass

  def determine_bs(self, net, q):
    import torch
    import torch.nn.functional as F
    bs = 0
    try:
      while True:
        bs += 1
        print('%s' % bs)
        x = torch.rand(bs, 3, 224, 224).cuda()
        y = net(x)

        tensor = torch.randint(0, 1000, (bs,))  # tensor([0, 1, 2, 0, 1])
        one_hot = F.one_hot(tensor, num_classes=1000).float().cuda()
        loss = (y - one_hot).mean()
        loss.backward()
    except RuntimeError:
      torch.cuda.empty_cache()
      q.put(bs - 1)
      # import traceback
      # print(traceback.format_exc())

