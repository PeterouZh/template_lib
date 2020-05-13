import re
import logging
from numpy import array_equal
from torch.nn.parallel import DistributedDataParallel


def _make_gen(reader):
  b = reader(1024 * 1024)
  while b:
    yield b
    b = reader(1024 * 1024)


def rawgencount(filename):
  "count num of lines of a file"
  f = open(filename, 'rb')
  f_gen = _make_gen(f.raw.read)
  n_lines = sum(buf.count(b'\n') for buf in f_gen)
  f.close()
  return n_lines

def array_eq_in_list(myarr, list_arrays):
  return next((True for elem in list_arrays if array_equal(elem, myarr)), False)


def top_accuracy(output, target, topk=(1,)):
  """ Computes the precision@k for the specified values of k """
  maxk = max(topk)
  batch_size = target.size(0)

  _, pred = output.topk(maxk, 1, True, True)
  pred = pred.t()
  # one-hot case
  if target.ndimension() > 1:
    target = target.max(1)[1]

  correct = pred.eq(target.view(1, -1).expand_as(pred))

  res = []
  for k in topk:
    correct_k = correct[:k].view(-1).float().sum(0)
    res.append(correct_k.mul_(1.0 / batch_size))

  return res


class AverageMeter():
  """ Computes and stores the average and current value """

  def __init__(self):
    self.reset()

  def reset(self):
    """ Reset all statistics """
    self.val = 0
    self.avg = 0
    self.sum = 0
    self.count = 0

  def update(self, val, n=1):
    """ Update statistics """
    self.val = val
    self.sum += val * n
    self.count += n
    self.avg = self.sum / self.count


def print_number_params(models_dict):
  logger = logging.getLogger('tl')
  for label, model in models_dict.items():
    logger.info('Number of params in {}:\t {}M'.format(
      label, sum([p.data.nelement() for p in model.parameters()])/1e6
    ))


def get_ddp_attr(obj, attr, **kwargs):
  return getattr(getattr(obj, 'module', obj), attr, **kwargs)


def get_ddp_module(model):
  if isinstance(model, DistributedDataParallel):
    model = model.module
  return model



def get_eval_attr(obj, name, default=None, **kwargs):
  if hasattr(obj, name):
    value = getattr(obj, name)
    value = eval(value, kwargs)
  else:
    value = default
  return value


def get_attr_eval(obj, name, **kwargs):
  if hasattr(obj, name):
    value = getattr(obj, name)
    value = eval(value, kwargs)
  else:
    value = kwargs['default']
  return value


def get_attr_kwargs(obj, name, **kwargs):
  if hasattr(obj, name):
    value = getattr(obj, name)
    if isinstance(value, str) and value.startswith('kwargs['):
      value = eval(value)
  elif name in kwargs:
    value = kwargs[name]
  else:
    value = kwargs['default']
  return value


def get_attr_format(obj, name, default=None, **kwargs):
  if hasattr(obj, name):
    value = getattr(obj, name)
    if isinstance(value, str):
      value = value.format(**kwargs)
  else:
    if default is None:
      raise AttributeError
    value = default
  return value


def is_debugging():
  import sys
  gettrace = getattr(sys, 'gettrace', None)

  if gettrace is None:
    assert 0, ('No sys.gettrace')
  elif gettrace():
    return True
  else:
    return False


def get_prefix_abb(prefix):
  # prefix_split = prefix.split('_')
  prefix_split = re.split('_|/', prefix)
  if len(prefix_split) == 1:
    prefix_abb = prefix
  else:
    prefix_abb = ''.join([k[0] for k in prefix_split])
  return prefix_abb


