import re
import logging


def print_number_params(models_dict):
  logger = logging.getLogger('tl')
  for label, model in models_dict.items():
    logger.info('Number of params in {}:\t {}M'.format(
      label, sum([p.data.nelement() for p in model.parameters()])/1e6
    ))


def get_ddp_attr(obj, attr, default=None):
  return getattr(getattr(obj, 'module', obj), attr, default)


def get_eval_attr(obj, name, context_dict, default=None):
  if hasattr(obj, name):
    value = getattr(obj, name)
    value = eval(value, context_dict)
  else:
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