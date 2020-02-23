import re


def get_eval_attr(attr, context_dict):
  if isinstance(attr, str):
    return eval(attr, context_dict)
  else:
    return attr


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