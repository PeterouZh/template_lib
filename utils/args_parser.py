import argparse


def none_or_str(value):
  if value == 'None':
    return None
  return value


def true_or_false(value):
  if value == 'True':
    return True
  return False


def build_parser():
  parser = argparse.ArgumentParser()
  parser.add_argument('--config', type=none_or_str, default='')
  parser.add_argument('--resume', type=true_or_false, default=False)
  parser.add_argument('--resume_path', type=none_or_str, default='')
  parser.add_argument('--resume_root', type=none_or_str, default='')
  return parser
