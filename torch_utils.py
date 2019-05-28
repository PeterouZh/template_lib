import pprint
import sys
import os
from collections import OrderedDict

from .logging_utils import get_logger
from .utils import save_args_pickle_and_txt
from .parse_config import YamlConfigParser


def get_add_argument(parser, args):
  """
  Usage:
      import peng_lib.torch_utils
      import argparse
      my_parser = argparse.ArgumentParser(description='MyParser', parents=[parser], add_help=False)
      parser = argparse.ArgumentParser(description='Dataset')
      args = parser.parse_args()
      ADD_argument = peng_lib.torch_utils.get_add_argument(parser, args)
      od = ADD_argument.od
  Get file and func name:
      ADD_argument('python_file', __file__, '')
      ADD_argument('time_str', time.strftime("%Y%m%d-%H%M%S"))
      ADD_argument('func_call', sys._getframe().f_code.co_name, '')
      ADD_argument('outdir', 'results/%s' % args.func_call)
  :return:
  """
  od = OrderedDict()

  def ADD_argument(key, value, help=None):
    od[key] = value
    if key in args:
      parser.set_defaults(**{key: value})
      setattr(args, key, value)
      return

    if isinstance(value, str):
      parser.add_argument('--%s' % key, type=str, default=value, help=help)
    elif isinstance(value, bool):
      parser.add_argument('--%s' % key, type=bool, default=value, help=help)
    elif isinstance(value, int):
      parser.add_argument('--%s' % key, type=int, default=value, help=help)
    elif isinstance(value, float):
      parser.add_argument('--%s' % key, type=float, default=value, help=help)
    elif isinstance(value, list) and isinstance(value[0], int):
      parser.add_argument('--%s' % key, type=list, default=value, help=help)
    elif isinstance(value, list) and isinstance(value[0], float):
      parser.add_argument('--%s' % key, type=list, default=value, help=help)
    elif isinstance(value, list) and isinstance(value[0], str):
      parser.add_argument('--%s' % key, type=list, default=value, help=help)
    else:
      assert 0
    setattr(args, key, value)
    return

  ADD_argument.od = od
  return ADD_argument


class TensorBoardTool:
  """ Run tensorboard in python
      Usage:
          from peng_lib.torch_utils import TensorBoardTool
          tbtool = TensorBoardTool(dir_path=args.tbdir)
          writer = tbtool.run()
          tbtool.add_text_args_and_od(args, od)
  """

  def __init__(self, dir_path, tb_logdir):
    self.dir_path = dir_path
    self.tb_logdir = tb_logdir

  def run(self):
    """ Launch tensorboard and create summary_writer

    :return:
    """
    import logging
    from tensorboard import default
    from tensorboard import program

    # Remove http messages
    logging.getLogger('tensorflow').setLevel(logging.ERROR)
    logging.getLogger('werkzeug').setLevel(logging.ERROR)
    # Start tensorboard server
    tb = program.TensorBoard(default.get_plugins())
    # tb = program.TensorBoard(default.get_plugins(), default.get_assets_zip_provider())
    port = os.getenv('PORT', '6006')

    tb.configure(argv=[None, '--logdir', self.tb_logdir, '--port', port])
    url = tb.launch()
    sys.stdout.write('TensorBoard at %s \n' % url)

    # create tensorboardx writer
    writer = self.SummmaryWriter()
    self.writer = writer
    return writer

  def SummmaryWriter(self):
    from tensorboardX import SummaryWriter
    writer = SummaryWriter(log_dir=self.dir_path)
    return writer

  def add_text_args_and_od(self, args, od):
    od_md = self.args_as_markdown_no_sorted_(od)
    self.writer.add_text('od', od_md, 0)

    default_args = set(vars(args)) - set(od)
    default_args = {key: vars(args)[key] for key in default_args}
    default_args_md = self.args_as_markdown_sorted_(default_args)
    self.writer.add_text('args', default_args_md, 0)

  @staticmethod
  def args_as_markdown(args):
    """ Return configs as markdown format """
    text = "|name|value|  \n|-|-|  \n"
    for attr, value in sorted(args.items()):
      text += "|{}|{}|  \n".format(attr, value)

    return text

  @staticmethod
  def args_as_markdown_no_sorted_(args):
    """ Return configs as markdown format

    :param args: dict
    :return:
    """
    text = "|name|value|  \n|:-:|:-:|  \n"
    for attr, value in args.items():
      text += "|{}|{}|  \n".format(attr, value)
    return text

  @staticmethod
  def args_as_markdown_sorted_(args):
    """ Return configs as markdown format """
    text = "|name|value|  \n|-|-|  \n"
    for attr, value in sorted(args.items()):
      text += "|{}|{}|  \n".format(attr, value)
    return text


class CheckpointTool(object):
  def __init__(self, ckptdir):
    self.ckptdir = ckptdir
    os.makedirs(ckptdir, exist_ok=True)
    pass

  def save_checkpoint(self, checkpoint_dict, is_best=True, filename='ckpt.tar'):
    """

    :param checkpoint_dict: dict
    :param is_best:
    :param filename:
    :return:
    """
    import torch, shutil
    filename = os.path.join(self.ckptdir, filename)
    state_dict = {}
    for key in checkpoint_dict:
      if hasattr(checkpoint_dict[key], 'state_dict'):
        state_dict[key] = getattr(checkpoint_dict[key], 'state_dict')()
      else:
        state_dict[key] = checkpoint_dict[key]

    torch.save(state_dict, filename)
    if is_best:
      shutil.copyfile(filename, filename + '.best')

  def load_checkpoint(self, checkpoint_dict, resumepath):
    """
        args.start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
    :param resumepath:
    :return:
    """
    import torch
    if os.path.isfile(resumepath):
      state_dict = torch.load(resumepath)

      for key in checkpoint_dict:
        if hasattr(checkpoint_dict[key], 'state_dict'):
          checkpoint_dict[key].load_state_dict(state_dict.pop(key))

      return state_dict
    else:
      print("=> no checkpoint found at '{}'".format(resumepath))
      assert 0


def get_tbwriter_logger_checkpoint(args, od, myargs, outdir, tbdir, logfile, ckptdir, **kwargs):
  # save args to pickle
  save_args_pickle_and_txt(args, os.path.join(outdir, 'args.pk'))

  # logger
  logger = get_logger(logfile, stream=True, propagate=False)
  myargs.logger = logger
  logger.info(pprint.pformat(od))

  # config.json
  if hasattr(args, 'config'):
    logger.info('=> Load config.yaml.')
    config_parser = YamlConfigParser(fname=args.config, saved_fname=od['configfile'])
    config = config_parser.config_dotdict
    myargs.config = config
    config_str = pprint.pformat(myargs.config)
    logger.info_msg(config_str)
    config_str = config_str.strip().replace('\n', '  \n>')
    pass

  # tensorboard
  tbtool = TensorBoardTool(dir_path=tbdir, tb_logdir=od['tb_logdir'])
  writer = tbtool.run()
  tbtool.add_text_args_and_od(args, od)
  myargs.writer = writer

  # checkpoint
  checkpoint = CheckpointTool(ckptdir=ckptdir)
  myargs.checkpoint = checkpoint
  myargs.checkpoint_dict = OrderedDict()

  if hasattr(args, 'config'):
    myargs.writer.add_text('config', config_str, 0)
    pass


def ignorePath(path):
  """
  For shutil.copytree func
  :param path:
  :return:
  """
  path = [os.path.abspath(elem) for elem in path]

  def ignoref(directory, contents):
    ig = [f for f in contents if os.path.abspath(os.path.join(directory, f)) in path]
    return ig

  return ignoref


def set_random_seed(manualSeed=1234):
  import torch
  import numpy as np
  import random
  random.seed(manualSeed)
  np.random.seed(manualSeed)

  torch.manual_seed(manualSeed)
  # if you are suing GPU
  torch.cuda.manual_seed(manualSeed)
  torch.cuda.manual_seed_all(manualSeed)

  torch.backends.cudnn.enabled = False
  torch.backends.cudnn.benchmark = False
  torch.backends.cudnn.deterministic = True
  return
