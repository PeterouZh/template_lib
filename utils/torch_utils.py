import pprint
import sys
import os
from collections import OrderedDict

from .logging_utils import get_logger
from .config_utils import YamlConfigParser


class TensorBoardTool(object):
  """ Run tensorboard in python
      Usage:
          from peng_lib.torch_utils import TensorBoardTool
          tbtool = TensorBoardTool(dir_path=args.tbdir)
          writer = tbtool.run()
          tbtool.add_text_args_and_od(args, od)
  """

  def __init__(self, tbdir):
    self.tbdir = tbdir

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

    tb.configure(argv=[None, '--logdir', self.tbdir, '--port', port])
    url = tb.launch()
    sys.stdout.write('TensorBoard at %s \n' % url)

    # create tensorboardx writer
    writer = self.SummmaryWriter()
    self.writer = writer
    return writer

  def SummmaryWriter(self):
    from tensorboardX import SummaryWriter
    writer = SummaryWriter(logdir=self.tbdir)
    return writer

  def add_text_str_args(self, args, name):
    args_str = pprint.pformat(args)
    args_str = args_str.strip().replace('\n', '  \n>')
    self.writer.add_text(name, args_str, 0)

  def add_text_md_args(self, args, name):
    args_md = self.args_as_markdown_no_sorted_(args)
    self.writer.add_text(name, args_md, 0)

  def add_text_args_and_od(self, args, od):
    od_md = self.args_as_markdown_no_sorted_(od)
    self.writer.add_text('od', od_md, 0)

    default_args = set(vars(args)) - set(od)
    default_args = {key: vars(args)[key] for key in default_args}
    default_args_md = self.args_as_markdown_sorted_(default_args)
    self.writer.add_text('args', default_args_md, 0)

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
