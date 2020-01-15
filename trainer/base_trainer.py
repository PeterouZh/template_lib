import functools
import os, collections
import re
import sys
import time
from collections import defaultdict

# from ..utils import modelarts_utils
from template_lib.utils import modelarts_utils


def get_prefix_abb(prefix):
  # prefix_split = prefix.split('_')
  prefix_split = re.split('_|/', prefix)
  if len(prefix_split) == 1:
    prefix_abb = prefix
  else:
    prefix_abb = ''.join([k[0] for k in prefix_split])
  return prefix_abb

def write_scalars_to_text(summary, prefix, step, textlogger,
                          log_axe, log_axe_sec, log_together=False):
  prefix_abb = get_prefix_abb(prefix=prefix)
  summary = {prefix_abb + '.' + k: v for k, v in summary.items()}
  textlogger.log(step, **summary)
  time_str = prefix + '-' + '-'.join(summary.keys())
  if log_axe:
    now = time.time()
    last_time = getattr(Trainer, time_str, 0)
    if now - last_time > log_axe_sec:
      if log_together:
        textlogger.log_ax(**summary)
      else:
        textlogger.log_axes(**summary)
      setattr(Trainer, time_str, now)


class Trainer(object):
  def __init__(self, args, myargs):
    self.args = args
    self.myargs = myargs
    self.init_static_method()
    self.config = myargs.config
    self.logger = myargs.logger
    self.train_dict = self.init_train_dict()

    # self.dataset_load()
    self.model_create()
    self.optimizer_create()
    self.scheduler_create()
    pass

  def init_static_method(self):
    self.summary_scalars = functools.partial(
      self.summary_scalars, writer=self.myargs.writer,
      textlogger=self.myargs.textlogger)
    self.summary_scalars_together = functools.partial(
      self.summary_scalars_together, writer=self.myargs.writer,
      textlogger=self.myargs.textlogger)
    self.summary_dicts = functools.partial(
      self.summary_dicts, writer=self.myargs.writer,
      textlogger=self.myargs.textlogger)

  def init_train_dict(self, ):
    train_dict = collections.OrderedDict()
    train_dict['epoch_done'] = 0
    train_dict['batches_done'] = 0
    self.myargs.checkpoint_dict['train_dict'] = train_dict
    return train_dict

  def dataset_load(self):
    raise NotImplemented

  def model_create(self):
    pass

  def print_number_params(self, models):
    for label, model in models.items():
      self.logger.info('Number of params in {}:\t {}M'.format(
        label, sum([p.data.nelement() for p in model.parameters()])/1e6
      ))

  def save_checkpoint(self, filename='ckpt.tar'):
    self.myargs.checkpoint.save_checkpoint(
      checkpoint_dict=self.myargs.checkpoint_dict, filename=filename)

  def load_checkpoint(self, filename='ckpt.tar'):
    state_dict = self.myargs.checkpoint.load_checkpoint(
      checkpoint_dict=self.myargs.checkpoint_dict, filename=filename)
    return state_dict

  def optimizer_create(self):
    pass

  def scheduler_create(self):
    pass

  def scheduler_step(self, epoch):
    pass

  @staticmethod
  def resume(myargs, resume_path):
    myargs.logger.info_msg('=> Resume from: \n%s', resume_path)
    loaded_state_dict = myargs.checkpoint.load_checkpoint(
      checkpoint_dict=myargs.checkpoint_dict,
      filename=resume_path)
    for key in myargs.train_dict:
      if key in loaded_state_dict['train_dict']:
        myargs.train_dict[key] = loaded_state_dict['train_dict'][key]

  def finetune(self, ):
    config = self.config.finetune
    self.args.finetune_path = config.finetune_path
    modelarts_utils.modelarts_finetune(
      self.args, finetune_path=config.finetune_path)
    if config.load_model:
      self.logger.info_msg('Loading finetune model weights.')
      filename = os.path.join(config.finetune_path, 'models/ckpt.tar')
      state_dict = self.load_checkpoint(filename=filename)
    pass

  def train(self):
    try:
      self.train_()
    except:
      from template_lib.utils import modelarts_utils
      modelarts_utils.modelarts_record_jobs(self.args, self.myargs,
                                            str_info='Exception!')
      import traceback
      print(traceback.format_exc())
      self.modelarts(join=True)

  def train_(self, ):
    config = self.config
    for epoch in range(self.train_dict['epoch_done'], config.epochs):
      self.logger.info('epoch: [%d/%d]' % (epoch, config.epochs))
      self.scheduler_step(epoch=epoch)
      self.train_one_epoch()

      self.train_dict['epoch_done'] += 1
      # test
      self.test()
    self.finalize()

  def train_one_epoch(self):
    raise NotImplemented

  @staticmethod
  def summary_scalars(summary, prefix, step,
                      writer=None, textlogger=None,
                      log_axe=True, log_axe_sec=300):
    if writer is not None:
      for key in summary:
        writer.add_scalar(prefix + '/%s' % key, summary[key], step)

    if textlogger is not None:
      write_scalars_to_text(summary=summary, prefix=prefix, step=step,
                            textlogger=textlogger,
                            log_axe=log_axe, log_axe_sec=log_axe_sec)
    if writer is None and textlogger is None:
      print('Both writer and textlogger are None!')


  @staticmethod
  def summary_scalars_together(summary, prefix, step,
                               writer=None, textlogger=None,
                               log_axe=True, log_axe_sec=300):
    prefix = prefix + '_together'
    if writer is not None:
      writer.add_scalars(prefix, summary, step)
    if textlogger is not None:
      write_scalars_to_text(summary=summary, prefix=prefix, step=step,
                            textlogger=textlogger,
                            log_axe=log_axe, log_axe_sec=log_axe_sec,
                            log_together=True)
    if writer is None and textlogger is None:
      print('Both writer and textlogger are None!')

  @staticmethod
  def summary_dicts(summary_dicts, prefix, step,
                    writer=None, textlogger=None,
                    log_axe=True, log_axe_sec=300):
    for summary_n, summary_v in summary_dicts.items():
      if summary_n == 'scalars':
        Trainer.summary_scalars(
          summary_v, prefix=prefix + '/' + summary_n, step=step,
          writer=writer, textlogger=textlogger,
          log_axe=log_axe, log_axe_sec=log_axe_sec)
      else:
        Trainer.summary_scalars_together(
          summary_v, prefix=prefix + '/' + summary_n, step=step,
          writer=writer, textlogger=textlogger,
          log_axe=log_axe, log_axe_sec=log_axe_sec)

  @staticmethod
  def summary_dict(summary_dict, prefix, step,
                   writer=None, textlogger=None,
                   log_axe=True, log_axe_sec=300):

    summary_dicts = defaultdict(dict)
    summary_dicts['scalars'].update(summary_dict)
    Trainer.summary_dicts(summary_dicts=summary_dicts, prefix=prefix, step=step,
                          writer=writer, textlogger=textlogger,
                          log_axe=log_axe, log_axe_sec=log_axe_sec)

  def summary_figures(self, summary_dicts, prefix):
    # prefix_abb = self.get_prefix_abb(prefix)
    # for summary_n, summary_v in summary_dicts.items():
    #   summary_v = {prefix_abb + '.' + k: v for k, v in summary_v.items()}
    #   if summary_n == 'scalars':
    #     self.myargs.textlogger.log_axes(**summary_v)
    #   else:
    #     self.myargs.textlogger.log_ax(**summary_v)
    pass

  def evaluate(self):
    raise NotImplemented

  def modelarts(self, join=False, end=False):
    modelarts_utils.modelarts_sync_results(self.args, self.myargs,
                                           join=join, end=end)



from template_lib import utils
import unittest, argparse

class TestingUnit(unittest.TestCase):

  def test_summary_scalars(self):
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
      os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    if 'PORT' not in os.environ:
      os.environ['PORT'] = '6006'
    if 'TIME_STR' not in os.environ:
      os.environ['TIME_STR'] = '0' if utils.is_debugging() else '1'
    # func name
    outdir = os.path.join('results', sys._getframe().f_code.co_name)
    myargs = argparse.Namespace()

    def build_args():
      argv_str = f"""
            --config ../configs/config.yaml 
            --command test_command
            """
      parser = utils.args_parser.build_parser()
      if len(sys.argv) == 1:
        args = parser.parse_args(args=argv_str.split())
      else:
        args = parser.parse_args()
      args.CUDA_VISIBLE_DEVICES = os.environ['CUDA_VISIBLE_DEVICES']
      args = utils.config_utils.DotDict(vars(args))
      return args, argv_str
    args, argv_str = build_args()

    args.outdir = outdir
    args, myargs = utils.config.setup_args_and_myargs(args=args, myargs=myargs)

    prefix = 'test_summary_scalars'
    for step in range(1000):
      summary = {'a': step, 'b': step}
      # Trainer.summary_scalars(summary, prefix, step=step,
      #                         writer=myargs.writer,
      #                         textlogger=myargs.textlogger)

      trainer = Trainer(args=args, myargs=myargs)
      trainer.summary_scalars(summary, prefix, step,
                              log_axe=True, log_axe_sec=100)

    input('End %s' % outdir)
    return

  def test_summary_scalars_together(self):
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
      os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    if 'PORT' not in os.environ:
      os.environ['PORT'] = '6006'
    if 'TIME_STR' not in os.environ:
      os.environ['TIME_STR'] = '0' if utils.is_debugging() else '1'
    # func name
    outdir = os.path.join('results', sys._getframe().f_code.co_name)
    myargs = argparse.Namespace()

    def build_args():
      argv_str = f"""
            --config ../configs/config.yaml 
            --command test_command
            """
      parser = utils.args_parser.build_parser()
      if len(sys.argv) == 1:
        args = parser.parse_args(args=argv_str.split())
      else:
        args = parser.parse_args()
      args.CUDA_VISIBLE_DEVICES = os.environ['CUDA_VISIBLE_DEVICES']
      args = utils.config_utils.DotDict(vars(args))
      return args, argv_str

    args, argv_str = build_args()

    args.outdir = outdir
    args, myargs = utils.config.setup_args_and_myargs(args=args, myargs=myargs)

    prefix = 'test_summary_scalars'
    for step in range(1000):
      summary = {'a': step, 'b': step}
      # Trainer.summary_scalars(summary, prefix, step=step,
      #                         writer=myargs.writer,
      #                         textlogger=myargs.textlogger)

      trainer = Trainer(args=args, myargs=myargs)
      trainer.summary_scalars_together(summary, prefix, step,
                                       log_axe=True, log_axe_sec=100)

    input('End %s' % outdir)
    return

  def test_summary_dict(self):
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
      os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    if 'PORT' not in os.environ:
      os.environ['PORT'] = '6006'
    if 'TIME_STR' not in os.environ:
      os.environ['TIME_STR'] = '0' if utils.is_debugging() else '1'
    # func name
    outdir = os.path.join('results', sys._getframe().f_code.co_name)
    myargs = argparse.Namespace()

    def build_args():
      argv_str = f"""
            --config ../configs/config.yaml 
            --command test_command
            """
      parser = utils.args_parser.build_parser()
      if len(sys.argv) == 1:
        args = parser.parse_args(args=argv_str.split())
      else:
        args = parser.parse_args()
      args.CUDA_VISIBLE_DEVICES = os.environ['CUDA_VISIBLE_DEVICES']
      args = utils.config_utils.DotDict(vars(args))
      return args, argv_str

    args, argv_str = build_args()

    args.outdir = outdir
    args, myargs = utils.config.setup_args_and_myargs(args=args, myargs=myargs)

    prefix = 'test_summary_scalars'
    import collections
    summary_dict = collections.defaultdict(dict)
    for step in range(1000):
      summary = {'a': step, 'b': step + 1}
      # Trainer.summary_scalars(summary, prefix, step=step,
      #                         writer=myargs.writer,
      #                         textlogger=myargs.textlogger)

      summary_dict['dict1'] = summary
      summary_dict['scalars'] = summary
      trainer = Trainer(args=args, myargs=myargs)
      trainer.summary_dicts(summary_dict, prefix, step,
                            log_axe=True, log_axe_sec=10)

    input('End %s' % outdir)
    return