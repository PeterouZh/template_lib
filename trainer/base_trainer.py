import os, collections
import sys

from ..utils import modelarts_utils

class Trainer(object):
  def __init__(self, args, myargs):
    self.args = args
    self.myargs = myargs
    self.config = myargs.config
    self.logger = myargs.logger
    self.train_dict = self.init_train_dict()

    # self.dataset_load()
    self.model_create()
    self.optimizer_create()
    self.schedule_create()
    pass

  def init_train_dict(self, ):
    train_dict = collections.OrderedDict()
    train_dict['epoch_done'] = 0
    train_dict['batches_done'] = 0
    self.myargs.checkpoint_dict['train_dict'] = train_dict
    return train_dict

  def dataset_load(self):
    raise NotImplemented

  def model_create(self):
    raise NotImplemented

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
    raise NotImplemented

  def scheduler_create(self):
    pass

  def scheduler_step(self, epoch):
    pass

  def resume(self):
    args = self.args
    myargs = self.myargs
    self.logger.info('=> Resume from: %s', args.resume_path)
    loaded_state_dict = myargs.checkpoint.load_checkpoint(
      checkpoint_dict=myargs.checkpoint_dict,
      filename=args.resume_path)
    for key in self.train_dict:
      if key in loaded_state_dict['train_dict']:
        self.train_dict[key] = loaded_state_dict['train_dict'][key]

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
      self.modelarts(join=True)
      import traceback
      print(traceback.format_exc())

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

  def summary_scalars(self, summary, prefix, step):
    myargs = self.myargs
    for key in summary:
      myargs.writer.add_scalar(prefix + '/%s' % key, summary[key], step)
    myargs.textlogger.log(step, **summary)

  def summary_scalars_together(self, summary, prefix, step):
    self.myargs.writer.add_scalars(prefix, summary, step)
    self.myargs.textlogger.log(step, **summary)

  def summary_dicts(self, summary_dicts, prefix, step):
    prefix_split = prefix.split('_')
    if len(prefix_split) == 1:
      prefix_abb = prefix
    else:
      prefix_abb = ''.join([k[0] for k in prefix_split])
    for summary_n, summary_v in summary_dicts.items():
      summary_v = {prefix_abb + '.' + k: v for k, v in summary_v.items()}
      if summary_n == 'scalars':
        self.summary_scalars(
          summary_v, prefix=prefix + '/' + summary_n, step=step)
      else:
        self.summary_scalars_together(
          summary_v, prefix=prefix + '/' + summary_n, step=step)

  def summary_figures(self, summary_dicts, prefix):
    prefix_split = prefix.split('_')
    if len(prefix_split) == 1:
      prefix_abb = prefix
    else:
      prefix_abb = ''.join([k[0] for k in prefix_split])
    for summary_n, summary_v in summary_dicts.items():
      summary_v = {prefix_abb + '.' + k: v for k, v in summary_v.items()}
      if summary_n == 'scalars':
        self.myargs.textlogger.log_axes(**summary_v)
      else:
        self.myargs.textlogger.log_ax(**summary_v)

  def evaluate(self):
    raise NotImplemented

  def modelarts(self, join=False, end=False):
    modelarts_utils.modelarts_sync_results(self.args, self.myargs,
                                           join=join, end=end)

