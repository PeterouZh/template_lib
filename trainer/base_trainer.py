import os, collections


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
      self.logger.info('Number of params in {}: {}'.format(
        label, sum([p.data.nelement() for p in model.parameters()])
      ))

  def save_checkpoint(self, filename='ckpt.tar'):
    self.myargs.checkpoint.save_checkpoint(
      checkpoint_dict=self.myargs.checkpoint_dict, filename=filename)

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
      resumepath=args.resume_path)
    for key in self.train_dict:
      if key in loaded_state_dict['train_dict']:
        self.train_dict[key] = loaded_state_dict['train_dict'][key]

  def finetune(self):
    raise NotImplemented

  def train(self, ):
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

  def summary_scalars_together(self, summary, prefix, step):
    self.myargs.writer.add_scalars(prefix, summary, step)

  def evaluate(self):
    raise NotImplemented
