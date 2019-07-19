import os, time
import multiprocessing


class CopyObsProcessing(multiprocessing.Process):
  """
    worker = CopyObsProcessing(args=(s, d, copytree))
    worker.start()
    worker.join()
  """
  def run(self):
    try:
      import moxing as mox
      s, d, copytree = self._args
      print('Starting %s, Copying %s \nto %s.' % (self.name, s, d))
      start_time = time.time()
      if copytree:
        mox.file.copy_parallel(s, d)
      else:
        mox.file.copy(s, d)
      elapsed_time = time.time() - start_time
      time_str = time.strftime('%H:%M:%S', time.gmtime(elapsed_time))
      print('End %s, elapsed time: %s'%(self.name, time_str))
    except:
      print("Don't use modelarts!")
    return


def modelarts_setup(args, myargs):
  try:
    import moxing as mox
    myargs.logger.info("Using modelarts!")
    assert os.environ['DLS_TRAIN_URL']
    # temporary obs dir for tensorboard files during training
    args.tb_obs = os.environ['DLS_TRAIN_URL']
    if mox.file.exists(args.tb_obs):
      mox.file.remove(args.tb_obs, recursive=True)
    mox.file.make_dirs(args.tb_obs)
    myargs.logger.info_msg('tb_obs: %s', args.tb_obs)
    assert os.environ['RESULTS_OBS']
    args.results_obs = os.environ['RESULTS_OBS']
    myargs.logger.info_msg('results_obs: %s', args.results_obs)

    def copy_obs(s, d, copytree=False):
      mox.file.copy(args.logfile, os.path.join(args.tb_obs, 'log.txt'))
      worker = CopyObsProcessing(args=(s, d, copytree))
      worker.start()
      return worker
    myargs.copy_obs = copy_obs
  except ModuleNotFoundError as e:
    myargs.logger.info("Don't use modelarts!")
  return


def modelarts_resume(args):
  try:
    import moxing as mox
    assert os.environ['RESULTS_OBS']
    args.results_obs = os.environ['RESULTS_OBS']

    exp_name = os.path.relpath(
      os.path.normpath(args.resume_root), './results')
    resume_root_obs = os.path.join(args.results_obs, exp_name)
    assert mox.file.exists(resume_root_obs)
    print('Copying %s \n to %s'%(resume_root_obs, args.resume_root))
    mox.file.copy_parallel(resume_root_obs, args.resume_root)
  except:
    print("Resume, don't use modelarts!")
  return
