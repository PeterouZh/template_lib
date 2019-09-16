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
  if args.resume_root and args.resume:
    # modelarts resume setup
    modelarts_resume(args)
  try:
    import moxing as mox
    myargs.logger.info("Using modelarts!")
    modelarts_record_jobs(args, myargs)

    assert os.environ['RESULTS_OBS']
    args.results_obs = os.environ['RESULTS_OBS']
    myargs.logger.info_msg('results_obs: %s', args.results_obs)
    assert args.outdir.startswith('results/')
    args.outdir_obs = os.path.join(args.results_obs, args.outdir[8:])

    def copy_obs(s, d, copytree=False):
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
  except ModuleNotFoundError as e:
    print("Resume, don't use modelarts!")
  return


def modelarts_finetune(args, finetune_path):
  try:
    import moxing as mox
    args.finetune_path = finetune_path
    assert args.finetune_path.startswith('results/')
    args.finetune_path_obs = os.path.join(
      args.results_obs, args.finetune_path[8:])

    assert mox.file.exists(args.finetune_path_obs)
    print('Copying %s \n to %s'%(args.finetune_path_obs, args.finetune_path))
    mox.file.copy_parallel(args.finetune_path_obs, args.finetune_path)
  except ModuleNotFoundError as e:
    print("Finetune, don't use modelarts!")
  return


def modelarts_sync_results(args, myargs, join=False, end=False):
  if hasattr(args, 'outdir_obs'):
    import moxing as mox
    # sync jobs.txt
    log_obs = os.environ['DLS_TRAIN_URL']
    jobs_file_obs = os.path.join(log_obs, 'jobs.txt')
    jobs_file = os.path.join(args.outdir, 'jobs.txt')
    if mox.file.exists(jobs_file_obs):
      mox.file.copy(jobs_file_obs, jobs_file)

    if end:
      modelarts_record_jobs(args, myargs, end=end)

    print('Copying args.outdir to outdir_obs ...', file=myargs.stdout)
    worker = myargs.copy_obs(args.outdir, args.outdir_obs,
                             copytree=True)
    if join:
      print('Join copy obs processing.', file=myargs.stdout)
      worker.join()
  return


def modelarts_record_jobs(args, myargs, end=False, str_info=''):
  try:
    import moxing as mox
    assert os.environ['DLS_TRAIN_URL']
    log_obs = os.environ['DLS_TRAIN_URL']
    jobs_file_obs = os.path.join(log_obs, 'jobs.txt')
    jobs_file = os.path.join(args.outdir, 'jobs.txt')
    if mox.file.exists(jobs_file_obs):
      mox.file.copy(jobs_file_obs, jobs_file)

    with open(jobs_file, 'a') as f:
      if not end:
        f.write(str_info + '\t' + args.outdir + ' ' + os.environ['PORT'])
      else:
        f.write('End: \t' + args.outdir + ' ' + os.environ['PORT'])
      f.write('\n')
    mox.file.copy(jobs_file, jobs_file_obs)

  except ModuleNotFoundError as e:
    myargs.logger.info("Don't use modelarts!")