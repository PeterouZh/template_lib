import shutil
from easydict import EasyDict
import multiprocessing
import time
import subprocess
import yaml
import os
import sys
import unittest
import argparse

from template_lib import utils
from template_lib.utils import modelarts_utils

class Worker(multiprocessing.Process):
  def run(self):
    command = self._args[0]
    print('+++Execute: %s'%command)
    os.system(command)
    return


def modelarts_record_bash_command(args, myargs, command=None):
  try:
    import moxing as mox
    assert os.environ['DLS_TRAIN_URL']
    log_obs = os.environ['DLS_TRAIN_URL']
    command_file_obs = os.path.join(log_obs, 'commands.txt')
    command_file = os.path.join(args.outdir, 'commands.txt')
    with open(command_file, 'a') as f:
      if not command:
        f.write(args.outdir)
      else:
        f.write(command)
      f.write('\n')
    mox.file.copy(command_file, command_file_obs)

  except ModuleNotFoundError as e:
    myargs.logger.info("Don't use modelarts!")


class TestingUnit(unittest.TestCase):

  def test_bash(self):
    """
    Usage:
        export CUDA_VISIBLE_DEVICES=2,3,4,5
        export PORT=6006
        export TIME_STR=1
        export PYTHONPATH=../submodule:..
        python -c "import test_bash; \
        test_bash.TestingUnit().test_bash()"
    :return:
    """
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
      os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2, 3, 4, 5, 6, 7'
    if 'PORT' not in os.environ:
      os.environ['PORT'] = '6106'
    if 'TIME_STR' not in os.environ:
      os.environ['TIME_STR'] = '0'

    # func name
    outdir = os.path.join('results', sys._getframe().f_code.co_name)
    myargs = argparse.Namespace()

    def build_args():
      argv_str = f"""
            --config ../configs/virtual_terminal.yaml \
            --resume False --resume_path None
            --resume_root None
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

    try:
      # Clean log_obs dir
      import moxing as mox
      assert os.environ['DLS_TRAIN_URL']
      log_obs = os.environ['DLS_TRAIN_URL']
      if mox.file.exists(log_obs):
        mox.file.remove(log_obs, recursive=True)
      mox.file.make_dirs(log_obs)
    except:
      pass
    args.outdir = outdir
    args, myargs = utils.config.setup_args_and_myargs(args=args, myargs=myargs)
    modelarts_record_bash_command(args, myargs)

    old_command = ''
    myargs.logger.info('Begin loop.')
    # Create bash_command.sh
    bash_file = os.path.join(args.outdir, 'bash_command.sh')
    with open(bash_file, 'w') as f:
      pass
    cwd = os.getcwd()
    # copy outdir to outdir_obs
    modelarts_utils.modelarts_sync_results(args, myargs, join=True)
    while True:
      try:
        import moxing as mox
        # copy oudir_obs to outdir
        time.sleep(3)
        mox.file.copy_parallel(args.outdir_obs, args.outdir)
      except:
        pass
      shutil.copy(bash_file, cwd)
      try:
        with open(args.configfile, 'rt') as handle:
          config = yaml.load(handle)
          config = EasyDict(config)
        command = config.command
      except:
        print('Parse config.yaml error!')
        command = None
        old_command = ''
      if command != old_command:
        old_command = command
        if type(command) is list and command[0].startswith('bash'):
          modelarts_record_bash_command(args, myargs, command[0])
          p = Worker(name='Command worker', args=(command[0], ))
          p.start()
        elif type(command) is list:
          command = list(map(str, command))
          # command = ' '.join(command)
          print('===Execute: %s' % command)
          err_f = open(os.path.join(args.outdir, 'err.txt'), 'w')
          try:
            cwd = os.getcwd()
            return_str = subprocess.check_output(
              command, encoding='utf-8', cwd=cwd, shell=True)
            print(return_str, file=err_f, flush=True)
          except subprocess.CalledProcessError as e:
            print("Oops!\n", e.output, "\noccured.",
                  file=err_f, flush=True)
            print(e.returncode, file=err_f, flush=True)
          err_f.close()

          # os.system(command)
        modelarts_utils.modelarts_sync_results(args, myargs, join=True)
      if hasattr(args, 'outdir_obs'):
        log_obs = os.environ['DLS_TRAIN_URL']
        jobs_file_obs = os.path.join(log_obs, 'jobs.txt')
        jobs_file = os.path.join(args.outdir, 'jobs.txt')
        if mox.file.exists(jobs_file_obs):
          mox.file.copy(jobs_file_obs, jobs_file)
        mox.file.copy_parallel(args.outdir, args.outdir_obs)



