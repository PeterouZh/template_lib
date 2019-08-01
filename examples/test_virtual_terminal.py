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
    print('Execute: %s'%command)
    os.system(command)
    return


def modelarts_record_bash_command(args, myargs, command=None):
  try:
    import moxing as mox
    assert os.environ['DLS_TRAIN_URL']
    log_obs = os.environ['DLS_TRAIN_URL']
    if mox.file.exists(log_obs):
      mox.file.remove(log_obs, recursive=True)
    mox.file.make_dirs(log_obs)
    command_file = os.path.join(args.outdir, 'commands.txt')
    with open(command_file, 'a') as f:
      if not command:
        f.write(args.outdir)
      else:
        f.write(command)
      f.write('\n')
    mox.file.copy(command_file, os.path.join(log_obs, 'commands.txt'))

  except ModuleNotFoundError as e:
    myargs.logger.info("Don't use modelarts!")


class TestingUnit(unittest.TestCase):

  def test_virtual_terminal(self):
    """
    Usage:
        export CUDA_VISIBLE_DEVICES=2,3,4,5
        export PORT=6006
        export TIME_STR=1
        export PYTHONPATH=../submodule:..
        python -c "import test_virtual_terminal; \
        test_virtual_terminal.TestingUnit().test_virtual_terminal()"
    :return:
    """
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
      os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2, 3, 4, 5, 6, 7'
    if 'PORT' not in os.environ:
      os.environ['PORT'] = '6006'
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

    # parse the config json file
    args = utils.config.process_config(outdir=outdir, config_file=args.config,
                                       resume_root=args.resume_root, args=args,
                                       myargs=myargs)

    old_command = ''
    myargs.logger.info('Begin loop.')
    modelarts_record_bash_command(args, myargs)
    modelarts_utils.modelarts_sync_results(args, myargs, join=True)
    while True:
      try:
        import moxing as mox
        mox.file.copy_parallel(args.outdir_obs, args.outdir)
      except:
        pass
      try:
        with open(args.configfile, 'rt') as handle:
          config = yaml.load(handle)
          config = EasyDict(config)
        command = config.command
      except:
        print('Parse config.yaml error!')
        command = None
      if command != old_command and command:
        old_command = command
        if type(command) is str and command.startswith('bash'):
          modelarts_record_bash_command(args, myargs, command)
          p = Worker(name='Command worker', args=(command, ))
          p.start()
        elif type(command) is list:
          command = list(map(str, command))
          # command = ' '.join(command)
          print('Execute: %s' % command)
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
      time.sleep(1)


