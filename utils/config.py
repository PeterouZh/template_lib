import os, sys
import shutil
import time
import logging
import collections
import json, yaml
from easydict import EasyDict
import pprint

from .dirs import create_dirs
from . import logging_utils
from . import config_utils
from . import torch_utils
from . import shutil_utils
from . import modelarts_utils
from . import tensorboardX_utils


def get_config_from_file(config_file, saved_path):

  try:
    if config_file.endswith('.json'):
      with open(config_file, 'r') as f:
        config_dict = json.load(f)
        config = EasyDict(config_dict)
    elif config_file.endswith('.yaml'):
      config_parser = config_utils.YamlConfigParser(fname=config_file,
                                                    saved_fname=saved_path)
      config = config_parser.config_dotdict

    return config
  except ValueError:
    print("INVALID JSON file format.. Please provide a good json file")
    exit(-1)


def setup_dirs_and_files(args):
  # create some important directories to be used for that experiment.
  args.ckptdir = os.path.join(args.outdir, "models/")
  args.tbdir = os.path.join(args.outdir, "tb/")
  args.textlogdir = os.path.join(args.outdir, 'textlog/')
  args.imgdir = os.path.join(args.outdir, 'saved_imgs/')
  create_dirs([args.ckptdir, args.tbdir, args.textlogdir, args.imgdir])

  args.logfile = os.path.join(args.outdir, "log.txt")
  args.configfile = os.path.join(args.outdir, "config.yaml")


def setup_outdir(args, resume_root, resume):
  TIME_STR = bool(int(os.getenv('TIME_STR', 0)))
  time_str = time.strftime("%Y%m%d-%H_%M_%S")
  args.outdir = args.outdir if not TIME_STR else (args.outdir + '_' + time_str)

  if resume_root and resume:
    args.outdir = resume_root
    print('Using config.yaml in resume_root: %s'%resume_root)
    args.config = os.path.join(args.outdir, "config.yaml")
  else:
    shutil.rmtree(args.outdir, ignore_errors=True)
    os.makedirs(args.outdir, exist_ok=True)
    try:
      print('Start copying code to outdir.')
      shutil.copytree('.', os.path.join(args.outdir, 'code'),
                      ignore=shutil_utils.ignoreAbsPath(['results', ]))
      shutil.copytree(
        '../submodule/template_lib',
        os.path.join(args.outdir, 'submodule/template_lib'),
        ignore=shutil_utils.ignoreNamePath(['results', 'submodule']))
      print('End copying code to outdir.')
    except:
      print("Error! Copying code to results.")

  return


def setup_logger_and_redirect_stdout(logfile, myargs):
  # setup logging in the project
  logger = logging_utils.get_logger(filename=logfile)
  myargs.logger = logger
  myargs.stdout = sys.stdout
  myargs.stderr = sys.stderr
  logging_utils.redirect_print_to_logger(logger=logger)
  return


def setup_config(config_file, saved_config_file, myargs):
  # Parse config file
  config = get_config_from_file(config_file, saved_path=saved_config_file)
  myargs.config = config
  # print(" THE config of experiment:")
  # print(pprint.pformat(config))
  return


def setup_tensorboardX(tbdir, args, config, myargs, start_tb=True):
  # tensorboard
  tbtool = tensorboardX_utils.TensorBoardTool(tbdir=tbdir)
  writer = tbtool.writer
  myargs.writer = writer
  if start_tb:
    tbtool.run()
  tbtool.add_text_md_args(args=args, name='args')
  tbtool.add_text_str_args(args=config, name='config')
  if hasattr(args, 'command'):
    command_config = getattr(config, args.command, 'None')
    tbtool.add_text_str_args(args=command_config, name='command')
    print(pprint.pformat(command_config))
  return


def setup_checkpoint(ckptdir, myargs):
  checkpoint = torch_utils.CheckpointTool(ckptdir=ckptdir)
  myargs.checkpoint = checkpoint
  myargs.checkpoint_dict = collections.OrderedDict()


def setup_args_and_myargs(args, myargs, start_tb=True):
  setup_outdir(args=args, resume_root=args.resume_root, resume=args.resume)
  setup_dirs_and_files(args=args)
  setup_logger_and_redirect_stdout(args.logfile, myargs)
  myargs.textlogger = logging_utils.TextLogger(
    log_root=args.textlogdir, reinitialize=(not args.resume),
    logstyle='%10.3f')

  print("The outdir is {}".format(args.outdir))
  print("The args: ")
  print(pprint.pformat(args))

  setup_config(config_file=args.config, saved_config_file=args.configfile,
               myargs=myargs)
  setup_tensorboardX(tbdir=args.tbdir, args=args, config=myargs.config,
                     myargs=myargs, start_tb=start_tb)

  modelarts_utils.modelarts_setup(args, myargs)

  setup_checkpoint(ckptdir=args.ckptdir, myargs=myargs)

  args = EasyDict(args)
  myargs.config = EasyDict(myargs.config)
  return args, myargs