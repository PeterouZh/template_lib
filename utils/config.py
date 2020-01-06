import argparse
import os, sys
import shutil
import time
import logging
import collections
import json, yaml
from easydict import EasyDict
import pprint
from datetime import datetime

from .dirs import create_dirs
from . import logging_utils
from . import config_utils
from . import torch_utils
from . import shutil_utils
from . import modelarts_utils
from . import tensorboardX_utils
from . import is_debugging, args_parser


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


def setup_dirs_and_files(args, **kwargs):
  # create some important directories to be used for that experiment.
  args.abs_outdir = os.path.realpath(args.outdir)
  args.ckptdir = os.path.join(args.outdir, "models/")
  args.tbdir = os.path.join(args.outdir, "tb/")
  args.textlogdir = os.path.join(args.outdir, 'textlog/')
  args.imgdir = os.path.join(args.outdir, 'saved_imgs/')
  create_dirs([args.ckptdir, args.tbdir, args.textlogdir, args.imgdir])
  args.logfile = os.path.join(args.outdir, "log.txt")
  args.config_command_file = os.path.join(args.outdir, "config_command.yaml")

  # append log dir name in configfile
  if ('add_number_to_configfile' in kwargs) and kwargs['add_number_to_configfile']:
    args.configfile = os.path.join(args.outdir, "c_%s.yaml"%kwargs['log_number'])
  else:
    args.configfile = os.path.join(args.outdir, "config.yaml")
  pass


def setup_outdir(args, resume_root, resume, **kwargs):
  TIME_STR = bool(int(os.getenv('TIME_STR', 0)))
  time_str = datetime.now().strftime("%Y%m%d-%H_%M_%S_%f")[:-3]
  args.outdir = args.outdir if not TIME_STR else (args.outdir + '_' + time_str)
  if 'log_number' in kwargs:
    args.outdir += '_%s'%kwargs['log_number']

  if resume_root and resume:
    args.outdir = resume_root
    print('Using config.yaml in resume_root: %s'%resume_root)
    args.config = os.path.join(args.outdir, "config.yaml")
  else:
    shutil.rmtree(args.outdir, ignore_errors=True)
    os.makedirs(args.outdir, exist_ok=True)
  #   try:
  #     print('Start copying code to outdir.')
  #     shutil.copytree('.', os.path.join(args.outdir, 'code'),
  #                     ignore=shutil_utils.ignoreAbsPath(['results', ]))
  #     shutil.copytree(
  #       '../submodule/template_lib',
  #       os.path.join(args.outdir, 'submodule/template_lib'),
  #       ignore=shutil_utils.ignoreNamePath(['results', 'submodule']))
  #     print('End copying code to outdir.')
  #   except:
  #     print("Error! Copying code to results.")
  return


def setup_logger_and_redirect_stdout(logfile, myargs):
  # sys.stdout is changed
  if isinstance(sys.stdout, logging_utils.StreamToLogger):
    sys.stdout = myargs.stdout
    sys.stderr = myargs.stderr
  # setup logging in the project
  logging_utils.get_logger(
    filename=logfile, logger_names=['template_lib', 'tl'], stream=True)
  logger = logging.getLogger('tl')
  myargs.logger = logger
  myargs.stdout = sys.stdout
  myargs.stderr = sys.stderr
  logging_utils.redirect_print_to_logger(logger=logger)
  return


def setup_config(config_file, saved_config_file, args, myargs):
  # Parse config file
  config = get_config_from_file(config_file, saved_path=saved_config_file)
  myargs.config = config

  config_command = getattr(config, args.command, None)
  if config_command:
    # inherit from base
    config_command = config_inherit_from_base(
      config=config_command, configs=config)
    config_command = convert_easydict_to_dict(config_command)
    saved_config_command = {args.command: config_command}
    config_utils.YamlConfigParser.write_yaml(
      saved_config_command, fname=args.config_command_file)
    # update command config
    myargs.config[args.command] = \
      config_utils.DotDict(config_command)

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
    logger = logging.getLogger(__name__)
    logger.info('command config: \n{}'.format(pprint.pformat(command_config)))
  return


def setup_checkpoint(ckptdir, myargs):
  checkpoint = torch_utils.CheckpointTool(ckptdir=ckptdir)
  myargs.checkpoint = checkpoint
  myargs.checkpoint_dict = collections.OrderedDict()


def get_git_hash():
  cwd = os.getcwd()
  os.chdir(os.path.join(cwd, '..'))
  try:
    import git
    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha
    print('git hash: \n%s'%sha)
    print('git checkout sha', end='')
    print('git submodule update --recursive')
  except:
    sha = 0
    print('Error in get_git_hash')
    import traceback
    print(traceback.format_exc(), flush=True)
  os.chdir(cwd)
  return sha


def setup_args_and_myargs(args, myargs, start_tb=True, **kwargs):
  setup_outdir(args=args, resume_root=args.resume_root, resume=args.resume,
               **kwargs)
  setup_dirs_and_files(args=args, **kwargs)
  setup_logger_and_redirect_stdout(args.logfile, myargs)
  get_git_hash()

  myargs.textlogger = logging_utils.TextLogger(
    log_root=args.textlogdir, reinitialize=(not args.resume),
    logstyle='%10.3f')

  logger = logging.getLogger(__name__)
  logger.info("The outdir: \n\t{}".format(args.abs_outdir))
  logger.info("The args: \n{}".format(pprint.pformat(args)))

  setup_config(
    config_file=args.config, saved_config_file=args.configfile,
    args=args, myargs=myargs)
  setup_tensorboardX(tbdir=args.tbdir, args=args, config=myargs.config,
                     myargs=myargs, start_tb=start_tb)

  modelarts_utils.modelarts_setup(args, myargs)

  setup_checkpoint(ckptdir=args.ckptdir, myargs=myargs)

  args = EasyDict(args)
  myargs.config = EasyDict(myargs.config)
  return args, myargs


def parse_args_and_setup_myargs(argv_str=None, start_tb=False):
  """
  Usage:

  :return:
  """
  if 'TIME_STR' not in os.environ:
    os.environ['TIME_STR'] = '0' if is_debugging() else '1'
  if not isinstance(argv_str, list):
    argv_str = argv_str.split()
  # parse args
  parser = args_parser.build_parser()
  unparsed_argv = []
  if argv_str:
    args, unparsed_argv = parser.parse_known_args(args=argv_str)
  else:
    args = parser.parse_args()
  args = config_utils.DotDict(vars(args))

  # setup args and myargs
  myargs = argparse.Namespace()
  args, myargs = setup_args_and_myargs(
    args=args, myargs=myargs, start_tb=start_tb)
  myargs.logger.info('\npython \t\\\n  ' + ' \\\n  '.join(argv_str))
  return args, myargs, unparsed_argv


def config2args(config, args):
  logger = logging.getLogger(__name__)
  for k, v in config.items():
    if not k in args:
      logger.info('\n\t%s = %s \tnot in args' % (k, v))
    setattr(args, k, v)
  return args


def update_config(super_config, config):
  """

  :param super_config:
  :param config:
  :return:
  """
  for k in config:
    if isinstance(config[k], dict) and hasattr(super_config, k):
      update_config(super_config[k], config[k])
    else:
      setattr(super_config, k, config[k])
  return super_config


def convert_easydict_to_dict(config):
  config = dict(config)
  for k in config:
    if isinstance(config[k], EasyDict):
      config[k] = dict(config[k])
      config.update({k: convert_easydict_to_dict(config[k])})
  return config


def config_inherit_from_base(config, configs, arg_base=[]):
  base = getattr(config, 'base', [])
  if not isinstance(arg_base, list):
    arg_base = [arg_base]
  base += arg_base
  if not base:
    return EasyDict(config)

  super_config = EasyDict()
  for b in base:
    b_config = getattr(configs, b, {})
    b_config = config_inherit_from_base(b_config, configs)
    super_config = update_config(super_config, b_config)
  super_config = update_config(super_config, config)
  return super_config
