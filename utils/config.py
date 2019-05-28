import os
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
  create_dirs([args.ckptdir, args.tbdir])

  args.logfile = os.path.join(args.outdir, "log.txt")
  args.configfile = os.path.join(args.outdir, "config.yaml")



def process_config(outdir, config_file, resume_root=None, args=None, myargs=None):
  """
  """
  TIME_STR = bool(int(os.getenv('TIME_STR', 0)))
  time_str = time.strftime("%Y%m%d-%H_%M_%S")

  args.outdir = outdir if not TIME_STR else (outdir + '_' + time_str)
  if resume_root:
    args.outdir = resume_root
  else:
    shutil.rmtree(args.outdir, ignore_errors=True)
    os.makedirs(args.outdir, exist_ok=True)
    shutil.copytree('.', os.path.join(args.outdir, 'code'),
                    ignore=shutil_utils.ignoreAbsPath(['results', ]))
    shutil.copytree('../submodule/template_lib',
                    os.path.join(args.outdir, 'submodule/template_lib'),
                    ignore=shutil_utils.ignoreNamePath(['results', 'submodule']))

  # Setup dirs in args
  setup_dirs_and_files(args=args)

  # setup logging in the project
  logger = logging_utils.get_logger(filename=args.logfile)
  myargs.logger = logger
  logger.info("The outdir is {}".format(args.outdir))
  logger.info("The args: ")
  logger.info_msg(pprint.pformat(args))

  # Parse config file
  config = get_config_from_file(config_file, saved_path=args.configfile)
  myargs.config = EasyDict(config)
  logger.info(" THE config of experiment:")
  logger.info_msg(pprint.pformat(config))

  # tensorboard
  tbtool = torch_utils.TensorBoardTool(tbdir=args.tbdir)
  writer = tbtool.run()
  tbtool.add_text_md_args(args=args, name='args')
  tbtool.add_text_str_args(args=config, name='config')
  myargs.writer = writer

  # checkpoint
  checkpoint = torch_utils.CheckpointTool(ckptdir=args.ckptdir)
  myargs.checkpoint = checkpoint
  myargs.checkpoint_dict = collections.OrderedDict()

  args = EasyDict(args)
  return args