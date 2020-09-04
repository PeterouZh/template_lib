import json
import logging
import pprint
import subprocess
from datetime import datetime
import argparse
import os
import shutil
import sys

import yaml
from easydict import EasyDict

from .config import setup_config, set_global_cfg

from template_lib.v2.logger import get_logger, set_global_textlogger, TextLogger


def get_command_and_outdir(self, func_name=sys._getframe().f_code.co_name, file=__file__):
  # func name
  assert func_name.startswith('test_')
  command = func_name[5:]
  class_name = self.__class__.__name__
  subdir = class_name[7:] if self.__class__.__name__.startswith('Testing') else class_name
  subdir = subdir.strip('_')
  outdir = f'results/{subdir}/{command}'

  file = os.path.relpath(file, os.path.curdir)
  file = file.replace('/', '.')
  run_str = f"""
             python -c "from {file[:-3]} import {class_name};\\\n  {class_name}().{func_name}()"
             """
  print(run_str.strip())
  return command, outdir


def build_parser(parser=None):
  if not parser:
    parser = argparse.ArgumentParser()
  parser.add_argument('--tl_config_file', type=str, default='')
  parser.add_argument('--tl_command', type=str, default='')
  parser.add_argument('--tl_outdir', type=str, default='results/temp')

  parser.add_argument('--tl_time_str', type=str, default='')
  return parser


def get_append_cmd_str(args):
  cmd_str_append = f"""
            --tl_config_file {args.tl_saved_config_command_file}
            --tl_command {args.tl_command}
            --tl_outdir {args.tl_outdir}
            --tl_time_str {args.tl_time_str}
            """
  return cmd_str_append


def update_parser_defaults_from_yaml(parser, name='args'):
  parser = build_parser(parser)

  args = parser.parse_args()
  tl_ckptdir = f'{args.tl_outdir}/ckptdir'
  tl_imgdir = f'{args.tl_outdir}/imgdir'
  tl_textdir = f'{args.tl_outdir}/textdir'

  os.makedirs(args.tl_outdir, exist_ok=True)
  os.makedirs(tl_ckptdir, exist_ok=True)
  os.makedirs(tl_imgdir, exist_ok=True)
  os.makedirs(tl_textdir, exist_ok=True)
  # log files
  tl_logfile = os.path.join(args.tl_outdir, "log.txt")
  local_rank = getattr(args, 'local_rank', 0)
  if local_rank == 0:
    logger = get_logger(filename=tl_logfile)

  # textlogger
  if local_rank == 0:
    textlogger = TextLogger(log_root=tl_textdir)
    set_global_textlogger(textlogger=textlogger)

  # Load yaml file and update parser defaults
  with open(args.tl_config_file, 'rt') as f:
    cfg = yaml.load(f)[args.tl_command]
  set_global_cfg(cfg)

  parser_set_defaults(parser, cfg=getattr(cfg, name, None),
                      tl_imgdir=tl_imgdir, tl_ckptdir=tl_ckptdir, tl_textdir=tl_textdir,
                      tl_logfile=tl_logfile)
  return parser


def _setup_outdir(args):
  TIME_STR = bool(int(os.getenv('TIME_STR', 0)))
  args.tl_time_str = datetime.now().strftime("%Y%m%d-%H_%M_%S_%f")[:-3]
  args.tl_outdir = args.tl_outdir if not TIME_STR else (args.tl_outdir + '_' + args.tl_time_str)

  shutil.rmtree(args.tl_outdir, ignore_errors=True)
  os.makedirs(args.tl_outdir, exist_ok=True)

  # dirs
  args.tl_abs_outdir = os.path.realpath(args.tl_outdir)

  # files
  args.tl_logfile = os.path.join(args.tl_outdir, "log.txt")
  args.tl_saved_config_file = os.path.join(args.tl_outdir, "config.yaml")
  args.tl_saved_config_command_file = os.path.join(args.tl_outdir, "config_command.yaml")
  pass


def parser_set_defaults(parser, cfg, **kwargs):
  if cfg:
    for k, v in cfg.items():
      parser.set_defaults(**{k: v})
  for k, v in kwargs.items():
    parser.set_defaults(**{k: v})
  return parser


def start_cmd_run(cmd_str):
  cmd = cmd_str.split()
  logger = logging.getLogger('tl')
  logger.info('\n' + ' \\\n  '.join(cmd))
  cmd[0] = sys.executable
  current_env = os.environ.copy()
  process = subprocess.Popen(cmd, env=current_env)

  process.wait()
  if process.returncode != 0:
    raise subprocess.CalledProcessError(returncode=process.returncode, cmd=cmd)
  pass


def get_dict_str(dict_obj):
  message = ''
  message += '----------------- start ---------------\n'
  message += json.dumps(dict_obj, indent=2)
  message += '----------------- End -------------------'
  return message


def setup_outdir_and_yaml(argv_str=None):
  """
  Usage:

  :return:
  """
  argv_str = argv_str.split()
  parser = build_parser()
  args, unparsed_argv = parser.parse_known_args(args=argv_str)

  args = EasyDict(vars(args))
  _setup_outdir(args=args)

  # get logger
  logger = get_logger(filename=args.tl_logfile, logger_names=['template_lib', 'tl'], stream=True)
  args_str = get_dict_str(args)
  logger.info(f"The args: \n{args_str}")

  # Load yaml
  config, config_command = setup_config(
    config_file=args.tl_config_file, saved_config_file=args.tl_saved_config_file, args=args)
  cfg_str = get_dict_str(config_command)
  logger.info(f"The cfg: \n{cfg_str}")
  return args















