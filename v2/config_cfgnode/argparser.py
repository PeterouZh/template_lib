import sys
import argparse
import logging
import os
from easydict import EasyDict

from ..logger import get_logger, set_global_textlogger, TextLogger
from .config import TLCfgNode, set_global_cfg, global_cfg
from ..config.argparser import (get_command_and_outdir, _setup_outdir, get_dict_str, get_git_hash,
                                get_append_cmd_str, start_cmd_run, parser_set_defaults)


def build_parser(parser=None):
  if not parser:
    parser = argparse.ArgumentParser()
  parser.add_argument('--tl_config_file', type=str, default='')
  parser.add_argument('--tl_command', type=str, default='')
  parser.add_argument('--tl_outdir', type=str, default='results/temp')
  parser.add_argument('--tl_opts', type=str, nargs='*', default=[])
  parser.add_argument('--tl_resume', action='store_true', default=False)
  parser.add_argument('--tl_resumedir', type=str, default='results/temp')
  parser.add_argument('--tl_debug', action='store_true', default=False)

  parser.add_argument('--tl_time_str', type=str, default='')
  return parser


def setup_config(config_file, args):
  """
  Load yaml and save command_cfg
  """
  cfg = TLCfgNode(new_allowed=True)
  cfg.merge_from_file(config_file)
  cfg.dump_to_file(args.tl_saved_config_file)

  command_cfg = TLCfgNode.load_yaml_with_command(config_file, command=args.tl_command)
  saved_command_cfg = TLCfgNode(new_allowed=True)
  setattr(saved_command_cfg, args.tl_command, command_cfg)
  saved_command_cfg.dump_to_file(args.tl_saved_config_command_file)

  return cfg, command_cfg


def setup_outdir_and_yaml(argv_str=None, return_cfg=False):
  """
  Usage:

  :return:
  """
  argv_str = argv_str.split()
  parser = build_parser()
  args, unparsed_argv = parser.parse_known_args(args=argv_str)

  args = EasyDict(vars(args))
  _setup_outdir(args=args, resume=args.tl_resume)

  # get logger
  logger = get_logger(filename=args.tl_logfile, logger_names=['template_lib', 'tl'], stream=True)
  args_str = get_dict_str(args)
  logger.info('\nargs:\n' + args_str)

  # git
  get_git_hash(logger)

  if args.tl_command.lower() == 'none':
    return args

  # Load yaml
  cfg, command_cfg = setup_config(config_file=args.tl_config_file, args=args)
  logger.info(f"\nThe cfg: \n{get_dict_str(command_cfg)}")
  if return_cfg:
    global_cfg.merge_from_dict(command_cfg)
    return args, command_cfg
  else:
    return args


def setup_logger_global_cfg_global_textlogger(args, tl_textdir, is_main_process=True):
  # log files
  tl_logfile = os.path.join(args.tl_outdir, "log.txt")
  if is_main_process:
    if len(logging.getLogger('tl').handlers) < 2:
      logger = get_logger(filename=tl_logfile)

  # textlogger
  if is_main_process:
    textlogger = TextLogger(log_root=tl_textdir)
    set_global_textlogger(textlogger=textlogger)

  # Load yaml file and update parser defaults
  if not args.tl_command.lower() == 'none':
    cfg = TLCfgNode.load_yaml_with_command(args.tl_config_file, args.tl_command)
    cfg.merge_from_list(args.tl_opts)
    set_global_cfg(cfg)
    logging.getLogger('tl').info("\nglobal_cfg: \n" + get_dict_str(global_cfg))
    saved_command_cfg = TLCfgNode(new_allowed=True)
    setattr(saved_command_cfg, args.tl_command, cfg)
    global_cfg.tl_saved_config_file = f"{args.tl_outdir}/config_command.yaml"
    saved_command_cfg.dump_to_file(global_cfg.tl_saved_config_file)
  else:
    cfg = {}
  return cfg, tl_logfile

def update_parser_defaults_from_yaml(parser, name='args', use_cfg_as_args=False, is_main_process=True):
  parser = build_parser(parser)

  args, _ = parser.parse_known_args()
  tl_ckptdir = f'{args.tl_outdir}/ckptdir'
  tl_imgdir = f'{args.tl_outdir}/imgdir'
  tl_textdir = f'{args.tl_outdir}/textdir'

  os.makedirs(args.tl_outdir, exist_ok=True)
  os.makedirs(tl_ckptdir, exist_ok=True)
  os.makedirs(tl_imgdir, exist_ok=True)
  os.makedirs(tl_textdir, exist_ok=True)

  cfg, tl_logfile = setup_logger_global_cfg_global_textlogger(args, tl_textdir, is_main_process=is_main_process)

  if use_cfg_as_args:
    default_args = cfg
  else:
    default_args = cfg[name] if name in cfg else None

  parser_set_defaults(parser, cfg=default_args,
                      tl_imgdir=tl_imgdir, tl_ckptdir=tl_ckptdir, tl_textdir=tl_textdir,
                      tl_logfile=tl_logfile)
  logging.getLogger('tl').info('sys.argv: \n python \n' + ' \ \n'.join(sys.argv))
  args, _ = parser.parse_known_args()
  for k, v in vars(args).items():
    if k.startswith('tl_'):
      global_cfg.merge_from_dict({k: v})
  return parser
