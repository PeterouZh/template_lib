import os
import subprocess
import sys
import unittest
import argparse

from template_lib.utils.config import parse_args_and_setup_myargs, config2args
from template_lib.examples import test_bash
from template_lib import utils
from template_lib.v2.config import get_command_and_outdir, setup_outdir_and_yaml, get_append_cmd_str, \
  start_cmd_run


class Testing_stylegan2(unittest.TestCase):

  def test_train_ffhq_128(self):
    """
    Usage:

        export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
        export PORT=6006
        export TIME_STR=1
        export PYTHONPATH=./exp:./stylegan2-pytorch:./
        python 	-c "from exp.tests.test_styleganv2 import Testing_stylegan2;\
          Testing_stylegan2().test_train_ffhq_128()"

    :return:
    """
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
      os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    if 'PORT' not in os.environ:
      os.environ['PORT'] = '6006'
    if 'TIME_STR' not in os.environ:
      os.environ['TIME_STR'] = '0' if utils.is_debugging() else '0'

    command, outdir = get_command_and_outdir(self, func_name=sys._getframe().f_code.co_name)
    argv_str = f"""
                --tl_config_file exp/configs/styleganv2.yaml
                --tl_command {command}
                --tl_outdir {outdir}
                """
    args = setup_outdir_and_yaml(argv_str)

    nproc_per_node = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
    cmd_str = f"""
        python -m torch.distributed.launch --nproc_per_node={nproc_per_node} --master_port=8888 
          exp/scripts/train.py 
        """
    cmd_str += get_append_cmd_str(args)
    start_cmd_run(cmd_str)
    pass


# from template_lib.v2.config import update_parser_defaults_from_yaml
# update_parser_defaults_from_yaml(parser)