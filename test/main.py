import os
import sys
import unittest
import argparse

os.chdir('..')
from template_lib.examples import test_bash
from template_lib import utils


class TestingUnit(unittest.TestCase):

  def test_Case(self):
    """
    Usage:
        export CUDA_VISIBLE_DEVICES=2,3,4,5
        export PORT=6006
        export TIME_STR=1
        export PYTHONPATH=../../submodule:..
        python -c "import main; \
        main.TestingUnit().test_Case()"
    :return:
    """
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
      os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    if 'PORT' not in os.environ:
      os.environ['PORT'] = '6006'
    if 'TIME_STR' not in os.environ:
      os.environ['TIME_STR'] = '0' if utils.is_debugging() else '1'
    # func name
    assert sys._getframe().f_code.co_name.startswith('test_')
    command = sys._getframe().f_code.co_name[5:]
    class_name = self.__class__.__name__[7:] \
      if self.__class__.__name__.startswith('Testing') \
      else self.__class__.__name__
    outdir = f'results/{class_name}/{command}'
    myargs = argparse.Namespace()

    def build_args():
      argv_str = f"""
            --config ../configs/config.yaml 
            --command {command}
            --outdir {outdir}
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

    args, myargs = utils.config.setup_args_and_myargs(args=args, myargs=myargs)

    test_bash.TestingUnit().test_resnet(
      bs=512, gpu=os.environ['CUDA_VISIBLE_DEVICES'])
    return






