import os
import sys
import unittest
import argparse
from easydict import EasyDict

import utils


class TestingUnit(unittest.TestCase):

  def test_Case(self):
    """
    Usage:
        export CUDA_VISIBLE_DEVICES=2,3,4,5
        export PORT=6006
        export TIME_STR=1
        export PYTHONPATH=../submodule:..
        python -c "import main; \
        main.TestingUnit().test_Case()"
    :return:
    """
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
      os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    if 'PORT' not in os.environ:
      os.environ['PORT'] = '6007'
    if 'TIME_STR' not in os.environ:
      os.environ['TIME_STR'] = '0'

    # func name
    outdir = os.path.join('results', sys._getframe().f_code.co_name)
    myargs = argparse.Namespace()

    def build_args():
      argv_str = f"""
            --config ../configs/config.yaml \
            --resume False --resume_path None
            --resume_root None
            """
      parser = utils.args_parser.build_parser()
      if len(sys.argv) == 1:
        args = parser.parse_args(args=argv_str.split())
      else:
        args = parser.parse_args()
      args = utils.config_utils.DotDict(vars(args))
      return args, argv_str
    args, argv_str = build_args()

    # parse the config json file
    args = utils.config.process_config(outdir=outdir, config_file=args.config,
                                       resume_root=args.resume_root, args=args,
                                       myargs=myargs)

    return

    import time, shutil, collections
    od = collections.OrderedDict()
    od['PORT'] = os.environ['PORT']
    od['CUDA_VISIBLE_DEVICES'] = os.environ['CUDA_VISIBLE_DEVICES']
    od['python_file'] = __file__
    od['time_str'] = time.strftime("%Y%m%d-%H_%M_%S")
    od['func_call'] = sys._getframe().f_code.co_name

    od['resume_root'] = ''

    od['outdir'] = ('results/{}' \
      .format(
      od['func_call'] if not TIME_STR else od['func_call'] + '_' + od[
        'time_str'])) \
      if not od['resume_root'] else od['resume_root']
    od['del_old_log'] = True
    if od['del_old_log'] and not od['resume_root']:
      shutil.rmtree(od['outdir'], ignore_errors=True)
    if not os.path.exists(od['outdir']): os.makedirs(od['outdir'])
    od['ckptdir'] = os.path.join(od['outdir'], 'models')
    od['tbdir'] = os.path.join(od['outdir'], "tb")
    od['tb_logdir'] = ','.join(['name0:' + od['tbdir'],
                                ])
    od['logfile'] = os.path.join(od['outdir'], "log.txt")
    od['configfile'] = os.path.join(od['outdir'], "config.yaml")
    od['gpus'] = list(range(len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))))

    from peng_lib.torch_utils import ignorePath
    if not od['resume_root']:
      shutil.copytree('.', os.path.join(od['outdir'], 'code'),
                      ignore=ignorePath(['results', ]))
      shutil.copytree('../submodule/peng_lib',
                      os.path.join(od['outdir'], 'submodule/peng_lib'),
                      ignore=ignorePath(['', ]))

    def build_args():
      argv_str = f"""
          --config points_gan/train_funcs/train_func_ball_autoencoder_gan_config.yaml \
          --resume {bool(od['resume_root'])} \
          --resume_path {os.path.join(od['resume_root'], 'models/ckpt.tar')} \
          --evaluate False \
          --evaluate_path results/test_Train_Ball_AutoEncoder_20190511-21_06_26/models/ckpt.tar.best \
          --finetune False \
          --finetune_path results/test_Train_Ball_AutoEncoder_new_20190513-21_07_16/models/ckpt.tar.best
          """
      from points_gan import arg_parser
      parser = arg_parser.build_parser()

      if len(sys.argv) == 1:
        args = parser.parse_args(args=argv_str.split())
      else:
        args = parser.parse_args()
      if args.evaluate:
        args.config = os.path.join(os.path.dirname(args.evaluate_path),
                                   '../config.yaml')
      return args, argv_str

    args, argv_str = build_args()

    if od['resume_root']:
      print('** Resume from dir: %s' % od['resume_root'])
      time.sleep(3)

    # all args
    args = argparse.Namespace(**vars(args), **od)

    od[
      'argv_str'] = \
      '------------------------------------------------------------'
    argv_str = argv_str.strip().strip('-').split('--')
    for elem in argv_str:
      elem = elem.split()
      if elem[0] == 'D_thin':
        od[elem[0]] = getattr(args, 'D_wide')
        continue
      dest = elem[0].replace('-', '_')
      od[dest] = getattr(args, dest)

    myargs = argparse.Namespace()
    from peng_lib.torch_utils import get_tbwriter_logger_checkpoint
    get_tbwriter_logger_checkpoint(args=args, od=od, myargs=myargs,
                                   **vars(args))

    from points_gan import train_gan
    train_gan.main(args=args, myargs=myargs, **vars(args))
    input("End training: %s" % od['func_call'])
