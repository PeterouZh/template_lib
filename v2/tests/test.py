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

  def test_ddp(self):
    """
    Usage:

        export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
        export TIME_STR=1
        export PYTHONPATH=./exp:./stylegan2-pytorch:./
        python 	-c "from exp.tests.test_styleganv2 import Testing_stylegan2;\
          Testing_stylegan2().test_train_ffhq_128()"

    :return:
    """
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
      os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    if 'TIME_STR' not in os.environ:
      os.environ['TIME_STR'] = '0' if utils.is_debugging() else '0'

    command, outdir = get_command_and_outdir(self, func_name=sys._getframe().f_code.co_name, file=__file__)
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
          {get_append_cmd_str(args)}
        """
    start_cmd_run(cmd_str)
    pass

  def test_detectron2(self):
    """
    Usage:
        export ANSI_COLORS_DISABLED=1

        export CUDA_VISIBLE_DEVICES=2
        export TIME_STR=1
        export PYTHONPATH=.:./exp
        python -c "from exp.tests.test_nas_cgan import TestingPrepareData;\
          TestingPrepareData().test_calculate_fid_stat_CIFAR10()"

    :return:
    """
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
      os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    if 'TIME_STR' not in os.environ:
      os.environ['TIME_STR'] = '0' if utils.is_debugging() else '0'

    command, outdir = get_command_and_outdir(self, func_name=sys._getframe().f_code.co_name, file=__file__)
    argv_str = f"""
                    --tl_config_file exp/nas_cgan/config/prepare_data.yaml
                    --tl_command {command}
                    --tl_outdir {outdir}
                    """
    args = setup_outdir_and_yaml(argv_str)

    num_gpus = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
    cmd_str = f"""
              python exp/scripts/train_net.py 
               {get_append_cmd_str(args)}
               --num-gpus {num_gpus}
              """
    start_cmd_run(cmd_str)
    pass

  def test_plot_lines_figure(self):
    """
    Usage:
        export LD_LIBRARY_PATH=~/anaconda3/envs/py36/lib/
        export TIME_STR=1
        export PYTHONPATH=./exp:./BigGAN_PyTorch_1_lib:./
        python -c "from exp.tests.test_BigGAN import TestingCIFAR10_BigGAN_v1;\
          TestingCIFAR10_BigGAN_v1().test_save_FID_cbn_index_012_figure()"

    :return:
    """
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
      os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    if 'TIME_STR' not in os.environ:
      os.environ['TIME_STR'] = '0' if utils.is_debugging() else '0'

    command, outdir = get_command_and_outdir(self, func_name=sys._getframe().f_code.co_name, file=__file__)
    argv_str = f"""
                    --tl_config_file exp/configs/BigGAN_v1.yaml
                    --tl_command {command}
                    --tl_outdir {outdir}
                    """
    args, cfg = setup_outdir_and_yaml(argv_str, return_cfg=True)

    import matplotlib.pyplot as plt
    import numpy as np
    from template_lib.utils import colors_dict

    fig, ax = plt.subplots()

    ax.set_xticks(range(0, 600, 100))
    ax.tick_params(labelsize=cfg.fontsize.tick_fs)
    ax.set_xlabel(cfg.xlabel, fontsize=cfg.fontsize.xylabel_fs)
    ax.set_ylabel(cfg.ylabel, fontsize=cfg.fontsize.xylabel_fs)

    colors = list(colors_dict.values())
    # colors = [plt.cm.cool(i / float(num_plot - 1)) for i in range(num_plot)]

    ax.set(**cfg.properties)
    for idx, (_, data_dict) in enumerate(cfg.lines.items()):
      log_file = os.path.join(data_dict.result_dir, data_dict.sub_path)
      data = np.loadtxt(log_file, delimiter=':')

      if 'xlim' in cfg.properties:
        data_xlim = cfg.properties.xlim[-1]
        data = data[data[:, 0] <= data_xlim]

      if cfg.get_min_value:
        best_index = data[:, 1].argmin()
      else:
        best_index = data[:, 1].argmax()
      best_x = int(data[:, 0][best_index])
      best_y = data[:, 1][best_index]

      if cfg.add_auxi_label:
        data_dict.properties.label = f'x_{best_x}-y_{best_y:.3f}-' + getattr(data_dict.properties, 'label', '')
      ax.plot(data[:, 0], data[:, 1], color=colors[idx], **data_dict.properties)
      pass

    ax.legend(prop={'size': cfg.fontsize.legend_size})
    fig.show()
    saved_file = os.path.join(args.tl_outdir, cfg.saved_file)
    fig.savefig(saved_file, bbox_inches='tight', pad_inches=0.01)
    print(f'Save to {saved_file}')
    pass

# from template_lib.v2.config import update_parser_defaults_from_yaml
# update_parser_defaults_from_yaml(parser)