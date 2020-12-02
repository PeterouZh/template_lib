import tqdm
import shutil
import os
import subprocess
import sys
import unittest
import argparse

from template_lib.examples import test_bash
from template_lib import utils
from template_lib.nni import update_nni_config_file


class Testing_PrepareImageNet(unittest.TestCase):

  def test_mv_val_dataset(self):
    """
    Usage:

        export CUDA_VISIBLE_DEVICES=0
        export TIME_STR=0
        export PYTHONPATH=./
        python -c "from template_lib.proj.imagenet.tests.test_imagenet import Testing_PrepareImageNet;\
          Testing_PrepareImageNet().test_mv_val_dataset()"

    :return:
    """
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
      os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    if 'TIME_STR' not in os.environ:
      os.environ['TIME_STR'] = '0' if utils.is_debugging() else '0'
    from template_lib.v2.config_cfgnode.argparser import \
      (get_command_and_outdir, setup_outdir_and_yaml, get_append_cmd_str, start_cmd_run)

    command, outdir = get_command_and_outdir(self, func_name=sys._getframe().f_code.co_name, file=__file__)
    argv_str = f"""
                --tl_config_file none
                --tl_command none
                --tl_outdir {outdir}
                """
    args = setup_outdir_and_yaml(argv_str)

    n_gpus = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
    cmd_str = f"""
        bash 
        ls -la datasets/ImageNet/ &&
        cp template_lib/proj/imagenet/imagenet_move_val_images.sh datasets/ImageNet &&
        cd datasets/ImageNet/val &&
        ls -la ../imagenet_move_val_images.sh &&
        bash ../imagenet_move_val_images.sh    
        """

    start_cmd_run(cmd_str)
    # from template_lib.v2.config_cfgnode import update_parser_defaults_from_yaml, global_cfg
    # from template_lib.modelarts import modelarts_utils
    # update_parser_defaults_from_yaml(parser)

    # modelarts_utils.setup_tl_outdir_obs(global_cfg)
    # modelarts_utils.modelarts_sync_results_dir(global_cfg, join=True)
    # modelarts_utils.prepare_dataset(global_cfg.get('modelarts_download', {}), global_cfg=global_cfg)
    #
    # modelarts_utils.prepare_dataset(global_cfg.get('modelarts_upload', {}), global_cfg=global_cfg, download=False)
    # modelarts_utils.modelarts_sync_results_dir(global_cfg, join=True)
    pass

  def test_extract_ImageNet_1000x50(self):
    """
    Usage:
        proj_root=moco-exp
        python template_lib/modelarts/scripts/copy_tool.py \
          -s s3://bucket-7001/ZhouPeng/codes/$proj_root -d /cache/$proj_root -t copytree
        cd /cache/$proj_root

        export CUDA_VISIBLE_DEVICES=0
        export TIME_STR=0
        export PYTHONPATH=./
        python -c "from template_lib.proj.imagenet.tests.test_imagenet import Testing_PrepareImageNet;\
          Testing_PrepareImageNet().test_extract_ImageNet_1000x50()"

    :return:
    """
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
      os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    if 'TIME_STR' not in os.environ:
      os.environ['TIME_STR'] = '0' if utils.is_debugging() else '0'
    from template_lib.v2.config_cfgnode.argparser import \
      (get_command_and_outdir, setup_outdir_and_yaml, get_append_cmd_str, start_cmd_run)
    from template_lib.v2.config_cfgnode import update_parser_defaults_from_yaml, global_cfg
    from template_lib.modelarts import modelarts_utils

    command, outdir = get_command_and_outdir(self, func_name=sys._getframe().f_code.co_name, file=__file__)
    argv_str = f"""
                --tl_config_file template_lib/proj/imagenet/tests/configs/PrepareImageNet.yaml
                --tl_command {command}
                --tl_outdir {outdir}
                """
    args, cfg = setup_outdir_and_yaml(argv_str, return_cfg=True)
    global_cfg.merge_from_dict(cfg)
    global_cfg.merge_from_dict(vars(args))

    modelarts_utils.setup_tl_outdir_obs(global_cfg)
    modelarts_utils.modelarts_sync_results_dir(global_cfg, join=True)
    modelarts_utils.prepare_dataset(global_cfg.get('modelarts_download', {}), global_cfg=global_cfg)

    train_dir = f'{cfg.data_dir}/train'
    counter_cls = 0
    for rootdir, subdir, files in os.walk(train_dir):
      if len(subdir) == 0:
        counter_cls += 1
        extracted_files = sorted(files)[:cfg.num_per_class]
        for file in tqdm.tqdm(extracted_files, desc=f'class: {counter_cls}'):
          img_path = os.path.join(rootdir, file)
          img_rel_path = os.path.relpath(img_path, cfg.data_dir)
          saved_img_path = f'{cfg.saved_dir}/{os.path.dirname(img_rel_path)}'
          os.makedirs(saved_img_path, exist_ok=True)
          shutil.copy(img_path, saved_img_path)
      pass

    modelarts_utils.prepare_dataset(global_cfg.get('modelarts_upload', {}), global_cfg=global_cfg, download=False)
    modelarts_utils.modelarts_sync_results_dir(global_cfg, join=True)
    pass