import sys
import yaml
import logging
import os
import math
import numpy as np
import time
import unittest

from fvcore.common.config import CfgNode as _CfgNode


BASE_KEY = "base"


class TLCfgNode(_CfgNode):
    """
    """
    @staticmethod
    def merge_a_into_b(a, b):
      # merge dict a into dict b. values in a will overwrite b.
      for k, v in a.items():
        if isinstance(v, dict) and k in b:
          assert isinstance(
            b[k], dict
          ), "Cannot inherit key '{}' from base!".format(k)
          TLCfgNode.merge_a_into_b(v, b[k])
        else:
          b[k] = v

    @staticmethod
    def _merge_base_cfg(cfg, loaded_cfg):
      if BASE_KEY in cfg:
        base_cfg = loaded_cfg.get(cfg[BASE_KEY])
        del cfg[BASE_KEY]
        TLCfgNode._merge_base_cfg(base_cfg, loaded_cfg)
        TLCfgNode.merge_a_into_b(cfg, base_cfg)
        cfg.clear()
        cfg.update(base_cfg)

    @staticmethod
    def load_yaml_with_command(cfg_filename: str, command: str, allow_unsafe: bool = False):
      """
      """
      loaded_cfg = TLCfgNode.load_yaml_with_base(cfg_filename, allow_unsafe=allow_unsafe)

      command_cfg = loaded_cfg.get(command)
      TLCfgNode._merge_base_cfg(command_cfg, loaded_cfg)

      command_cfg = TLCfgNode(command_cfg)
      return command_cfg

    def dump_to_file(self, saved_file):
      with open(saved_file, "w") as f:
        self.dump(stream=f, sort_keys=False, indent=2)

    def dump_to_file_with_command(self, saved_file, command):
      command_cfg = TLCfgNode(new_allowed=True)
      setattr(command_cfg, command, self)
      with open(saved_file, "w") as f:
        command_cfg.dump(stream=f, sort_keys=False, indent=2)

    def merge_from_list(self, opt_list, new_allowed=False):

      if new_allowed:
        for k in opt_list[::2]:
          sub_k_list = k.split('.')
          cur_cfg = self
          for idx, sub_k in enumerate(sub_k_list):
            if sub_k not in cur_cfg:
              if idx != len(sub_k_list) - 1:
                cur_cfg.setdefault(sub_k, TLCfgNode())
                cur_cfg = cur_cfg.get(sub_k)
              else:
                cur_cfg.setdefault(sub_k)
            else:
              if idx != len(sub_k_list) - 1:
                cur_cfg = cur_cfg.get(sub_k)
      super(TLCfgNode, self).merge_from_list(opt_list)
      return self

global_cfg = TLCfgNode()

def set_global_cfg(cfg: TLCfgNode) -> None:
  global global_cfg
  global_cfg.clear()
  global_cfg.update(cfg)
  pass



class Test_TLCfgNode(unittest.TestCase):

  def test_load_yaml_with_command(self):
    """
    """
    config_yaml = "template_lib/v2/tests/configs/config.yaml"
    command = 'test2'

    cfg = TLCfgNode.load_yaml_with_command(config_yaml, command=command)
    pass

  def test_merge_list(self):
    """
    """
    sys.argv = [__file__] + "--tl_opts key5.sub1 01 --test test".split(' ')
    config_yaml = "template_lib/v2/tests/configs/config.yaml"
    command = 'test2'

    cfg = TLCfgNode.load_yaml_with_command(config_yaml, command=command)

    from argparse import ArgumentParser
    description = 'testing for passing multiple arguments and to get list of args'
    parser = ArgumentParser(description=description)
    parser.add_argument('--tl_opts', type=str, nargs='*', default=[])
    parser.add_argument('--test', type=str)
    args = parser.parse_args()

    cfg.merge_from_list(args.tl_opts)
    pass



