import os
import argparse
import torch


from template_lib.d2.utils import comm
from template_lib.v2.config import update_parser_defaults_from_yaml


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--local_rank", type=int, default=0)
  parser.add_argument("--run_func", type=str)

  update_parser_defaults_from_yaml(parser)

  args = parser.parse_args()

  n_gpu = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
  args.distributed = n_gpu > 1

  if args.distributed:
    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(backend="nccl", init_method="env://")
    comm.synchronize()

  # eval(args.run_func)()
  return args