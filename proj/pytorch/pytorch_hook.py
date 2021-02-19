import os
import subprocess
import sys
import unittest
import argparse

from template_lib.examples import test_bash
from template_lib import utils
from template_lib.nni import update_nni_config_file


class PytorchHook(unittest.TestCase):

  def test_register_forward_hook(self):

    import torch
    import torch.nn as nn

    class TestForHook(nn.Module):
      def __init__(self):
        super().__init__()

        self.linear_1 = nn.Linear(in_features=2, out_features=2)
        self.linear_2 = nn.Linear(in_features=2, out_features=1)
        self.relu = nn.ReLU()
        self.relu6 = nn.ReLU6()
        self.initialize()

      def forward(self, x):
        linear_1 = self.linear_1(x)
        linear_2 = self.linear_2(linear_1)
        relu = self.relu(linear_2)
        relu_6 = self.relu6(relu)
        layers_in = (x, linear_1, linear_2)
        layers_out = (linear_1, linear_2, relu)
        return relu_6, layers_in, layers_out

      def initialize(self):
        """ 定义特殊的初始化，用于验证是不是获取了权重"""
        self.linear_1.weight = torch.nn.Parameter(torch.FloatTensor([[1, 1], [1, 1]]))
        self.linear_1.bias = torch.nn.Parameter(torch.FloatTensor([1, 1]))
        self.linear_2.weight = torch.nn.Parameter(torch.FloatTensor([[1, 1]]))
        self.linear_2.bias = torch.nn.Parameter(torch.FloatTensor([1]))
        return True

    # 1：定义用于获取网络各层输入输出tensor的容器
    # 并定义module_name用于记录相应的module名字
    module_name = []
    features_in_hook = []
    features_out_hook = []

    # 2：hook函数负责将获取的输入输出添加到feature列表中
    # 并提供相应的module名字
    def hook(module, fea_in, fea_out):
      print("hooker working")
      module_name.append(module.__class__)
      features_in_hook.append(fea_in)
      features_out_hook.append(fea_out)
      return None

    # 3：定义全部是1的输入
    x = torch.FloatTensor([[0.1, 0.1], [0.1, 0.1]])

    # 4:注册钩子可以对某些层单独进行
    net = TestForHook()
    net_chilren = net.children()
    for child in net_chilren:
      if not isinstance(child, nn.ReLU6):
        child.register_forward_hook(hook=hook)

    # 5:测试网络输出
    out, features_in_forward, features_out_forward = net(x)
    print("*" * 5 + "forward return features" + "*" * 5)
    print(features_in_forward)
    print(features_out_forward)
    print("*" * 5 + "forward return features" + "*" * 5)

    # 6:测试features_in是不是存储了输入
    print("*" * 5 + "hook record features" + "*" * 5)
    print(features_in_hook)
    print(features_out_hook)
    print(module_name)
    print("*" * 5 + "hook record features" + "*" * 5)

    # 7：测试forward返回的feautes_in是不是和hook记录的一致
    print("sub result")
    for forward_return, hook_record in zip(features_in_forward, features_in_hook):
      print(forward_return - hook_record[0])

    pass






  