from typing import Dict, Iterable, Callable
import copy
import unittest

import torch
from torch import nn, Tensor

__all__ = ["VerboseModel", "FeatureExtractor", "GradExtractor", ]

class VerboseModel(nn.Module):
  def __init__(self, model: nn.Module, submodels=None):
    super().__init__()
    self.model = copy.deepcopy(model)

    # Register a hook for each layer
    for name, layer in self.model.named_children():
      layer.__name__ = name
      layer.register_forward_hook(self._hook())

    if submodels is not None:
      if not isinstance(submodels, (list, tuple)):
        submodels = list(submodels)
      for submodel in submodels:
        for name, layer in getattr(self.model, submodel).named_children():
          layer.__name__ = f"{submodel}.{name}"
          layer.register_forward_hook(self._hook())
    pass

  def _hook(self, ) -> Callable:
    def fn(layer, input, output):
      if hasattr(input[0], 'shape'):
        input_shape = str(list(input[0].shape))
      else:
        input_shape = str(type(input[0]))

      if hasattr(output, 'shape'):
        output_shape = str(list(output.shape))
      else:
        output_shape = str(type(output))
      print(f"{layer.__name__:<30}: {input_shape:<30}->{output_shape}")

    return fn

  def forward(self, x: Tensor) -> Tensor:
    return self.model(x)


class FeatureExtractor(nn.Module):
  def __init__(self, model: nn.Module, layers: Iterable[str]):
    super().__init__()
    self.model = model
    self.layers = layers
    self._features = {layer: torch.empty(0) for layer in layers}

    for layer_id in layers:
      layer = dict([*self.model.named_modules()])[layer_id]
      layer.register_forward_hook(self.save_outputs_hook(layer_id))
    pass

  def save_outputs_hook(self, layer_id: str) -> Callable:
    def fn(_, __, output):
      self._features[layer_id] = output
    return fn

  def forward(self, x: Tensor) -> Dict[str, Tensor]:
    self._features.clear()
    _ = self.model(x)
    return self._features


class GradExtractor(nn.Module):
  def __init__(self, model: nn.Module, param_names: Iterable[str]=[]):
    super().__init__()
    self.model = model
    self.param_names = param_names
    self._grads = {}

    for name, param in model.named_parameters():
      print(f"{name:<30}: {list(param.shape)}")
      if name in self.param_names:
        param.register_hook(hook=self._hook(name))
    pass

  def _hook(self, name: str) -> Callable:
    def fn(grad):
      self._grads[name] = grad.clone()
      return grad
    return fn

  def forward(self, x: Tensor) -> Dict[str, Tensor]:
    self._grads.clear()
    out = self.model(x)
    return out



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

  def test_VerboseModel(self):

    from torchvision.models.segmentation import deeplabv3_resnet101

    verbose_resnet = VerboseModel(deeplabv3_resnet101(pretrained=True), submodels=['backbone', 'classifier', ])
    dummy_input = torch.ones(10, 3, 256, 256)
    _ = verbose_resnet(dummy_input)

    pass

  def test_FeatureExtractor(self):

    from torchvision.models import resnet50

    verbose_resnet = VerboseModel(resnet50())
    dummy_input = torch.ones(10, 3, 224, 224)

    resnet_features = FeatureExtractor(resnet50(), layers=["layer4", "avgpool", ])
    features = resnet_features(dummy_input)

    print({name: output.shape for name, output in features.items()})

    pass

  def test_GradExtractor(self):

    from torchvision.models import resnet50

    resnet_grad = GradExtractor(resnet50(), param_names=['fc.weight', 'conv1.weight'])
    dummy_input = torch.ones(10, 3, 224, 224)

    out = resnet_grad(dummy_input)
    loss = out.mean()
    loss.backward()
    grads = resnet_grad._grads
    print({name: output.mean() for name, output in grads.items()})

    pass












