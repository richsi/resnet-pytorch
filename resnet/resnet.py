import torch
from torch import nn


class Block(nn.Module):
  def __init__(self):
    super().__init__()

  def forward(self):
    pass



def make_layer(block, in_planes, planes, num_blocks, stride):
  """
  Args:
    block:        which block class (Block)
    in_planes:    number of channels coming in 
    planes:       number of channels in each block
    num_blocks:   how many blocks to stack
    stride:       stride for the first block in this layer
  """
  layers = []
  # First block of layer may downsample with stride
  layers.append(block(in_planes, planes, stride))

  #in_planes = planes * block.expansion


class ResNet34(nn.Module):
  def __init__(self, num_classes=1000):
    super().__init__()

    self.in_planes = 64

    self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=4, bias=False)
    
  def forward(self):
    