import torch
from torch import nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
  """
  Creates a basic residual block with downsampling
  
  Args:
    in_planes:  number of channels coming in
    planes:     number of channels going out 
  """
  def __init__(self, in_planes, planes, stride=1):
    super().__init__()

    self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
    self.bn1 = nn.BatchNorm2d(planes)

    self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
    self.bn2 = nn.BatchNorm2d(planes)

    self.downsample = None
    if stride != 1 or in_planes != planes:
      self.downsample = nn.Sequential(
        nn.Conv2d(in_planes, planes, kernel_size=1, stride=2, bias=False),
        nn.BatchNorm2d(planes)
      )

  def forward(self, x):
    identity = x
    out = F.relu(self.bn1(self.conv1(x)))
    out = self.bn2(self.conv2(out))

    if self.downsample is not None:
      identity = self.downsample(x)

    out += identity
    return F.relu(out)





class ResNet34(nn.Module):
  def __init__(self, num_classes=1000):
    super().__init__()
    self.relu = nn.ReLU(inplace=True)

    self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
    self.bn = nn.BatchNorm2d(64)
    self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    self.layer1 = self._make_layer(BasicBlock, 64, 64, num_blocks=3, stride=1)
    self.layer2 = self._make_layer(BasicBlock, 64, 128, num_blocks=4, stride=2)
    self.layer3 = self._make_layer(BasicBlock, 128, 256, num_blocks=6, stride=2)
    self.layer4 = self._make_layer(BasicBlock, 256, 512, num_blocks=3, stride=2)

    self.avgpool = nn.AdaptiveAvgPool2d(output_size=1)
    self.fc = nn.Linear(in_features=512, out_features=num_classes, bias=True)

    
  def forward(self, x):
    x = self.maxpool(self.bn(self.conv1(x)))
    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)
    x = self.avgpool(x)
    x = torch.flatten(x, 1) # (bs x 512) 
    x = self.fc(x)
    return x


  def _make_layer(self, block, in_planes, planes, num_blocks, stride):
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
    for _ in range(1, num_blocks):
      layers.append(block(planes, planes, stride=1)) # will always be of stride 1
    return nn.Sequential(*layers)