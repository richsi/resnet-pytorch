import torch

from resnet.data import load_cifar10
from resnet.resnet import ResNet34

if __name__ == "__main__":

  resize = 256
  batch_size = 64

  cifar_train_loader, cifar_test_loader = load_cifar10(resize, batch_size)

  # for images, labels in loader:

  model = ResNet34()