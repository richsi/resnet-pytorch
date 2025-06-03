import torch
import torch.nn as nn
import torch.optim as optim
import torchvision

from resnet.data import load_cifar10
from resnet.resnet import ResNet34
from resnet.experiment import run

if __name__ == "__main__":

  resize = 256
  batch_size = 64

  # Data init
  train_loader, val_loader= load_cifar10(resize, batch_size)

  # Model init
  device = "cuda" if torch.cuda.is_available() else "cpu"
  my_resnet = ResNet34(num_classes=10).to(device)
  tv_resnet = torchvision.models.resnet34(num_classes=10).to(device)

  # Optimizer & Loss init
  criterion = nn.CrossEntropyLoss()

  optimizer_my = optim.SGD(my_resnet.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
  optimizer_tv = optim.SGD(tv_resnet.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)

  scheduler_my = optim.lr_scheduler.ReduceLROnPlateau(optimizer_my, mode="min", factor=0.1, patience=1) # increase patience on higher epochs
  scheduler_tv = optim.lr_scheduler.ReduceLROnPlateau(optimizer_my, mode="min", factor=0.1, patience=1)

  run(
    my_resnet,
    tv_resnet,
    optimizer_my,
    optimizer_tv,
    scheduler_my,
    scheduler_tv,
    train_loader,
    val_loader,
    criterion,
    device,
    num_epochs=3
  )