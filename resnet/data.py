import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def load_cifar10(resize, batch_size, num_workers=4):
  assert(resize in [224, 256, 384, 480, 640])

  # 1. Defining transition
  transform = transforms.Compose([
    transforms.Resize((resize, resize)),
    transforms.ToTensor()
  ])

  # 2. Creating CIFAR10 dataset, passing trasform in
  train_dataset = datasets.CIFAR10(
    root="data",
    train=True,
    download=True,
    transform=transform
  )

  test_dataset = datasets.CIFAR10(
    root="data",
    train=False,
    download=True,
    transform=transform
  )

  # 3. Wrap in a DataLoader
  train_loader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=num_workers)
  test_loader = DataLoader(test_dataset, batch_size, shuffle=False, num_workers=num_workers)

  return train_loader, test_loader