import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

def load_cifar10(resize, batch_size, num_workers=4):
  assert(resize in [224, 256, 384, 480, 640])

  # 1. Defining transition
  transform = transforms.Compose([
    transforms.Resize((resize, resize)),
    transforms.ToTensor(),
    transforms.Normalize(
      mean=[0.485, 0.456, 0.406], # standard ImageNet means
      std=[0.229, 0.224, 0.225]
    )
  ])

  # 2. Creating CIFAR10 dataset, passing trasform in
  full_train = datasets.CIFAR10(
    root="data",
    train=True,
    download=True,
    transform=transform
  )

  train_len = int(len(full_train) * 0.9)
  val_len = len(full_train) - train_len
  train_set, val_set = random_split(full_train, [train_len, val_len])

  train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
  val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

  return train_loader, val_loader 