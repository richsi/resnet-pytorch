import torch 

def run(
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
):

  history = {
    "my_train_loss": [], "my_train_acc": [], "my_val_loss": [], "my_val_acc": [],
    "tv_train_loss": [], "tv_train_acc": [], "tv_val_loss": [], "tv_val_acc": []
  }

  for epoch in range(num_epochs):
    # Train
    my_train_loss, my_train_acc = train_one_epoch(my_resnet, optimizer_my, train_loader, criterion, device)
    tv_train_loss, tv_train_acc = train_one_epoch(tv_resnet, optimizer_tv, train_loader, criterion, device)

    # Validate
    my_val_loss, my_val_acc = validate(my_resnet, val_loader, criterion, device)
    tv_val_loss, tv_val_acc = validate(tv_resnet, val_loader, criterion, device)

    # Step schedulers
    scheduler_my.step()
    scheduler_tv.step()

    # Record
    history["my_train_loss"].append(my_train_loss)
    history["my_train_acc"].append(my_train_acc)
    history["my_val_loss"].append(my_val_loss)
    history["my_val_acc"].append(my_val_acc)

    history["tv_train_loss"].append(tv_train_loss)
    history["tv_train_acc"].append(tv_train_acc)
    history["tv_val_loss"].append(tv_val_loss)
    history["tv_val_acc"].append(tv_val_acc)

    # Print progress
    print(f"Epoch {epoch+1}/{num_epochs}")
    print(f"  My   | Train Loss: {my_train_loss:.4f}, Train Acc: {my_train_acc:.4f}  | Val Loss: {my_val_loss:.4f}, Val Acc: {my_val_acc:.4f}")
    print(f"  Torch| Train Loss: {tv_train_loss:.4f}, Train Acc: {tv_train_acc:.4f}  | Val Loss: {tv_val_loss:.4f}, Val Acc: {tv_val_acc:.4f}")
    print("-" * 70)

  print("Final Results after {} epochs:".format(num_epochs))
  print(f"My ResNet34    → Val Acc: {history['my_val_acc'][-1]:.4f}")
  print(f"Torch ResNet34 → Val Acc: {history['tv_val_acc'][-1]:.4f}")



def train_one_epoch(model, optimizer, loader, criterion, device):
  model.train()
  total_loss, correct, total = 0.0, 0, 0
  for images, labels in loader:
    images, labels = images.to(device), labels.to(device)
    optimizer.zero_grad()
    outputs = model(images)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    total_loss += loss.item() * images.size(0)
    preds = outputs.argmax(dim=1)
    correct += (preds == labels).sum().item()
    total += images.size(0)

  avg_loss = total_loss / total
  accuracy = correct / total
  return avg_loss, accuracy


def validate(model, loader, criterion, device):
  model.eval()
  total_loss, correct, total = 0.0, 0, 0
  with torch.no_grad():
    for images, labels in loader:
      images, labels = images.to(device), labels.to(device)
      outputs = model(images)
      loss = criterion(outputs, labels)

      total_loss += loss.item() * images.size(0)
      preds = outputs.argmax(dim=1)
      correct += (preds == labels).sum().item()
      total += images.size(0)

  avg_loss = total_loss / total
  accuracy = correct / total
  return avg_loss, accuracy