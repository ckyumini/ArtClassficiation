import torch
from torch import autocast
from tqdm import tqdm

def train_one_epoch(model, dataloader, criterion, optimizer, device, scaler, accumulation_steps):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    optimizer.zero_grad()
    for step, (images, additional_features, labels) in enumerate(tqdm(dataloader, desc="Training", leave=False)):
        images = images.to(device)
        additional_features = additional_features.to(device).float()
        labels = labels.to(device)

        with autocast('cuda'):
            outputs = model(images, additional_features)
            loss = criterion(outputs, labels)

        # Gradient Accumulation
        loss = loss / accumulation_steps
        scaler.scale(loss).backward()

        if (step + 1) % accumulation_steps == 0 or step == len(dataloader) - 1:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def evaluate(model, dataloader, criterion, device):
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for images, additional_features, labels in tqdm(dataloader, desc="Evaluating", leave=False):
            images = images.to(device)
            additional_features = additional_features.to(device).float()
            labels = labels.to(device)

            with autocast('cuda'):
                outputs = model(images, additional_features)
                loss = criterion(outputs, labels)

            val_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    val_epoch_loss = val_loss / val_total
    val_epoch_acc = val_correct / val_total
    return val_epoch_loss, val_epoch_acc

def test(model, dataloader, criterion, device):
    test_loss, test_acc = evaluate(model, dataloader, criterion, device)
    print(f"\nTest Loss: {test_loss:.4f}, Test Acc: {test_acc * 100:.2f}%")
    return test_loss, test_acc
