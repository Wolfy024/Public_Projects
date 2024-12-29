import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision import models
import config
import data


def train(model, loader, criterion, optimizer):
    model.train()
    running_loss = 0
    correct = 0
    total = 0
    for images, labels in loader:
        images, labels = images.to(config.device), labels.to(config.device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    accuracy = 100 * correct / total
    return running_loss / len(loader), accuracy


def test(model, loader, criterion):
    model.eval()
    running_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(config.device), labels.to(config.device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    accuracy = 100 * correct / total
    return running_loss / len(loader), accuracy


if __name__ == "__main__":
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    model = models.resnet50(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 10)  # Update the final layer for CIFAR-10 (10 classes)
    model = model.to(config.device)
    criterion = nn.CrossEntropyLoss()
    train_loader, test_loader, class_names = data.data_loader(location=config.location, batch_size=64)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    best_accuracy = 0
    for epoch in range(config.epochs):
        train_loss, train_accuracy = train(model, train_loader, criterion, optimizer)
        test_loss, test_accuracy = test(model, test_loader, criterion)
        scheduler.step()
        print(f"Epoch [{epoch + 1}/{config.epochs}]")
        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%")
        print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")
        # Save the best model
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            torch.save(model.state_dict(), rf"C:\Users\viraj\PycharmProjects\Public_Projects\CIFAR10_Pre\Models\model_acc_{test_accuracy:.2f}.pth")
            print("Model saved!")

    print(f"Best Test Accuracy: {best_accuracy:.2f}%")
