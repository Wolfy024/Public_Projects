import data
from Architecture import Model
import torch
import config


def train(model, optim, criterion, train_loader, device, epoch):
    model.train()
    total_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optim.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optim.step()
        total_loss += loss.item()
        # if batch_idx % 100 == 0:
        #     print(f"Epoch {epoch} Batch {batch_idx}: Loss = {loss.item():.4f}")


def evaluate(model, criterion, test_loader, device, epoch):
    model.eval()
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    total_loss = 0
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            total_loss += loss.item()
            _, predicted = torch.max(output, dim=1)
            total_correct += (predicted == target).sum().item()
            total_samples += target.size(0)
    avg_loss = total_loss / len(test_loader)
    accuracy = total_correct / total_samples * 100
    print(f"Test Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%, Epoch: {epoch}")
    return accuracy


if __name__ == "__main__":
    model = Model.CIFAR10Model(3, 8, 10)
    model = model.to(config.device)
    optim = torch.optim.AdamW(model.parameters(), lr=config.lr)
    # scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=10, gamma=0.1)
    criterion = torch.nn.CrossEntropyLoss()
    train_loader, test_loader, classes = data.data_loader(config.location, config.batch_size)
    device = config.device
    best_accuracy = 0.0

    for epoch in range(config.epochs):
        train(model, optim, criterion, train_loader, device, epoch)
        accuracy = evaluate(model, criterion, test_loader, device, epoch)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), rf"Models\mnist_best_model_{accuracy}.pth")
            print(f"Epoch {epoch}: New best accuracy = {best_accuracy:.4f}, model saved.")
        else:
            print(f"Epoch {epoch}: Accuracy = {accuracy:.4f}, no improvement.")
