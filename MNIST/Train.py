from Architecture.Model import MNIST
from data import data_loader
from torch.optim.lr_scheduler import StepLR
import config
import torch


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
        # if batch_idx % 10000 == 0:
        #     print(f"Epoch {epoch} Batch {batch_idx}: Loss = {loss.item():.4f}")
        # avg_loss = total_loss / len(train_loader)
        # print(f"Epoch {epoch} Training Loss: {avg_loss:.4f}")


def test(model, criterion, test_loader, device):
    model.eval()
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
    print(f"Test Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
    return accuracy


if __name__ == "__main__":
    device = config.device
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    train_loader, test_loader = data_loader(config.data_location, config.batch_size)
    model = MNIST(1, 32, 10)
    optim = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = StepLR(optim, step_size=10, gamma=0.1)
    criterion = torch.nn.CrossEntropyLoss()
    model.to(device)
    best_accuracy = 0.0

    for epoch in range(config.epochs):
        train(model, optim, criterion, train_loader, device, epoch)
        accuracy = test(model, criterion, test_loader, device)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), rf"Models\mnist_best_model_{accuracy}.pth")
            print(f"Epoch {epoch}: New best accuracy = {best_accuracy:.4f}, model saved.")
        else:
            print(f"Epoch {epoch}: Accuracy = {accuracy:.4f}, no improvement.")