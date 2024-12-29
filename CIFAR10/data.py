import torch
from torchvision import datasets, transforms


def data_loader(location: str, batch_size: int):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    train_dataset = datasets.CIFAR10(root=location, train=True, transform=transform, download=True)
    test_dataset = datasets.CIFAR10(root=location, train=False, transform=transform, download=True)
    class_names = train_dataset.classes
    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader, class_names


if __name__ == '__main__':
    train_loader, test_loader, class_names = data_loader(location='Data', batch_size=64)
    print(class_names)
    for i in train_loader:
        print(i[0].shape, i[1])
        break
