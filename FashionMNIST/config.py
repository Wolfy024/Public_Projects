import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
batch_size = 64
data_location = r"C:\Users\viraj\PycharmProjects\Public_Projects\FashionMNIST\Data"
epochs = 10
