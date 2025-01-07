import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")
batch_size = 32
epochs = 50
lr = 2e-3
width = 32
length = 32
location = r"C:\Users\viraj\PycharmProjects\Public_Projects\CyclicWatermark2\Data\output.zip"

