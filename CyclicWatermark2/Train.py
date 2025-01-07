import config
from Architecture.Model import UnetGAN, Discriminator
import data
from torchvision import transforms
import matplotlib.pyplot as plt
import random
import os
import torch

global ijj
ijj = 0


def train(gen1, disc1, gen2, disc2, train_loader, optim1, optim2, epoch, lambda_cycle=10):
    """
    Train function that computes generator and discriminator losses, and updates the weights of both.

    Parameters:
        gen1: Generator 1 (first generator).
        disc1: Discriminator 1.
        gen2: Generator 2 (second generator).
        disc2: Discriminator 2.
        train_loader: DataLoader for the training dataset.
        optim1: Optimizer for generator 1 and discriminator 1.
        optim2: Optimizer for generator 2 and discriminator 2.
        epoch: Current epoch number.
        lambda_cycle: Weight for the cycle consistency loss.
    """
    gen1.train()
    disc1.train()
    gen2.train()
    disc2.train()

    for batch_idx, data in enumerate(train_loader):
        original = data.to(config.device)
        watermark = get_batch_from_dataset(train_loader.dataset, config.batch_size).to(config.device)

        feedthis = torch.cat((original, watermark), dim=1)

        optim1.zero_grad()
        optim2.zero_grad()

        fake1 = gen1(feedthis)
        fake2 = gen2(fake1)

        disc1_real = disc1(original)
        disc1_fake = disc1(fake1)
        disc2_real = disc2(watermark)
        disc2_fake = disc2(fake2)

        loss1 = torch.nn.BCELoss()(disc1_real, torch.ones_like(disc1_real)) + \
                torch.nn.BCELoss()(disc1_fake, torch.zeros_like(disc1_fake))
        loss2 = torch.nn.BCELoss()(disc2_real, torch.ones_like(disc2_real)) + \
                torch.nn.BCELoss()(disc2_fake, torch.zeros_like(disc2_fake))

        g_loss1 = torch.nn.BCELoss()(disc1_fake, torch.ones_like(disc1_fake))
        g_loss2 = torch.nn.BCELoss()(disc2_fake, torch.ones_like(disc2_fake))

        cycle_loss = torch.nn.L1Loss()(fake1, original) + torch.nn.L1Loss()(fake2, watermark)
        cycle_loss *= lambda_cycle

        g_total_loss = g_loss1 + g_loss2 + cycle_loss

        loss1.backward(retain_graph=True)
        loss2.backward(retain_graph=True)

        g_total_loss.backward()

        optim1.step()
        optim2.step()

        print(f"Epoch {epoch} Batch {batch_idx}: Loss1 = {loss1.item():.4f}, Loss2 = {loss2.item():.4f}, "
              f"Gen1 Loss = {g_loss1.item():.4f}, Gen2 Loss = {g_loss2.item():.4f}, Cycle Loss = {cycle_loss.item():.4f}")


def test(gen1, gen2, test_loader, output_folder="output_images"):
    global ijj
    """
    Test function to visualize the original image, watermark image, and generator outputs, 
    and save the plot as an image.

    Parameters:
        gen1: CyclicGenerator1 instance.
        gen2: CyclicGenerator2 instance.
        test_loader: DataLoader object.
        output_folder: Folder where the plot image will be saved.
    """
    os.makedirs(output_folder, exist_ok=True)

    data_iter = iter(test_loader)
    original_batch = next(data_iter)
    watermark_batch = next(data_iter)

    original = original_batch[0].to(config.device).unsqueeze(0)  #Add batch dimension
    watermark = watermark_batch[0].to(config.device).unsqueeze(0)

    feedthis = torch.cat((original, watermark), dim=1)

    gen1.eval()
    gen2.eval()
    with torch.no_grad():
        fake1 = gen1(feedthis)
        fake2 = gen2(fake1)

    def unnormalize(img):
        img = img.squeeze(0).permute(1, 2, 0).cpu()
        img = (img * 0.5 + 0.5)
        return img

    def accuracy(img1, img2):
        return torch.mean(torch.abs(img1 - img2))

    original_img = unnormalize(original)
    watermark_img = unnormalize(watermark)
    fake1_img = fake1.squeeze(0).permute(1, 2, 0).cpu()
    fake2_img = fake2.squeeze(0).permute(1, 2, 0).cpu()

    print(f"Accuracy of Gen1: {accuracy(original, fake1):.4f}")
    print(f"Accuracy of Gen2: {accuracy(watermark, fake2):.4f}")

    plt.figure(figsize=(10, 8))
    plt.subplot(2, 2, 1)
    plt.imshow(original_img)
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(2, 2, 2)
    plt.imshow(watermark_img)
    plt.title("Watermark Image")
    plt.axis("off")

    plt.subplot(2, 2, 3)
    plt.imshow(fake1_img)
    plt.title("Output of Gen1")
    plt.axis("off")

    plt.subplot(2, 2, 4)
    plt.imshow(fake2_img)
    plt.title("Output of Gen2")
    plt.axis("off")

    plot_filename = os.path.join(output_folder, f"output_plot{ijj}.png")
    ijj += 1
    plt.tight_layout()
    plt.savefig(plot_filename)
    plt.close()

    print(f"Saved plot to: {plot_filename}")


def get_batch_from_dataset(dataset, batch_size=32):
    """
    Creates a batch of images from the dataset.

    Parameters:
        dataset (Dataset): The dataset to sample from.
        batch_size (int): The number of images in the batch.

    Returns:
        Tensor: A batch of images.
    """
    indices = random.sample(range(len(dataset)), batch_size)
    images = [dataset[idx] for idx in indices]
    batch_images = torch.stack(images)
    return batch_images


if __name__ == "__main__":
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    gen1 = UnetGAN(6, 3).to(config.device)
    disc1 = Discriminator().to(config.device)
    gen2 = UnetGAN(3, 3).to(config.device)
    disc2 = Discriminator().to(config.device)
    zip_loader = data.ZipDataLoader(
        zip_path=config.location,
        num_files=2000, transform=transform, batch_size=config.batch_size)
    train_loader, test_loader = zip_loader.get_data_loaders()
    optim1 = torch.optim.Adam(list(gen1.parameters()) + list(disc1.parameters()), lr=config.lr)
    optim2 = torch.optim.Adam(list(gen2.parameters()) + list(disc2.parameters()), lr=config.lr)
    for epoch in range(config.epochs):
        train(gen1, disc1, gen2, disc2, train_loader, optim1, optim2, epoch)
        test(gen1, gen2, test_loader)
