import zipfile
import io
import random
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset, random_split


class ZipDataLoader:
    def __init__(self, zip_path, num_files, transform=None, test_split=0.2, batch_size=32):
        """
        Initializes the ZipDataLoader.

        Parameters:
            zip_path (str): Path to the ZIP file.
            num_files (int): Number of files to use from the ZIP.
            transform (callable): Transformation to apply to the data.
            test_split (float): Proportion of data for testing (default: 0.2).
            batch_size (int): Batch size for DataLoader (default: 32).
        """
        self.zip_path = zip_path
        self.num_files = num_files
        self.transform = transform or transforms.ToTensor()
        self.test_split = test_split
        self.batch_size = batch_size

    def _extract_files(self):
        """Extracts the specified number of files from the ZIP archive."""
        with zipfile.ZipFile(self.zip_path, 'r') as zip_file:
            all_files = zip_file.namelist()
            selected_files = random.sample(all_files, min(self.num_files, len(all_files)))
            data = []
            for file_name in selected_files:
                with zip_file.open(file_name) as file:
                    img = Image.open(file).convert('RGB')  # Convert image to RGB
                    data.append(img)
            return data

    def _split_data(self, data):
        """Splits the data into training and testing datasets."""
        test_size = int(len(data) * self.test_split)
        train_size = len(data) - test_size
        return random_split(data, [train_size, test_size])

    def get_data_loaders(self):
        """
        Returns the train and test DataLoader objects.

        Returns:
            tuple: (train_loader, test_loader)
        """
        data = [self.transform(img) for img in self._extract_files()]
        train_data, test_data = self._split_data(data)
        train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=self.batch_size, shuffle=False)
        return train_loader, test_loader


if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    zip_loader = ZipDataLoader(
        zip_path=r"C:\Users\viraj\PycharmProjects\Public_Projects\CyclicWatermark\Data\img_align_celeba.zip",
        num_files=6000, transform=transform, batch_size=1)
    train_loader, test_loader = zip_loader.get_data_loaders()
    print(len(train_loader.dataset[1:33]))
    print(len(train_loader))
