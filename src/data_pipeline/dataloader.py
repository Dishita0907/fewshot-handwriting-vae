import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from PIL import Image
import torchvision.transforms as transforms

class HandwritingDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = Path(data_dir)
        self.transform = transform or transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((64, 64)),
            transforms.ToTensor()
        ])
        self.image_paths = list(self.data_dir.rglob("*.png"))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path)
        if self.transform:
            image = self.transform(image)
        return image

def get_dataloader(data_dir, batch_size=32, shuffle=True):
    dataset = HandwritingDataset(data_dir)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)