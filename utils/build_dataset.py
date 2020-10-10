import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class BuildDataset(Dataset):
    def __init__(self, df, transforms):
        self.img_path = df['image'].values
        self.labels = df['label'].values
        self.transforms = transforms

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        frame = Image.open(self.img_path[idx])
        label = torch.tensor(self.labels[idx], dtype=torch.float)

        return self.transforms(frame), label