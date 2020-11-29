import torch
from torch.utils.data import Dataset
import numpy as np


class PanNukeDataset(Dataset):
    def __init__(
        self,
        images,
        masks,
        types,
            transforms):
        self.images = images
        self.masks = masks
        self.types = types
        self.transforms = transforms

    def __len__(self):
        return(len(self.images))

    def __getitem__(self, idx):
        """Will load the mask, get random coordinates around/with the mask,
        load the image by coordinates
        """
        sample_image = self.images[idx]
        sample_mask = np.argmax(self.masks[idx].astype(int), axis=2)
        sample_type = self.types[idx]
        augmented = self.transforms(image=sample_image, mask=sample_mask)
        sample_image = augmented['image']
        sample_image = sample_image.transpose(2, 0, 1)  # channels first

        data = {
            'features': torch.from_numpy(sample_image.copy()).float(),
            'type': sample_type,
            'mask': torch.from_numpy(sample_mask.copy())
            }
        return(data)
