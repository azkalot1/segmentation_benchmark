import torch
from torch.utils.data import Dataset
import numpy as np
from skimage.io import imread


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
        sample_image = self.images[idx].astype(np.uint8)
        sample_mask = np.argmax((self.masks[idx] > 0).astype(int), axis=2)
        sample_type = self.types[idx]
        augmented = self.transforms(image=sample_image, mask=sample_mask)
        sample_image = augmented['image']
        sample_mask = augmented['mask']
        sample_mask = sample_mask.astype(np.int64)
        sample_image = sample_image.transpose(2, 0, 1)  # channels first

        data = {
            'features': torch.from_numpy(sample_image.copy()).float(),
            'type': sample_type,
            'mask': torch.from_numpy(sample_mask.copy())
            }
        return(data)


class ChestXRayDataset(Dataset):
    def __init__(
        self,
        images,
        masks,
            transforms):
        self.images = images
        self.masks = masks
        self.transforms = transforms

    def __len__(self):
        return(len(self.images))

    def __getitem__(self, idx):
        """Will load the mask, get random coordinates around/with the mask,
        load the image by coordinates
        """
        sample_image = imread(self.images[idx])
        if len(sample_image.shape) == 3:
            sample_image = sample_image[..., 0] / 255
        else:
            sample_image = np.expand_dims(sample_image, 2) / 255
        sample_mask = imread(self.masks[idx]) / 255
        augmented = self.transforms(image=sample_image, mask=sample_mask)
        sample_image = augmented['image']
        sample_mask = augmented['mask']
        sample_image = sample_image.transpose(2, 0, 1)  # channels first
        sample_mask = np.expand_dims(sample_mask, 0)
        data = {'features': torch.from_numpy(sample_image.copy()).float(),
                'mask': torch.from_numpy(sample_mask.copy())}
        return(data)
