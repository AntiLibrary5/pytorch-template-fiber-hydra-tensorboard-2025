import torch
from torch.utils.data import Dataset
from omegaconf import DictConfig


class RandomImageDataset(Dataset):
    """A simple dataset that returns a fixed random grayscale image."""

    def __init__(self, cfg: DictConfig):
        """
        Args:
            cfg (DictConfig): Configuration for the dataset.
        """
        self.cfg = cfg.data
        self.image_size = self.cfg.image_size
        self.transform = self._get_transform()

        # Generate a fixed random image
        self.fixed_image = torch.rand(1, self.image_size, self.image_size)

    def _get_transform(self):
        """Placeholder for data transformations."""
        # Users can add their own transforms here
        return None

    def __len__(self):
        return self.cfg.num_samples

    def __getitem__(self, idx):
        """
        Returns the fixed random image and its identity.
        """
        if self.transform:
            image = self.transform(self.fixed_image)
        else:
            image = self.fixed_image

        return image, image  # Input and target are the same for reconstruction