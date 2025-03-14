from torch.utils.data import DataLoader
from omegaconf import DictConfig
from src.data.dataset import RandomImageDataset


def get_dataloader(cfg: DictConfig, phase: str = "train") -> DataLoader:
    """
    Returns a DataLoader for the given phase (train/val/test).

    Args:
        cfg (DictConfig): Configuration for the dataset and dataloader.
        phase (str): Phase of the dataloader (train/val/test).

    Returns:
        DataLoader: Configured DataLoader.
    """
    dataset = RandomImageDataset(cfg)

    dataloader_cfg = cfg.data.loader[phase]
    dataloader = DataLoader(
        dataset,
        batch_size=dataloader_cfg.batch_size,
        shuffle=dataloader_cfg.shuffle,
        num_workers=dataloader_cfg.num_workers,
        pin_memory=dataloader_cfg.pin_memory,
    )

    return dataloader