import os
from pathlib import Path
import torch
from loguru import logger
import random
import numpy as np


def set_seed(seed):
    """Set random seeds for reproducibility across all relevant libraries."""
    random.seed(seed)  # Python random module
    np.random.seed(seed)  # NumPy
    torch.manual_seed(seed)  # PyTorch
    torch.cuda.manual_seed(seed)  # For CUDA algorithms
    torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
    torch.backends.cudnn.deterministic = True  # Ensure deterministic results
    torch.backends.cudnn.benchmark = False  # Disable optimization that can introduce randomness


def get_latest_checkpoint(checkpoint_dir):
    """Return the path of the latest checkpoint file, or None if not found."""
    if not os.path.exists(checkpoint_dir):
        logger.error(f"Checkpoint dir {checkpoint_dir} does not exist")
        exit(0)

    ckpt_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pt')]
    if not ckpt_files:
        logger.error(f"No checkpoints found in {checkpoint_dir}")
        exit(0)

    def extract_epoch(fname):
        try:
            # Expect filename like: ckpt_epoch_{epoch}.pt
            return int(fname.split('_')[-1].split('.')[0])
        except Exception:
            return -1

    ckpt_files = sorted(ckpt_files, key=extract_epoch)
    return Path(checkpoint_dir) / Path(ckpt_files[-1])