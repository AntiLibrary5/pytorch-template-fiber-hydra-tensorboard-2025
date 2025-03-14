import os
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from loguru import logger
from torch.utils.tensorboard import SummaryWriter


class TensorBoardLogger:
    def __init__(self, log_dir: str):
        """
        Initialize the TensorBoard logger.

        Args:
            log_dir: Directory where TensorBoard logs will be saved
        """
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir)
        self.log_dir = log_dir
        self.logger = logger
        self.logger.info(f"TensorBoard logging to: {log_dir}")

    def log_scalar(self, tag: str, value: Union[float, torch.Tensor], step: int):
        """
        Log a scalar value to TensorBoard.

        Args:
            tag: Name of the metric
            value: Value of the metric
            step: Training step/iteration
        """
        if isinstance(value, torch.Tensor):
            value = value.item()
        self.writer.add_scalar(tag, value, step)

    def log_scalars(self, main_tag: str, tag_scalar_dict: Dict[str, Union[float, torch.Tensor]], step: int):
        """
        Log multiple scalars with the same main tag.

        Args:
            main_tag: Main tag for the metrics (e.g., 'training', 'validation')
            tag_scalar_dict: Dictionary of tag names to values
            step: Training step/iteration
        """
        processed_dict = {}
        for tag, value in tag_scalar_dict.items():
            if isinstance(value, torch.Tensor):
                processed_dict[tag] = value.item()
            else:
                processed_dict[tag] = value

        self.writer.add_scalars(main_tag, processed_dict, step)

    def log_image(self, tag: str, image: Union[torch.Tensor, np.ndarray], step: int):
        """
        Log an image to TensorBoard.

        Args:
            tag: Name for the image
            image: Image tensor or array
            step: Training step/iteration
        """
        if isinstance(image, np.ndarray):
            # Handle numpy array
            if image.ndim == 2:  # Grayscale
                image = torch.from_numpy(image).unsqueeze(0)
            elif image.ndim == 3 and image.shape[2] in [1, 3, 4]:  # HWC format
                image = torch.from_numpy(image.transpose(2, 0, 1))

        # Handle torch tensor
        if isinstance(image, torch.Tensor):
            if image.ndim == 2:  # Grayscale
                image = image.unsqueeze(0)
            elif image.ndim == 3 and image.dim() == 3:
                if image.shape[0] not in [1, 3, 4]:  # If not in CHW format
                    image = image.permute(2, 0, 1)

            # Make sure it's on CPU and detached
            image = image.cpu().detach()

        self.writer.add_image(tag, image, step)

    def log_images(self, tag: str, images: Union[torch.Tensor, np.ndarray], step: int):
        """
        Log multiple images to TensorBoard.

        Args:
            tag: Name for the images
            images: Batch of images (B, C, H, W)
            step: Training step/iteration
        """
        if isinstance(images, np.ndarray):
            images = torch.from_numpy(images)

        # Make sure it's on CPU and detached
        images = images.cpu().detach()

        self.writer.add_images(tag, images, step)

    def log_histogram(self, tag: str, values: Union[torch.Tensor, np.ndarray], step: int):
        """
        Log a histogram of values to TensorBoard.

        Args:
            tag: Name for the histogram
            values: Values to plot in the histogram
            step: Training step/iteration
        """
        if isinstance(values, np.ndarray):
            values = torch.from_numpy(values)

        self.writer.add_histogram(tag, values, step)

    def log_model_graph(self, model, input_to_model: torch.Tensor):
        """
        Log the model graph to TensorBoard.

        Args:
            model: PyTorch model
            input_to_model: Example input to the model
        """
        self.writer.add_graph(model, input_to_model)

    def close(self):
        """
        Close the TensorBoard writer.
        """
        self.writer.close()
