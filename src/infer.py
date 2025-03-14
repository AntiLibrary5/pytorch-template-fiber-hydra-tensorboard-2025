import argparse
import sys
from pathlib import Path

import hydra
from loguru import logger
from omegaconf import OmegaConf
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from src.utils.logging.msg_logger import setup_logging
from src.utils.visualization import visualize_reconstruction
from src.utils.io import save_image
from src.utils.utils import get_latest_checkpoint, set_seed
from src.data.loader import get_dataloader


@hydra.main(config_path=None, config_name=None, version_base="1.3")
def infer(cfg):
    # Set up msg logger
    exp_dir = Path(f"./model_save/{cfg.experiment.exp_name}")
    setup_logging(exp_dir=exp_dir, log_filename="infer")

    logger.info("Starting inference.")
    logger.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")

    # Load test dataset
    test_loader = get_dataloader(cfg, phase="test")

    # Load model
    model = hydra.utils.instantiate(cfg.model)
    model.eval()

    # Load checkpoint
    ckpt_path = get_latest_checkpoint(exp_dir) # fetch the last ckpt path in exp dir
    if not ckpt_path.exists():
        logger.error(f"Checkpoint not found at {ckpt_path}. Please ensure the checkpoint exists.")
        return

    logger.info(f"Loading checkpoint from {ckpt_path}")
    ckpt = torch.load(ckpt_path)
    model.load_state_dict(ckpt["model"])

    # Create output directory for predictions
    preds_dir = exp_dir / "preds"
    preds_dir.mkdir(exist_ok=True)

    # Perform inference
    logger.info("Running inference on test dataset...")
    with torch.no_grad():
        for batch_idx, (sample, _) in enumerate(test_loader):
            # Forward pass
            recon = model(sample)

            # Visualize and save results
            for i in range(sample.shape[0]):  # Iterate through batch
                input_image = sample[i][0].cpu().numpy()  # Grayscale image
                recon_image = recon[i][0].cpu().numpy()
                diff_image = abs(input_image - recon_image)  # Absolute difference

                # Save individual images
                save_image(preds_dir / f"input_{batch_idx}_{i}.png", input_image)
                save_image(preds_dir / f"recon_{batch_idx}_{i}.png", recon_image)
                save_image(preds_dir / f"diff_{batch_idx}_{i}.png", diff_image)

                # Create a combined visualization
                fig = visualize_reconstruction(input_image, recon_image, diff_image)
                fig.savefig(preds_dir / f"comparison_{batch_idx}_{i}.png")
                plt.close(fig)

            # Limit the number of batches for quick testing (optional)
            if batch_idx == cfg.inference.max_batches - 1:
                break

    logger.info(f"Inference completed! Results saved in {preds_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--experiment", type=str, required=True)
    args, remaining = parser.parse_known_args()

    # Modify sys.argv to include the right config path and name
    # but preserve any other arguments for hydra to process as overrides
    sys.argv = [sys.argv[0]] + [
        f"--config-dir=./model_save/{args.experiment}/.hydra",
        "--config-name=config"
    ] + remaining
    set_seed(42)
    infer()