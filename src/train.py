import hydra
from loguru import logger
from omegaconf import OmegaConf
import torch
import torch.nn as nn
import torch.optim as optim
from lightning.fabric import Fabric

from src.utils.logging.msg_logger import setup_logging
from src.utils.logging.tb_logger import TensorBoardLogger
from src.data.loader import get_dataloader
from src.utils.utils import get_latest_checkpoint, set_seed


@hydra.main(config_path="../configs", config_name="config.yaml", version_base="1.3")
def train(cfg):
    # Set up msg logger
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    setup_logging(exp_dir=hydra_cfg.run.dir, log_filename=hydra_cfg.job.name)

    # Set up TensorBoard logger
    tb_dir = f"{hydra_cfg.run.dir}/tb"
    tb_logger = TensorBoardLogger(tb_dir)

    logger.opt(colors=True).info("<blue>Starting experiment:</blue> <green>{}</green> <blue>in directory</blue> <green>{}</green>", cfg.experiment.exp_name, hydra_cfg.run.dir)
    logger.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")

    # Initialize Lightning Fabric for hardware-agnostic training
    fabric = Fabric(accelerator=cfg.training.accelerator, devices=cfg.training.devices, precision=cfg.training.precision)
    fabric.launch()
    # Log selected configuration
    logger.opt(colors=True).info("<blue>Using strategy:</blue> <green>{}</green>", fabric.strategy.__class__.__name__)
    logger.opt(colors=True).info("<blue>Using device(s):</blue> <green>{}</green>", fabric.device)

    # Create dataset and dataloader
    train_loader = get_dataloader(cfg, phase="train")
    train_loader = fabric.setup_dataloaders(train_loader)

    # Create model, optimizer, and loss function
    model = hydra.utils.instantiate(cfg.model)
    optimizer = optim.Adam(model.parameters(), lr=cfg.training.lr)
    criterion = nn.MSELoss()

    # Wrap model and optimizer with Fabric
    model, optimizer = fabric.setup(model, optimizer)

    # Resume checkpoint if needed
    if cfg.training.resume_from_last:
        ckpt_path = get_latest_checkpoint(hydra_cfg.run.dir)  # fetch the last ckpt path in exp dir
        logger.info(f"Resuming from checkpoint: {ckpt_path}")
        fabric.load(ckpt_path, {"model": model, "optimizer": optimizer})

    # Training loop
    for epoch in range(cfg.training.epochs):
        epoch_loss = 0.0

        # Training logic
        model.train()
        for batch_idx, (sample, target) in enumerate(train_loader):
            optimizer.zero_grad()

            # Forward pass
            recon = model(sample)
            loss = criterion(recon, target)

            # Backward pass
            fabric.backward(loss)
            optimizer.step()

            epoch_loss += loss.item()

        # Log epoch metrics
        if epoch % cfg.training.logging.epoch_frequency == 0:
            epoch_loss /= len(train_loader)
            logger.info(f"Epoch {epoch + 1}/{cfg.training.epochs}, Loss: {epoch_loss:.4f}")
            tb_logger.log_scalar("training/epoch_loss", epoch_loss, epoch)

        # Log images
        if epoch % cfg.training.logging.image_frequency == 0:
            tb_sample = sample[0][0].cpu()  # Log first image in the batch
            tb_recon = recon[0][0].detach().cpu()  # Log its reconstruction
            tb_diff = torch.abs(tb_sample - tb_recon)
            tb_logger.log_image("training/sample", tb_sample, epoch)
            tb_logger.log_image("training/recon", tb_recon, epoch)
            tb_logger.log_image("training/diff", tb_diff, epoch)

        # Save checkpoint
        if epoch % cfg.training.ckpt_frequency == 0:
            ckpt_path = f"{hydra_cfg.run.dir}/ckpt_epoch_{epoch}.pt"
            fabric.save(ckpt_path, {"model": model, "optimizer": optimizer})
            logger.info(f"Checkpoint saved at {ckpt_path}")

    logger.info("Training completed!")
    tb_logger.close()


if __name__ == "__main__":
    set_seed(42)
    train()