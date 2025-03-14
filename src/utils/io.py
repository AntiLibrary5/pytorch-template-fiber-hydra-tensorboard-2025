from pathlib import Path
import matplotlib.pyplot as plt


def save_image(filepath, image):
    """
    Saves an image to the specified filepath.

    Args:
        filepath (Path or str): Path to save the image.
        image (np.ndarray): Image to save.
    """
    plt.imsave(filepath, image, cmap="gray")