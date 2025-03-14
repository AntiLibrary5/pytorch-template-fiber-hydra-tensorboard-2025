import matplotlib.pyplot as plt


def visualize_reconstruction(input_image, recon_image, diff_image):
    """
    Visualizes the input, reconstruction, and difference images in a single plot.

    Args:
        input_image (np.ndarray): Input grayscale image.
        recon_image (np.ndarray): Reconstructed grayscale image.
        diff_image (np.ndarray): Absolute difference between input and reconstruction.

    Returns:
        plt.Figure: Matplotlib figure containing the visualization.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Reconstruction Results")

    # Plot input image
    axes[0].imshow(input_image, cmap="gray")
    axes[0].set_title("Input Image")
    axes[0].axis("off")

    # Plot reconstructed image
    axes[1].imshow(recon_image, cmap="gray")
    axes[1].set_title("Reconstructed Image")
    axes[1].axis("off")

    # Plot difference image
    axes[2].imshow(diff_image, cmap="gray")
    axes[2].set_title("Difference Image")
    axes[2].axis("off")

    return fig