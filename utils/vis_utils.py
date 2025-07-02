import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
import numpy as np


def plot_embeddings(embeddings: np.ndarray, labels=None, discrete=True, save_path=None):
    """Visualizes high-dimensional embeddings in 2D space with optional labeling.
    
    Args:
        embeddings: A 2D numpy array of shape (n_samples, 2) containing the embeddings.
        labels: Optional array-like of shape (n_samples,) for coloring points. Can be discrete or continuous.
        discrete: If True (default), treats labels as discrete categories. If False, treats as continuous values.
        save_path: Optional path to save the figure. If None, displays the plot interactively.
    
    Example:
        >>> embeddings = np.random.rand(100, 50)
        >>> labels = np.random.randint(0, 3, 100)
        >>> plot_embeddings(embeddings, labels)
    """
    df = pd.DataFrame(
        embeddings, columns=[f"Dim_{i+1}" for i in range(embeddings.shape[1])]
    )

    if labels is not None:
        df['Label'] = labels

    plt.style.use('ggplot')
    plt.figure(figsize=(10, 8))

    if labels is not None:
        if discrete:
            unique_labels = np.unique(labels)
            n_unique_labels = len(unique_labels)

            if n_unique_labels <= 10:
                palette = sns.color_palette("tab10", n_colors=n_unique_labels)
            else:
                palette = sns.color_palette("husl", n_colors=n_unique_labels)
        else:
            palette = "viridis"

        sns.scatterplot(
            x="Dim_1", y="Dim_2", hue="Label", palette=palette, data=df
        )
        plt.legend(title="Label", bbox_to_anchor=(1.05, 1), loc='upper left')
    else:
        sns.scatterplot(
            x="Dim_1", y="Dim_2", data=df
        )

    plt.title("Embedding Feature", fontsize=12)
    plt.xlabel("Dim 1", fontsize=12)
    plt.ylabel("Dim 2", fontsize=12)

    _handle_plot(save_path)


def plot_loss_curve(loss_history: dict, save_path=None):
    """Plots training loss curves from a dictionary of loss values over epochs.
    
    Args:
        loss_history: Dictionary where keys are loss names (e.g., 'train_loss', 'val_loss')
                     and values are lists of loss values per epoch.
        save_path: Optional path to save the figure. If None, displays the plot interactively.
    
    Example:
        >>> loss_history = {
        ...     'train_loss': [0.9, 0.6, 0.4, 0.3],
        ...     'val_loss': [1.0, 0.7, 0.5, 0.4]
        ... }
        >>> plot_loss_curve(loss_history)
    """
    loss_df = pd.DataFrame(loss_history)
    count = len(loss_history.keys())

    loss_df.reset_index(inplace=True)
    loss_df = pd.melt(loss_df, id_vars='index',
                      var_name='Variable', value_name='Value')

    plt.figure(figsize=(10, 8))

    sns.set_theme(style="darkgrid")
    palette = sns.color_palette("husl", count)

    sns.lineplot(
        data=loss_df,
        x='index',
        y='Value',
        hue='Variable',
        style='Variable',
        palette=palette,
        dashes=True
    )

    plt.title("Training Loss", fontsize=12)
    plt.xlabel("Epochs", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.grid(True)
    plt.legend(title="Loss Types", fontsize=12)

    _handle_plot(save_path)


def compare_mnist_reconstructions(images: np.ndarray, reconstructed_images: np.ndarray, save_path=None):
    """Displays side-by-side comparison of original and reconstructed MNIST images.
    
    Randomly samples 10 images and shows original (top row) and reconstructed (bottom row) versions.
    
    Args:
        images: Original MNIST images tensor of shape (n_samples, 28, 28) or (n_samples, 784)
        reconstructed_images: Reconstructed images tensor of same shape as images
        save_path: Optional path to save the figure. If None, displays the plot interactively.
    
    Raises:
        AssertionError: If input and reconstructed image shapes don't match.
    
    Example:
        >>> images = torch.randn(100, 1, 28, 28)
        >>> reconstructions = torch.randn(100, 1, 28, 28)
        >>> compare_mnist_reconstructions(images, reconstructions)
    """
    assert images.shape == reconstructed_images.shape, "The shapes of origin and reconstructed images must be the same."

    num_samples = 10
    indices = torch.randperm(images.size(0))[:num_samples]
    original_samples = images[indices]
    refined_samples = reconstructed_images[indices]

    original_samples = original_samples.view(-1, 28, 28)
    refined_samples = refined_samples.view(-1, 28, 28)

    plt.figure(figsize=(10, 5))

    for i in range(num_samples):
        plt.subplot(2, num_samples, i + 1)
        plt.imshow(original_samples[i], cmap='gray')
        plt.axis('off')
        if i == 0:
            plt.title("Original", fontsize=12)

    for i in range(num_samples):
        plt.subplot(2, num_samples, num_samples + i + 1)
        plt.imshow(refined_samples[i], cmap='gray')
        plt.axis('off')
        if i == 0:
            plt.title("Reconstructed", fontsize=12)

    plt.tight_layout()
    _handle_plot(save_path)


def _handle_plot(save_path=None):
    """Internal helper function to handle plot display/saving.
    
    Args:
        save_path: If provided, saves plot to this path. Otherwise shows the plot.
    """
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        if plt.isinteractive():
            plt.show()
