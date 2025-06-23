import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
import numpy as np


def plot_embeddings(embeddings: np.ndarray, labels=None, discrete=True, save_path=None):
    df = pd.DataFrame(
        embeddings, columns=[f"Dim_{i+1}" for i in range(embeddings.shape[1])]
    )

    if labels is not None:
        df['Label'] = labels

    plt.style.use('ggplot')
    plt.figure(figsize=(10, 8))

    if labels is not None:
        if discrete:
            # 计算唯一标签的数量
            unique_labels = np.unique(labels)
            n_unique_labels = len(unique_labels)

            # 根据唯一标签数量选择调色板
            if n_unique_labels <= 10:
                palette = sns.color_palette("tab10", n_colors=n_unique_labels)
            else:
                palette = sns.color_palette("husl", n_colors=n_unique_labels)
        else:
            # 连续标签使用渐变色
            palette = "viridis"

        # 绘制散点图
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


def plot_mnist(images: np.ndarray, reconstructed_images: np.ndarray, save_path=None):
    assert images.shape == reconstructed_images.shape, "原始图像和重建图像的形状必须相同"

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


def plot_loss_curve(loss_history: dict, save_path=None):
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


def _handle_plot(save_path=None):
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        if plt.isinteractive():
            plt.show()
        # else:
        #     plt.savefig("output.png", bbox_inches='tight')
        #     plt.close()
