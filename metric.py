import os
import random
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import MDS, TSNE, smacof
from data.archive import gene_loader
from umap import UMAP

from data import cifar_loader, mnist_loader
from utils import data_utils, metric_utils, vis_utils, model_utils

default_device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
default_dtype = torch.float64


def dcm_metric(data):
    eps = 1e-8
    data = data.to(device=default_device, dtype=default_dtype)
    samples = data
    # print(type(data))
    features = data.T
    # 规范化图像特征和文本特征
    samples = samples / (samples.norm(dim=-1, keepdim=True) + eps)
    features = features / (features.norm(dim=-1, keepdim=True) + eps)

    distance_s_s = torch.cdist(samples, samples, p=2)
    distance_f_f = torch.cdist(features, features, p=2)
    # print(distance_s_s == distance_s_s.T)
    # print(distance_f_f.shape)
    # print(distance_f_f == distance_f_f.T)
    similarity_matrix_s_f = torch.mm(samples, torch.eye(samples.size(1), device=samples.device, dtype=samples.dtype))
    distance_s_f = 1 - similarity_matrix_s_f
    # 计算各个距离矩阵的均值
    # mean_all = (mean_i_i + mean_t_t + mean_i_t) / 3
    distance_s_s /= torch.mean(distance_s_s)
    distance_f_f /= torch.mean(distance_f_f)
    distance_s_f /= torch.mean(distance_s_f)
    # 合并距离矩阵
    combined_distance = (
        torch.cat(
            [
                torch.cat([distance_s_s, distance_s_f], dim=1),
                torch.cat([distance_s_f.T, distance_f_f], dim=1),
            ],
            dim=0,
        )
        .cpu()
        .numpy()
    )
    # print(combined_distance == combined_distance.T)
    # combined_distance = (combined_distance + combined_distance.T) / 2
    bz = samples.shape[0]
    # if self.path == "train":
    embeddings, stress = smacof(
        combined_distance,
        n_components=2,
        metric=True,
        n_init=1,
        max_iter=300,
        normalized_stress="auto",
    )
    # else:
    #     data, stress = smacof(
    #         combined_distance,
    #         n_components=2,
    #         metric=True,
    #         n_init=1,
    #         max_iter=300,
    #         normalized_stress="auto",
    #     )
    image_low_dim_data = embeddings[:bz]
    text_low_dim_data = embeddings[bz:]

    data = data.cpu().detach().numpy()

    t30 = metric_utils.trustworthiness(data, image_low_dim_data)
    c30 = metric_utils.continuity(data, image_low_dim_data)

    return t30, c30, image_low_dim_data


def tsne_metric(data):
    data = data_utils.to_numpy(data)
    tsne = TSNE(n_components=2)
    embeddings = tsne.fit_transform(data)

    t30 = metric_utils.trustworthiness(data, embeddings)
    c30 = metric_utils.continuity(data, embeddings)

    return t30, c30, embeddings


def mds_metric(data):
    data = data_utils.to_numpy(data)
    mds = MDS(n_components=2)
    embeddings = mds.fit_transform(data)

    t30 = metric_utils.trustworthiness(data, embeddings)
    c30 = metric_utils.continuity(data, embeddings)

    return t30, c30, embeddings


def pca_metric(data):
    data = data_utils.to_numpy(data)
    pca = PCA(n_components=2)
    embeddings = pca.fit_transform(data)

    t30 = metric_utils.trustworthiness(data, embeddings)
    c30 = metric_utils.continuity(data, embeddings)

    return t30, c30, embeddings


def umap_metric(data):
    data = data_utils.to_numpy(data)
    umap = UMAP(n_components=2)
    embeddings = umap.fit_transform(data)

    t30 = metric_utils.trustworthiness(data, embeddings)
    c30 = metric_utils.continuity(data, embeddings)

    return t30, c30, embeddings


def ours_metric(data, model):
    data = data.to(device=default_device, dtype=default_dtype)
    model = model.to(device=default_device, dtype=default_dtype)
    embeddings = model.encode(data)

    data = data.cpu().detach().numpy()
    embeddings = embeddings.cpu().detach().numpy()

    t30 = metric_utils.trustworthiness(data, embeddings)
    c30 = metric_utils.continuity(data, embeddings)

    return t30, c30, embeddings


def generate_log_dir(*dirs):
    # 获取当前工作目录的绝对路径
    current_dir = os.getcwd()

    # 生成唯一时间戳（添加随机数以增强唯一性）
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    unique_id = random.randint(1000, 9999)  # 添加随机数
    unique_timestamp = f"{timestamp}_{unique_id}"

    # 生成日志目录路径
    log_dir = os.path.join(current_dir, "log", "metric",
                           *dirs, unique_timestamp)

    # 创建目录（如果不存在）
    os.makedirs(log_dir, exist_ok=True)

    return log_dir


def main():
    log_dir = generate_log_dir()
    data_utils.setup_seed(43)

    # 加载数据
    bio_data = gene_loader.get_gene_data()
    mnist_data, mnist_label = mnist_loader.get_mnist_train_data(6000)
    cifar_data, cifar_label = cifar_loader.get_cifar_train_data()

    # 加载模型
    bio_model = model_utils.prepare_gene_mlp_model(
        "/home/log/train/GENE/MLP/sample/20250313_063747_4943_good", "final")
    mnist_model = model_utils.prepare_mnist_mlp_model(
        "/home/log/train/MNIST/MLP/sample/20250313_051407_1003_good", "final")
    cifar_model = model_utils.prepare_cifar_mlp_model(
        "/home/log/train/CIFAR/MLP/sample/20250313_065744_6054_good", "final")

    # 数据字典，包含数据、标签和模型
    data_dict = {
        "Gene": (bio_data, None, bio_model),
        "Mnist": (mnist_data, mnist_label, mnist_model),
        "Cifar": (cifar_data, cifar_label, cifar_model)
    }

    # 方法字典，包含方法名称和对应的函数
    methods_dict = {
        "t-SNE": tsne_metric,
        "MDS": mds_metric,
        "PCA": pca_metric,
        "UMAP": umap_metric,
        "Ours": ours_metric
    }

    # 初始化结果字典
    results_dict = {key: {"Method": [], "T30": [], "C30": []}
                    for key in data_dict.keys()}

    for key, (data, labels, model) in data_dict.items():
        for method, method_func in methods_dict.items():
            # 调用方法函数
            if method == "Ours":
                t30, c30, embeddings = method_func(data, model)
            else:
                t30, c30, embeddings = method_func(data)

            # 保存 metric 结果
            results_dict[key]["Method"].append(method)
            results_dict[key]["T30"].append(t30)
            results_dict[key]["C30"].append(c30)

            # 保存 embeddings 为 .npy 文件
            embeddings_filename = os.path.join(
                log_dir, f"{key}_{method}_embeddings.npy")
            np.save(embeddings_filename, embeddings)
            print(
                f"Embeddings for {key} ({method}) saved to {embeddings_filename}")

            # 可视化并保存图像
            vis_utils.plot_embeddings(embeddings, labels)
            plt.savefig(os.path.join(log_dir, f"{key}_{method}_embedding.png"))
            plt.close()
            print(f"Visualization for {key} ({method}) saved to {log_dir}")

        # 将结果保存为 CSV 文件
        df = pd.DataFrame(results_dict[key])
        csv_filename = os.path.join(log_dir, f"{key}_metrics.csv")
        df.to_csv(csv_filename, index=False)
        print(f"Metrics for {key} saved to {csv_filename}")


def dcm():
    log_dir = generate_log_dir()
    data_utils.setup_seed(43)

    # 加载数据
    bio_data = gene_loader.get_gene_data()
    mnist_data, mnist_label = mnist_loader.get_mnist_train_data(6000)
    cifar_data, cifar_label = cifar_loader.get_cifar_train_data()

    # 加载模型
    bio_model = None
    mnist_model = None
    cifar_model = None

    # 数据字典，包含数据、标签和模型
    data_dict = {
        "Gene": (bio_data, None, bio_model),
        "Mnist": (mnist_data, mnist_label, mnist_model),
        "Cifar": (cifar_data, cifar_label, cifar_model)
    }

    # 方法字典，包含方法名称和对应的函数
    methods_dict = {
        "DCM": dcm_metric
    }

    # 初始化结果字典
    results_dict = {key: {"Method": [], "T30": [], "C30": []}
                    for key in data_dict.keys()}

    for key, (data, labels, model) in data_dict.items():
        for method, method_func in methods_dict.items():
            # 调用方法函数
            if method == "Ours":
                t30, c30, embeddings = method_func(data, model)
            else:
                t30, c30, embeddings = method_func(data)

            # 保存 metric 结果
            results_dict[key]["Method"].append(method)
            results_dict[key]["T30"].append(t30)
            results_dict[key]["C30"].append(c30)

            # 保存 embeddings 为 .npy 文件
            embeddings_filename = os.path.join(
                log_dir, f"{key}_{method}_embeddings.npy")
            np.save(embeddings_filename, embeddings)
            print(
                f"Embeddings for {key} ({method}) saved to {embeddings_filename}")

            # 可视化并保存图像
            vis_utils.plot_embeddings(embeddings, labels)
            plt.savefig(os.path.join(log_dir, f"{key}_{method}_embedding.png"))
            plt.close()
            print(f"Visualization for {key} ({method}) saved to {log_dir}")

        # 将结果保存为 CSV 文件
        df = pd.DataFrame(results_dict[key])
        csv_filename = os.path.join(log_dir, f"{key}_metrics.csv")
        df.to_csv(csv_filename, index=False)
        print(f"Metrics for {key} saved to {csv_filename}")

if __name__ == "__main__":
    dcm()