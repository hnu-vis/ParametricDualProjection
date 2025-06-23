import json
import os
from tqdm import tqdm

import matplotlib.pyplot as plt
import pandas as pd
from pynndescent import NNDescent
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from model.base_embedder import BaseEmbedder
import loss as loss_functions
from loss.nce_loss import ntxent_loss, refined_ntxent_loss


default_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
default_dtype = torch.float64


class BaseTrainer:
    def __init__(
        self,
        data: torch.Tensor,
        model: BaseEmbedder,
        log_dir: str,
        labels: torch.Tensor = None,
        device: torch.device = default_device,
        dtype: torch.dtype = default_dtype
    ):
        self.data = data.to(device=device, dtype=dtype)
        self.model = model.to(device=device, dtype=dtype)
        self.labels = labels
        self.config = model.config
        self.device = device
        self.dtype = dtype

        self.input_dim = model.config.get_input_dim()
        self.output_dim = model.config.get_output_dim()
        self.latent_dim = model.config.get_latent_dim()

        self.warmup_epochs = model.config.warmup_epochs
        self.main_epochs = model.config.main_epochs
        self.stable_epochs = model.config.stable_epochs

        self.embedding_weight = 1.0
        self.latent_distribution_weight = 1.0
        self.noise_distribution_weight = 1.0
        self.reconstruction_weight = 1.0

        self.batch_size = self.config.batch_size
        self.epochs = self.config.epochs
        self.lr = self.config.lr

        self.save_interval = self.epochs // 10

        self.curr_batch_id = 0
        self.curr_epoch = 0

        self.dataloader = DataLoader(
            self.data, batch_size=self.batch_size, shuffle=True)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=self.epochs)

        self.embedding_feature = None
        self.loss = None

        self.log_dir = log_dir

    def embedding_loss(self, input_data, embedding_feature):
        embedding_loss = loss_functions.tsne_loss(
            input_data, embedding_feature)
        return embedding_loss

    def loss_fn(self, input_data, embedding_feature, latent_feature, gaussian_noise, reconstructed_data):
        embedding_loss = self.embedding_loss(input_data, embedding_feature)

        latent_distribution_loss = loss_functions.gaussian_loss(latent_feature)
        noise_distribution_loss = loss_functions.gaussian_loss(gaussian_noise)

        mse_loss = nn.MSELoss()
        reconstruction_loss = mse_loss(input_data, reconstructed_data)

        total_loss = (
            self.embedding_weight * embedding_loss
            + self.latent_distribution_weight * latent_distribution_loss
            + self.noise_distribution_weight * noise_distribution_loss
            + self.reconstruction_weight * reconstruction_loss
        )

        return (
            total_loss,
            embedding_loss,
            latent_distribution_loss,
            noise_distribution_loss,
            reconstruction_loss,
        )

    def before_train(self):
        self.curr_epoch = 0
    
    def after_train(self):
        pass

    def before_epoch(self):
        pass

    def after_epoch(self):
        pass

    def during_epoch(self):
        pass

    def train(self):
        self.before_train()

        self.loss = {
            "total loss": [],
            "embedding loss": [],
            "latent distribution loss": [],
            "noise distribution loss": [],
            "reconstruction loss": []
        }

        for i, epoch in tqdm(enumerate(range(self.epochs)), desc="Training Progress"):
            self.curr_epoch = i

            epoch_total_loss = 0.0
            epoch_embedding_loss = 0.0
            epoch_latent_distribution_loss = 0.0
            epoch_noise_distribution_loss = 0.0
            epoch_reconstruction_loss = 0.0

            for batch_id, batch in tqdm(enumerate(self.dataloader), desc=f"Processing Epoch {epoch+1}"):
                self.curr_batch_id = batch_id
                self.optimizer.zero_grad()

                # 前向传播
                embedding_feature, latent_feature, gaussian_noise = self.model.encode_for_train(
                    batch)
                reconstructed_data = self.model.decode_for_train(
                    embedding_feature)

                # 计算损失
                total_loss, embedding_loss, latent_distribution_loss, noise_distribution_loss, reconstruction_loss = self.loss_fn(
                    batch, embedding_feature, latent_feature, gaussian_noise, reconstructed_data
                )

                # 反向传播和优化
                total_loss.backward()
                self.optimizer.step()

                # 累加损失
                epoch_total_loss += total_loss.item()
                epoch_embedding_loss += embedding_loss.item()
                epoch_latent_distribution_loss += latent_distribution_loss.item()
                epoch_noise_distribution_loss += noise_distribution_loss.item()
                epoch_reconstruction_loss += reconstruction_loss.item()

            self.scheduler.step()

            # 计算平均损失
            avg_total_loss = epoch_total_loss / len(self.dataloader)
            avg_embedding_loss = epoch_embedding_loss / len(self.dataloader)
            avg_latent_distribution_loss = epoch_latent_distribution_loss / \
                len(self.dataloader)
            avg_noise_distribution_loss = epoch_noise_distribution_loss / \
                len(self.dataloader)
            avg_reconstruction_loss = epoch_reconstruction_loss / \
                len(self.dataloader)

            # 记录损失
            self.loss["total loss"].append(avg_total_loss)
            self.loss["embedding loss"].append(avg_embedding_loss)
            self.loss["latent distribution loss"].append(
                avg_latent_distribution_loss)
            self.loss["noise distribution loss"].append(
                avg_noise_distribution_loss)
            self.loss["reconstruction loss"].append(avg_reconstruction_loss)

            # 打印损失
            print(
                f"Epoch {epoch+1}/{self.epochs}, "
                f"Total Loss: {avg_total_loss:.4f}, "
                f"Embedding Loss: {avg_embedding_loss:.4f}, "
                f"Latent Distribution Loss: {avg_latent_distribution_loss:.4f}, "
                f"Noise Distribution Loss: {avg_noise_distribution_loss:.4f}, "
                f"Reconstruction Loss: {avg_reconstruction_loss:.4f}"
            )

            if (epoch + 1) % self.save_interval == 0:
                self.embedding_feature = self.model.encode(self.data)

                self.plot(epoch+1)
                self.save(epoch+1)

        self.embedding_feature = self.model.encode(self.data)
        self.plot_final()
        self.save_final()

    def plot_loss(self):
        loss_df = pd.DataFrame(self.loss)

        loss_df.reset_index(inplace=True)

        loss_df = pd.melt(loss_df, id_vars='index',
                          var_name='Variable', value_name='Value')

        plt.figure(figsize=(10, 8))

        sns.set_theme(style="darkgrid")
        palette = sns.color_palette("husl", 5)

        sns.lineplot(
            data=loss_df,
            x='index',
            y='Value',
            hue='Variable',
            style='Variable',
            palette=palette,
            # markers=True,  # 可选：为每条线添加标记点
            dashes=True
        )

        plt.title("Training Loss", fontsize=12)
        plt.xlabel("Epochs", fontsize=12)
        plt.ylabel("Loss", fontsize=12)

        plt.grid(True)

        plt.legend(title="Loss Types", fontsize=12)

        plot_path = os.path.join(self.log_dir, f"loss.png")
        plt.savefig(plot_path)
        plt.close()

    def plot_result(self, i):
        # 将嵌入特征和标签转换为 NumPy 数组
        result = self.embedding_feature.cpu().detach().numpy()  # 嵌入特征
        labels = self.labels.cpu().detach().numpy(
        ) if self.labels is not None else None  # 标签

        # 创建一个 DataFrame 用于 seaborn 绘图
        df = pd.DataFrame(
            result, columns=[f"Dim_{i+1}" for i in range(result.shape[1])])

        # 如果标签存在，添加到 DataFrame 中
        if labels is not None:
            df['Label'] = labels

        # 设置 seaborn 主题
        sns.set_theme(style="darkgrid")
        palette = sns.color_palette("tab10")

        # 绘制散点图
        plt.figure(figsize=(10, 8))
        if labels is not None:
            # 如果有标签，根据标签分组绘制
            sns.scatterplot(
                x="Dim_1", y="Dim_2", hue="Label", palette=palette, data=df
            )
            plt.legend(title="Label")
        else:
            sns.scatterplot(
                x="Dim_1", y="Dim_2", data=df
            )

        plt.title("Embedding Feature", fontsize=12)
        plt.xlabel("Dim 1", fontsize=12)
        plt.ylabel("Dim 2", fontsize=12)

        plot_path = os.path.join(self.log_dir, f"embedding_{i}.png")
        plt.savefig(plot_path)
        plt.close()

    def plot(self, i):
        self.plot_result(i)

    def plot_final(self):
        self.plot_loss()
        self.plot_result("final")

    def save_model(self, i):
        model_path = os.path.join(self.log_dir, f"model_weight_{i}.pth")
        torch.save(self.model.state_dict(), model_path)

    def save_loss(self):
        # 将损失字典转换为 pandas DataFrame
        loss_df = pd.DataFrame(self.loss)

        # 定义保存路径
        loss_csv_path = os.path.join(self.log_dir, f"loss.csv")
        loss_df.to_csv(loss_csv_path, index=False)

    def save_config(self):
        # 将配置转换为字典
        config_dict = self.config.to_dict()

        # 定义保存路径
        config_path = os.path.join(self.log_dir, "config.json")

        # 保存为 JSON 文件
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=4)

        print(f"Config saved to {config_path}")

    def save(self, i):
        self.save_model(i)

    def save_final(self):
        self.save_config()
        self.save_model("final")
        self.save_loss()


class VaeTrainer(BaseTrainer):
    def __init__(self, data, model, log_dir, labels = None, device = default_device, dtype = default_dtype):
        super().__init__(data, model, log_dir, labels, device, dtype)


class ConstractiveTrainer(BaseTrainer):
    def __init__(self, data, model, log_dir, labels=None, device=default_device, dtype=default_dtype):
        super().__init__(data, model, log_dir, labels, device, dtype)

        self.k = 15
        self.temparature = 0.15

    def knn_euclidean(self, points, k):
        # 将 batch 转换为 NumPy 数组
        batch_np = points.cpu().detach().numpy()

        # 使用 NNDescent 找到每个样本的 k+1 个最近邻（包含自身）
        nnd = NNDescent(batch_np, metric="euclidean", n_neighbors=k + 1)
        indices, _ = nnd.query(batch_np)

        # 排除自身（第一个最近邻）
        indices = indices[:, 1:]  # 去掉第一列（自身）

        return indices

    def get_positive_indices_batch(self, batch, k=15):
        """
        给定一个 batch 的数据，返回正样本对的索引矩阵。

        参数:
            batch (torch.Tensor): 当前 batch 的数据，形状为 (batch_size, num_features)。
            k (int): kNN 的 k 值，默认为 15。

        返回:
            pos_indices (torch.Tensor): 正样本对的索引矩阵，形状为 (batch_size, 1)。
        """
        device = batch.device
        dtype = batch.dtype
        batch_size = batch.shape[0]

        # 计算 kNN
        distances = torch.cdist(batch, batch)  # 计算 batch 内部所有样本对之间的欧氏距离
        _, indices = torch.topk(distances, k + 1, largest=False)  # 找到每个样本的 k+1 个最近邻居

        # 排除自身
        neighbors = indices[:, 1:]  # 排除自身，形状为 (batch_size, k)

        # 计算条件概率 p_j|i
        rho_i = distances[:, 1].unsqueeze(1)  # 每个点到其最近邻居的距离，形状为 (batch_size, 1)
        sigma_i = torch.sum(torch.exp(-(distances[:, 1:] - rho_i)), dim=1, keepdim=True)  # 归一化项，形状为 (batch_size, 1)
        p_j_given_i = torch.exp(-(distances[:, 1:] - rho_i) / sigma_i)  # 条件概率 p_j|i，形状为 (batch_size, k)

        # 将 p_j_given_i 扩展为 (batch_size, batch_size) 的矩阵
        p_j_given_i_full = torch.zeros(batch_size, batch_size, device=device, dtype=dtype)
        for i in range(batch_size):
            p_j_given_i_full[i, neighbors[i]] = p_j_given_i[i, :k]  # 确保形状一致

        # 计算对称概率 p_ij
        p_ij = p_j_given_i_full + p_j_given_i_full.transpose(0, 1) - p_j_given_i_full * p_j_given_i_full.transpose(0, 1)

        # 归一化概率
        p_ij = p_ij / p_ij.sum(dim=1, keepdim=True)

        # 根据概率分布采样正样本对
        pos_indices = torch.multinomial(p_ij, 1)  # 从概率分布中采样，形状为 (batch_size, 1)

        return pos_indices

    def get_embeddings_for_loss(self, embeddings, pos_indices):
        """
        从嵌入表示中提取正样本的嵌入表示。

        参数:
            embeddings (torch.Tensor): 当前 batch 的嵌入表示，形状为 (batch_size, latent_dim)。
            pos_indices (torch.Tensor): 正样本对的索引矩阵，形状为 (batch_size, 1)。

        返回:
            x_sim_embedding (torch.Tensor): 正样本的嵌入表示，形状为 (batch_size, latent_dim)。
        """
        # 使用 pos_indices 提取正样本的嵌入表示
        x_sim_embedding = embeddings[pos_indices.squeeze()]  # 去掉多余的维度

        return x_sim_embedding

    def batch_logits(self, x_embeddings, x_sim_embeddings, a=1.0, b=1.0):
        """
        计算正样本对和负样本对的 logits。

        参数:
            x_embeddings (torch.Tensor): 当前 batch 的嵌入表示，形状为 (batch_size, latent_dim)。
            x_sim_embeddings (torch.Tensor): 正样本的嵌入表示，形状为 (batch_size, latent_dim)。
            a (float): 公式中的超参数 a，默认为 1.0。
            b (float): 公式中的超参数 b，默认为 1.0。

        返回:
            logits (torch.Tensor): 正负样本对的相似度矩阵，形状为 (batch_size, 1 + batch_size - 1)。
        """
        batch_size = x_embeddings.shape[0]

        # 拼接当前 batch 的嵌入表示和正样本的嵌入表示
        # 形状为 (2 * batch_size, latent_dim)
        all_embeddings = torch.cat([x_embeddings, x_sim_embeddings], dim=0)

        # 计算所有样本对之间的欧氏距离的平方
        # 形状为 (2 * batch_size, 2 * batch_size)
        pairwise_distances = torch.cdist(all_embeddings, all_embeddings, p=2)
        pairwise_distances_sq = pairwise_distances ** 2  # 欧氏距离的平方

        # 计算相似度 q_{ij}
        # 形状为 (2 * batch_size, 2 * batch_size)
        q_ij = 1 / (1 + a * (pairwise_distances_sq ** b))

        # 提取正样本对的相似度
        l_pos = torch.diag(q_ij, batch_size)  # 提取对角线偏移 batch_size 的元素
        r_pos = torch.diag(q_ij, -batch_size)  # 提取对角线偏移 -batch_size 的元素
        positives = torch.cat([l_pos, r_pos]).view(
            2 * batch_size, 1)  # 形状为 (2 * batch_size, 1)

        # 提取负样本对的相似度
        mask = torch.ones_like(q_ij, dtype=torch.bool)  # 创建全为 True 的掩码
        mask.fill_diagonal_(False)  # 排除自相似对
        mask[torch.arange(batch_size), torch.arange(
            batch_size, 2 * batch_size)] = False  # 排除正样本对
        mask[torch.arange(batch_size, 2 * batch_size),
             torch.arange(batch_size)] = False  # 排除正样本对

        # 形状为 (2 * batch_size, 2 * batch_size - 2)
        negatives = q_ij[mask].view(2 * batch_size, -1)

        # 拼接正负样本对的 logits
        # 形状为 (2 * batch_size, 1 + 2 * batch_size - 2)
        logits = torch.cat([positives, negatives], dim=1)

        return logits

    def embedding_loss(self, input_data, embedding_feature):
        """
        计算嵌入损失，支持多阶段训练策略。

        参数:
            input_data: 输入数据。
            embedding_feature: 嵌入特征，形状为 (batch_size, latent_dim)。

        返回:
            loss: 计算得到的损失值。
        """
        # 获取正样本对的索引
        pos_indices = self.get_positive_indices_batch(input_data, self.k)

        # 获取正样本对的嵌入特征
        embedding_sim_feature = self.get_embeddings_for_loss(
            embedding_feature, pos_indices)

        # 计算正负样本对的 logits
        logits = self.batch_logits(
            embedding_feature, embedding_sim_feature, a=1.8956058664239412, b=0.8006378441176886
        )

        # # 提取正样本对和负样本对的相似度
        # q_pos = logits[:, 0]  # 正样本对的相似度，形状为 (2 * batch_size,)
        # # 负样本对的相似度，形状为 (2 * batch_size, 2 * batch_size - 2)
        # q_neg = logits[:, 1:]

        # 根据当前训练阶段选择损失函数
        if self.curr_epoch < self.warmup_epochs:
            # 预热阶段：使用 NT-Xent 损失
            loss = ntxent_loss(logits, tau=self.temparature, item_weights=self.embedding_weight)
            # loss = ntxent_loss(q_pos, q_neg, tau=self.temparature)
        elif self.curr_epoch < self.warmup_epochs + self.main_epochs:
            # 主训练阶段：使用改进的对比损失
            loss = refined_ntxent_loss(
                logits, tau=self.temparature, item_weights=self.embedding_weight
            )
            # loss = refined_ntxent_loss(
            #     q_pos, q_neg, tau=self.temparature, alpha_i=5.0, mu=0.11, sigma=0.13, eta=-40.0
            # )
        else:
            # 稳定阶段：使用 NT-Xent 损失
            loss = ntxent_loss(logits, tau=self.temparature, item_weights=self.embedding_weight)

        return loss


class SampleTrainer(ConstractiveTrainer):
    def __init__(self, data, model, log_dir, label=None, device=default_device, dtype=default_dtype):
        super().__init__(data, model, log_dir, label, device, dtype)

        self.embedding_weight = 0.8
        self.latent_distribution_weight = 1e-3
        self.noise_distribution_weight = 1e-3
        self.reconstruction_weight = 0.2


class FeatureTrainer(ConstractiveTrainer):
    def __init__(self, data, model, log_dir, labels=None, device=default_device, dtype=default_dtype):
        super().__init__(data, model, log_dir, labels, device, dtype)

        self.embedding_weight = 0.8
        self.latent_distribution_weight = 1e-3
        self.noise_distribution_weight = 1e-3
        self.reconstruction_weight = 0.2
