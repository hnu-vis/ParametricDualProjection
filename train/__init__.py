import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from pynndescent import NNDescent
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import loss
from loss.nce_loss import ntxent_loss, refined_ntxent_loss
from model import BaseEmbedder
from utils import data_utils, vis_utils


class BaseTrainer:
    def __init__(
        self,
        dataset: Dataset,
        model: BaseEmbedder,
        log_dir: str,
        device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"),
        dtype: torch.dtype = torch.float32
    ):
        # 基础初始化
        self.dataset = dataset  # 数据集
        self.data, self.labels = dataset[:]
        self.model = model.to(device=device, dtype=dtype)
        self.log_dir = log_dir
        self.device = device
        self.dtype = dtype

        # 配置初始化
        self.setup_config()
        self.setup_loss_weights()
        self.setup_training_params()

        # 训练组件初始化
        self.setup_dataloader()
        self.setup_optimizer()
        self.setup_scheduler()

        # 训练状态
        self.curr_epoch = 0
        self.curr_batch_id = 0
        self.embedding_feature = None
        self.loss_history = {
            "total": [],
            "embedding": [],
            "auxiliary": [],
            "latent": [],
            "reconstruction": []
        }

    # 初始化相关方法
    def setup_config(self):
        """从模型配置中初始化参数"""
        self.config = self.model.config
        self.input_dim = self.config.get_input_dim()
        self.output_dim = self.config.get_output_dim()
        self.latent_dim = self.config.get_latent_dim()

        self.warmup_epochs = self.config.warmup_epochs
        self.main_epochs = self.config.main_epochs
        self.stable_epochs = self.config.stable_epochs

    def setup_loss_weights(self):
        self.embedding_weight = 1.0
        self.auxiliary_weight = 1.0
        self.latent_weight = 1.0
        self.reconstruction_weight = 1.0

    def setup_training_params(self):
        self.batch_size = self.config.batch_size
        self.epochs = self.config.epochs
        self.lr = self.config.lr
        self.save_interval = self.epochs // 10

    def setup_dataloader(self):
        self.dataloader = DataLoader(
            self.dataset,  # 使用 dataset
            batch_size=self.batch_size,
            shuffle=True
        )

    def setup_optimizer(self):
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.lr
        )

    def setup_scheduler(self):
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=self.epochs
        )

    # 训练流程方法
    def train(self):
        self.before_training()

        progress_bar = tqdm(range(self.epochs), desc="Training")
        for self.curr_epoch in progress_bar:
            self.before_epoch()
            epoch_metrics = self.run_epoch()
            self.after_epoch(epoch_metrics)

            progress_bar.set_postfix({
                "Tot.": f"{epoch_metrics['total']:.4f}",
                "Emb.": f"{epoch_metrics['embedding']:.4f}",
                "Aux.": f"{epoch_metrics['auxiliary']:.4f}",
                "Lat.": f"{epoch_metrics['latent']:.4f}",
                "Rec.": f"{epoch_metrics['reconstruction']:.4f}"
            })

        self.after_training()

    def run_epoch(self):
        epoch_metrics = {
            "total": 0.0,
            "embedding": 0.0,
            "auxiliary": 0.0,
            "latent": 0.0,
            "reconstruction": 0.0
        }

        for batch_id, (data, labels) in enumerate(self.dataloader):  # 解包数据和标签
            self.curr_batch_id = batch_id
            data = data.to(self.device, dtype=self.dtype)  # 将数据移动到设备

            batch_metrics = self.training_step(data)  # 传递数据和标签

            for key in epoch_metrics:
                epoch_metrics[key] += batch_metrics[key]

        # 计算平均损失
        for key in epoch_metrics:
            epoch_metrics[key] /= len(self.dataloader)

        return epoch_metrics

    def training_step(self, data: torch.Tensor):
        self.optimizer.zero_grad()

        # 前向传播
        embedding_features, auxiliary_features, latent_features, reconstructed_data = self.model.forward(
            data)

        # 损失计算
        losses = self.compute_losses(
            data, embedding_features, auxiliary_features, latent_features, reconstructed_data
        )

        # 反向传播
        losses["total"].backward()
        self.optimizer.step()

        # 返回标量值
        return {k: v.item() for k, v in losses.items()}

    def compute_losses(self, data, embedding_features, auxiliary_features, latent_features, reconstructed_data):
        return {
            "total": self.total_loss(data, embedding_features, auxiliary_features, latent_features, reconstructed_data),
            "embedding": self.embedding_loss(data, embedding_features),
            "auxiliary": self.auxiliary_loss(data, auxiliary_features),
            "latent": self.latent_loss(latent_features),
            "reconstruction": self.reconstruction_loss(data, reconstructed_data)
        }

    def total_loss(self, data, embedding_features, auxiliary_features, latent_features, reconstructed_data):
        return (
            self.embedding_weight * self.embedding_loss(data, embedding_features) +
            self.auxiliary_weight * self.auxiliary_loss(data, auxiliary_features) +
            self.latent_weight * self.latent_loss(latent_features) +
            self.reconstruction_weight *
            self.reconstruction_loss(data, reconstructed_data)
        )

    def embedding_loss(self, data, embedding_features):
        return loss.mds_loss(data, embedding_features)

    def auxiliary_loss(self, data, auxiliary_features):
        return loss.mds_loss(data, auxiliary_features)

    def latent_loss(self, latent_features):
        return loss.gaussian_loss(latent_features)

    def reconstruction_loss(self, data, reconstructed_data):
        return nn.MSELoss()(data, reconstructed_data)

    # 生命周期钩子方法
    def before_training(self):
        os.makedirs(self.log_dir, exist_ok=True)
        self.model.train()

    def after_training(self):
        self.save_final()
        self.model.eval()

    def before_epoch(self):
        self.update_loss_weights()
        self.model.train()

    def after_epoch(self, metrics):
        # 记录损失历史
        for key in self.loss_history:
            self.loss_history[key].append(metrics[key])

        # 更新学习率
        self.scheduler.step()

        # 定期保存
        if (self.curr_epoch + 1) % self.save_interval == 0:
            self.save_checkpoint()

    # 权重更新机制
    def update_loss_weights(self):
        pass

    # 保存和可视化方法
    def save_checkpoint(self):
        data = self.data.to(self.device, dtype=self.dtype)
        self.embedding_feature = self.model.encode(data)
        self.plot_current()
        self.save_model(f"{self.curr_epoch+1}")

    def save_final(self):
        data = self.data.to(self.device, dtype=self.dtype)
        self.embedding_feature = self.model.encode(data)
        self.plot_final()
        self.save_model("final")
        self.save_loss_history()
        self.save_config()

    def save_model(self, suffix):
        path = os.path.join(self.log_dir, f"model_{suffix}.pth")
        torch.save(self.model.state_dict(), path)

    def save_loss_history(self):
        df = pd.DataFrame(self.loss_history)
        df.to_csv(os.path.join(self.log_dir, "loss_history.csv"), index=False)

    def save_config(self):
        config = self.config.to_dict()

        with open(os.path.join(self.log_dir, "config.json"), "w") as f:
            json.dump(config, f, indent=4)

    def plot_current(self):
        self.plot_embedding(self.curr_epoch + 1)
        self.plot_loss_curve()

    def plot_final(self):
        self.plot_embedding("final")
        self.plot_loss_curve()

    def plot_embedding(self, suffix):
        embeddings = data_utils.to_numpy(self.embedding_feature)
        labels = data_utils.to_numpy(self.labels)
        vis_utils.plot_embeddings(
            embeddings,
            labels,
            discrete=True,
            save_path=os.path.join(self.log_dir, f"embedding_{suffix}.png")
        )

    def plot_loss_curve(self):
        vis_utils.plot_loss_curve(
            self.loss_history,
            save_path=os.path.join(self.log_dir, f"loss_curve.png")
        )


class ConstractiveTrainer(BaseTrainer):
    def __init__(
        self,
        dataset,
        model,
        log_dir,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        dtype=torch.float32
    ):
        super().__init__(dataset, model, log_dir, device, dtype)
        self.k = 10
        self.temparature = 0.20

    def compute_dist(self, X, Y):
        return torch.cdist(X, Y)

    def get_positive_indices_batch(self, batch, labels=None, k=15):
        device = batch.device
        dtype = batch.dtype
        batch_size = batch.shape[0]

        if labels is not None:
            # 假设 labels 是 (batch_size,) 的张量
            pos_indices = torch.zeros(
                batch_size, 1, device=device, dtype=torch.long)

            # 计算每个样本的标签与其他样本标签的匹配矩阵
            # (batch_size, batch_size)
            label_matches = (labels.unsqueeze(1) == labels.unsqueeze(0))

            # 排除自身（对角线设为 False）
            label_matches.diagonal().fill_(False)

            # 获取每个样本的正样本候选索引
            # 返回 (row_indices, col_indices)
            same_label_indices = torch.where(label_matches)
            row_indices, col_indices = same_label_indices

            # 计算每个样本的正样本数量
            counts = torch.bincount(
                row_indices, minlength=batch_size)  # (batch_size,)
            valid_rows = counts > 0  # 标记有正样本的行

            # 为每个样本随机选择一个正样本
            if valid_rows.any():
                # 计算每组的起始位置
                cum_counts = torch.cat(
                    [torch.tensor([0], device=device), counts.cumsum(0)[:-1]])
                # 为每个有效行生成随机偏移，范围根据 counts[valid_rows] 单独确定
                offsets = torch.zeros_like(
                    row_indices, dtype=torch.long, device=device)
                valid_row_counts = counts[valid_rows]  # 每个有效行的正样本数量
                valid_row_indices = torch.arange(len(valid_rows), device=device)[
                    valid_rows]  # 有效行的原始索引
                # 为每个样本生成符合其正样本数量的随机偏移
                group_offsets = torch.randint(
                    0, valid_row_counts.max(), (valid_rows.sum(),), device=device)
                group_offsets = group_offsets % valid_row_counts  # 限制偏移不超过每行的正样本数量
                offsets[cum_counts[valid_rows]] = group_offsets  # 将偏移赋值到对应位置
                sample_indices = cum_counts[valid_rows] + \
                    offsets[cum_counts[valid_rows]]
                pos_indices[valid_rows] = col_indices[sample_indices].unsqueeze(
                    1)

            # 对于没有正样本的样本，用自身填充
            pos_indices[~valid_rows] = torch.arange(
                batch_size, device=device, dtype=torch.long)[~valid_rows].unsqueeze(1)

            return pos_indices
        else:
            # 计算 kNN
            distances = torch.cdist(batch, batch)  # 计算 batch 内部所有样本对之间的欧氏距离
            _, indices = torch.topk(
                distances, k + 1, largest=False)  # 找到每个样本的 k+1 个最近邻居

            # 排除自身
            neighbors = indices[:, 1:]  # 排除自身，形状为 (batch_size, k)

            # 计算条件概率 p_j|i
            # 每个点到其最近邻居的距离，形状为 (batch_size, 1)
            rho_i = distances[:, 1].unsqueeze(1)
            # 归一化项，形状为 (batch_size, 1)
            sigma_i = torch.sum(
                torch.exp(-(distances[:, 1:] - rho_i)), dim=1, keepdim=True)
            # 条件概率 p_j|i，形状为 (batch_size, k)
            p_j_given_i = torch.exp(-(distances[:, 1:] - rho_i) / sigma_i)

            # 将 p_j_given_i 扩展为 (batch_size, batch_size) 的矩阵
            p_j_given_i_full = torch.zeros(
                batch_size, batch_size, device=device, dtype=dtype)
            for i in range(batch_size):
                p_j_given_i_full[i, neighbors[i]
                                 ] = p_j_given_i[i, :k]  # 确保形状一致

            # 计算对称概率 p_ij
            p_ij = p_j_given_i_full + \
                p_j_given_i_full.transpose(
                    0, 1) - p_j_given_i_full * p_j_given_i_full.transpose(0, 1)

            # 归一化概率
            p_ij = p_ij / p_ij.sum(dim=1, keepdim=True)

            # 根据概率分布采样正样本对
            # 从概率分布中采样，形状为 (batch_size, 1)
            pos_indices = torch.multinomial(p_ij, 1)
            return pos_indices

    def get_embeddings_for_loss(self, embeddings, pos_indices):
        # pos_indices 的形状为 (batch_size, 1)
        # (batch_size, latent_dim)
        x_sim_embedding = embeddings[pos_indices.squeeze()]
        return x_sim_embedding

    def batch_logits(self, x_embeddings, x_sim_embeddings, a=1.0, b=1.0):
        batch_size = x_embeddings.shape[0]
        # (2 * batch_size, latent_dim)
        all_embeddings = torch.cat([x_embeddings, x_sim_embeddings], dim=0)

        # 计算距离矩阵
        # (2 * batch_size, 2 * batch_size)
        pairwise_distances = torch.cdist(all_embeddings, all_embeddings, p=2)
        pairwise_distances_sq = pairwise_distances ** 2
        q_ij = 1 / (1 + a * (pairwise_distances_sq ** b))

        # 提取正样本相似度
        l_pos = torch.diag(q_ij, batch_size)  # (batch_size,)
        r_pos = torch.diag(q_ij, -batch_size)  # (batch_size,)
        positives = torch.cat([l_pos, r_pos]).view(
            2 * batch_size, 1)  # (2 * batch_size, 1)

        # 提取负样本相似度
        mask = torch.ones_like(q_ij, dtype=torch.bool)
        mask.fill_diagonal_(False)
        mask[torch.arange(batch_size), torch.arange(
            batch_size, 2 * batch_size)] = False
        mask[torch.arange(batch_size, 2 * batch_size),
             torch.arange(batch_size)] = False
        # (2 * batch_size, 2 * batch_size - 2)
        negatives = q_ij[mask].view(2 * batch_size, -1)

        # 拼接 logits
        # (2 * batch_size, 1 + 2 * batch_size - 2)
        logits = torch.cat([positives, negatives], dim=1)
        return logits

    def embedding_loss(self, data, embedding_features):
        data = data.view(data.size(0), -1)
        pos_indices = self.get_positive_indices_batch(
            data, None, k=self.k)  # 默认 labels=None
        embedding_sim_features = self.get_embeddings_for_loss(
            embedding_features, pos_indices)
        logits = self.batch_logits(
            embedding_features, embedding_sim_features, a=1.8956058664239412, b=0.8006378441176886)

        if self.curr_epoch < self.warmup_epochs:
            loss = ntxent_loss(logits, tau=self.temparature,
                               item_weights=self.embedding_weight)
        elif self.curr_epoch < self.warmup_epochs + self.main_epochs:
            loss = refined_ntxent_loss(
                logits, tau=self.temparature, item_weights=self.embedding_weight)
        else:
            loss = ntxent_loss(logits, tau=self.temparature,
                               item_weights=self.embedding_weight)
        return loss

    def auxiliary_loss(self, data, auxiliary_features):
        data = data.view(data.size(0), -1)
        pos_indices = self.get_positive_indices_batch(
            data, None, k=self.k)  # 默认 labels=None
        embedding_sim_features = self.get_embeddings_for_loss(
            auxiliary_features, pos_indices)
        logits = self.batch_logits(
            auxiliary_features, embedding_sim_features, a=1.8956058664239412, b=0.8006378441176886)

        if self.curr_epoch < self.warmup_epochs:
            loss = ntxent_loss(logits, tau=self.temparature,
                               item_weights=self.embedding_weight)
        elif self.curr_epoch < self.warmup_epochs + self.main_epochs:
            loss = refined_ntxent_loss(
                logits, tau=self.temparature, item_weights=self.embedding_weight)
        else:
            loss = ntxent_loss(logits, tau=self.temparature,
                               item_weights=self.embedding_weight)
        return loss
