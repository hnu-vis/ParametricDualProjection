from . import BaseTrainer, ConstractiveTrainer
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
from data.archive.gene_loader import GeneDataset, GeneFeatureDataset
from tqdm import tqdm

import loss
from loss.nce_loss import ntxent_loss, refined_ntxent_loss
from model import BaseEmbedder
from utils import data_utils, vis_utils


class GeneTrainer(ConstractiveTrainer):
    def __init__(
        self,
        dataset: GeneDataset,
        model: BaseEmbedder,
        log_dir: str,
        device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"),
        dtype: torch.dtype = torch.float32
    ):
        super().__init__(dataset, model, log_dir, device, dtype)
        self.loss_history = {
            "total": [],
            "embedding": [],
            "auxiliary": [],
            "yield": [],
            "yield_auxiliary": [],
            "grain_shape": [],
            "grain_shape_auxiliary": [],
            "latent": [],
            "reconstruction": []
        }

    def setup_loss_weights(self):
        self.embedding_weight = 1.0
        self.auxiliary_weight = 1.0
        self.yield_weight = 1.0
        self.yield_auxiliary_weight = 1.0
        self.grain_shape_weight = 1.0
        self.grain_shape_auxiliary_weight = 1.0
        self.latent_weight = 1.0
        self.reconstruction_weight = 1.0


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
                "Yie.": f"{epoch_metrics['yield']:.4f}",
                "Yia.": f"{epoch_metrics['yield_auxiliary']:.4f}",
                "Lwr.": f"{epoch_metrics['grain_shape']:.4f}",
                "Lwa.": f"{epoch_metrics['grain_shape_auxiliary']:.4f}",
                "Lat.": f"{epoch_metrics['latent']:.4f}",
                "Rec.": f"{epoch_metrics['reconstruction']:.4f}"
            })

        self.after_training()

    def run_epoch(self):
        epoch_metrics = {
            "total": 0.0,
            "embedding": 0.0,
            "auxiliary": 0.0,
            "yield": 0.0,
            "yield_auxiliary": 0.0,
            "grain_shape": 0.0,
            "grain_shape_auxiliary": 0.0,
            "latent": 0.0,
            "reconstruction": 0.0
        }

        for batch_id, (data, labels) in enumerate(self.dataloader):  # 解包数据和标签
            self.curr_batch_id = batch_id
            data = data.to(self.device, dtype=self.dtype)  # 将数据移动到设备
            labels = labels.to(self.device)
            yield_labels = labels[:, 0]
            grain_shape_labels = labels[:, 1]

            batch_metrics = self.training_step(data, yield_labels, grain_shape_labels)  # 传递数据和标签

            for key in epoch_metrics:
                epoch_metrics[key] += batch_metrics[key]

        # 计算平均损失
        for key in epoch_metrics:
            epoch_metrics[key] /= len(self.dataloader)

        return epoch_metrics

    def training_step(self, data: torch.Tensor, yield_labels: torch.Tensor, grain_shape_labels: torch.Tensor):
        self.optimizer.zero_grad()

        # 前向传播
        embedding_features, auxiliary_features, latent_features, reconstructed_data = self.model.forward(
            data)

        # 损失计算
        losses = self.compute_losses(
            data, embedding_features, auxiliary_features, latent_features, reconstructed_data, yield_labels, grain_shape_labels
        )

        # 反向传播
        losses["total"].backward()
        self.optimizer.step()

        # 返回标量值
        return {k: v.item() for k, v in losses.items()}

    def compute_losses(self, data, embedding_features, auxiliary_features, latent_features, reconstructed_data, yield_labels, grain_shape_labels):
        return {
            "total": self.total_loss(data, embedding_features, auxiliary_features, latent_features, reconstructed_data, yield_labels, grain_shape_labels),
            "embedding": self.embedding_loss(data, embedding_features),
            "auxiliary": self.auxiliary_loss(data, auxiliary_features),
            "yield": self.yield_loss(data, embedding_features, yield_labels),
            "yield_auxiliary": self.yield_auxiliary_loss(data, auxiliary_features, yield_labels),
            "grain_shape": self.grain_shape_loss(data, embedding_features, grain_shape_labels),
            "grain_shape_auxiliary": self.grain_shape_auxiliary_loss(data, auxiliary_features, grain_shape_labels),
            "latent": self.latent_loss(latent_features),
            "reconstruction": self.reconstruction_loss(data, reconstructed_data)
        }

    def total_loss(self, data, embedding_features, auxiliary_features, latent_features, reconstructed_data, yield_labels, grain_shape_labels):
        return (
            self.embedding_weight * self.embedding_loss(data, embedding_features) +
            self.auxiliary_weight * self.auxiliary_loss(data, auxiliary_features) +
            self.yield_weight * self.yield_loss(data, embedding_features, yield_labels) +
            self.yield_auxiliary_weight * self.yield_auxiliary_loss(data, auxiliary_features, yield_labels) +
            self.grain_shape_weight * self.grain_shape_loss(data, embedding_features, grain_shape_labels) +
            self.grain_shape_auxiliary_weight * self.grain_shape_loss(data, auxiliary_features, grain_shape_labels) +
            self.latent_weight * self.latent_loss(latent_features) +
            self.reconstruction_weight *
            self.reconstruction_loss(data, reconstructed_data)
        )

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
        yield_labels = labels[:, 0]
        grain_shape_labels = labels[:, 1]
        vis_utils.plot_embeddings(
            embeddings,
            yield_labels,
            discrete=False,
            save_path=os.path.join(self.log_dir, f"embedding_yield_{suffix}.png")
        )
        vis_utils.plot_embeddings(
            embeddings,
            grain_shape_labels,
            discrete=False,
            save_path=os.path.join(self.log_dir, f"embedding_grain_shape_{suffix}.png")
        )

    def plot_loss_curve(self):
        vis_utils.plot_loss_curve(
            self.loss_history,
            save_path=os.path.join(self.log_dir, f"loss_curve.png")
        )

    def yield_loss(self, data, embedding_features, labels):
        data = data.view(data.size(0), -1)
        pos_indices = self.get_positive_indices_batch(
            data, labels, k=self.k)  # 默认 labels=None
        embedding_sim_features = self.get_embeddings_for_loss(
            embedding_features, pos_indices)
        logits = self.batch_logits(
            embedding_features, embedding_sim_features, a=1.8956058664239412, b=0.8006378441176886)

        if self.curr_epoch < self.warmup_epochs:
            loss = ntxent_loss(logits, tau=self.temparature,
                               item_weights=self.embedding_weight)
        elif self.curr_epoch < self.warmup_epochs + self.main_epochs:
            loss = refined_ntxent_loss(
                logits, tau=0.15, item_weights=self.embedding_weight)
        else:
            loss = ntxent_loss(logits, tau=self.temparature,
                               item_weights=self.embedding_weight)
        return loss
    
    def yield_auxiliary_loss(self, data, auxiliary_features, labels):
        data = data.view(data.size(0), -1)
        pos_indices = self.get_positive_indices_batch(
            data, labels, k=self.k)  # 默认 labels=None
        embedding_sim_features = self.get_embeddings_for_loss(
            auxiliary_features, pos_indices)
        logits = self.batch_logits(
            auxiliary_features, embedding_sim_features, a=1.8956058664239412, b=0.8006378441176886)

        if self.curr_epoch < self.warmup_epochs:
            loss = ntxent_loss(logits, tau=self.temparature,
                               item_weights=self.embedding_weight)
        elif self.curr_epoch < self.warmup_epochs + self.main_epochs:
            loss = refined_ntxent_loss(
                logits, tau=0.15, item_weights=self.embedding_weight)
        else:
            loss = ntxent_loss(logits, tau=self.temparature,
                               item_weights=self.embedding_weight)
        return loss
    
    def grain_shape_loss(self, data, embedding_features, labels):
        data = data.view(data.size(0), -1)
        pos_indices = self.get_positive_indices_batch(
            data, labels, k=self.k)  # 默认 labels=None
        embedding_sim_features = self.get_embeddings_for_loss(
            embedding_features, pos_indices)
        logits = self.batch_logits(
            embedding_features, embedding_sim_features, a=1.8956058664239412, b=0.8006378441176886)

        if self.curr_epoch < self.warmup_epochs:
            loss = ntxent_loss(logits, tau=self.temparature,
                               item_weights=self.embedding_weight)
        elif self.curr_epoch < self.warmup_epochs + self.main_epochs:
            loss = refined_ntxent_loss(
                logits, tau=0.15, item_weights=self.embedding_weight)
        else:
            loss = ntxent_loss(logits, tau=self.temparature,
                               item_weights=self.embedding_weight)
        return loss
    
    def grain_shape_auxiliary_loss(self, data, auxiliary_features, labels):
        data = data.view(data.size(0), -1)
        pos_indices = self.get_positive_indices_batch(
            data, labels, k=self.k)  # 默认 labels=None
        embedding_sim_features = self.get_embeddings_for_loss(
            auxiliary_features, pos_indices)
        logits = self.batch_logits(
            auxiliary_features, embedding_sim_features, a=1.8956058664239412, b=0.8006378441176886)

        if self.curr_epoch < self.warmup_epochs:
            loss = ntxent_loss(logits, tau=self.temparature,
                               item_weights=self.embedding_weight)
        elif self.curr_epoch < self.warmup_epochs + self.main_epochs:
            loss = refined_ntxent_loss(
                logits, tau=0.15, item_weights=self.embedding_weight)
        else:
            loss = ntxent_loss(logits, tau=self.temparature,
                               item_weights=self.embedding_weight)
        return loss


class GeneSampleTrainer(GeneTrainer):
    def setup_loss_weights(self):
        """初始化损失权重系数"""
        self.embedding_weight = 1.0
        self.auxiliary_weight = 1e-2
        self.yield_weight = 1
        self.yield_auxiliary_weight = 1e-2
        self.grain_shape_weight = 1
        self.grain_shape_auxiliary_weight = 1e-2
        # self.yield_weight = 0.55
        # self.grain_shape_weight = 0.7
        self.latent_weight = 1e-2
        self.reconstruction_weight = 0.1


class GeneFeatureTrainer(ConstractiveTrainer):
    def __init__(
        self,
        dataset: GeneFeatureDataset,
        model: BaseEmbedder,
        log_dir: str,
        device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"),
        dtype: torch.dtype = torch.float32
    ):
        super().__init__(dataset, model, log_dir, device, dtype)
        self.loss_history = {
            "total": [],
            "embedding": [],
            "auxiliary": [],
            "chr": [],
            "chr_auxiliary": [],
            "latent": [],
            "reconstruction": []
        }

    def setup_loss_weights(self):
        self.embedding_weight = 1.0
        self.auxiliary_weight = 1e-2
        self.chr_weight = 1.0
        self.chr_auxiliary_weight = 1e-2
        self.latent_weight = 1e-2
        self.reconstruction_weight = 0.1


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
                "Chr.": f"{epoch_metrics['chr']:.4f}",
                "Cha.": f"{epoch_metrics['chr_auxiliary']:.4f}",
                "Lat.": f"{epoch_metrics['latent']:.4f}",
                "Rec.": f"{epoch_metrics['reconstruction']:.4f}"
            })

        self.after_training()

    def run_epoch(self):
        epoch_metrics = {
            "total": 0.0,
            "embedding": 0.0,
            "auxiliary": 0.0,
            "chr": 0.0,
            "chr_auxiliary": 0.0,
            "latent": 0.0,
            "reconstruction": 0.0
        }

        for batch_id, (data, labels) in enumerate(self.dataloader):  # 解包数据和标签
            self.curr_batch_id = batch_id
            data = data.to(self.device, dtype=self.dtype)  # 将数据移动到设备
            labels = labels.to(self.device)

            batch_metrics = self.training_step(data, labels)  # 传递数据和标签

            for key in epoch_metrics:
                epoch_metrics[key] += batch_metrics[key]

        # 计算平均损失
        for key in epoch_metrics:
            epoch_metrics[key] /= len(self.dataloader)

        return epoch_metrics

    def training_step(self, data: torch.Tensor, labels: torch.Tensor):
        self.optimizer.zero_grad()

        # 前向传播
        embedding_features, auxiliary_features, latent_features, reconstructed_data = self.model.forward(
            data)

        # 损失计算
        losses = self.compute_losses(
            data, embedding_features, auxiliary_features, latent_features, reconstructed_data, labels
        )

        # 反向传播
        losses["total"].backward()
        self.optimizer.step()

        # 返回标量值
        return {k: v.item() for k, v in losses.items()}

    def compute_losses(self, data, embedding_features, auxiliary_features, latent_features, reconstructed_data, labels):
        return {
            "total": self.total_loss(data, embedding_features, auxiliary_features, latent_features, reconstructed_data, labels),
            "embedding": self.embedding_loss(data, embedding_features),
            "auxiliary": self.auxiliary_loss(data, auxiliary_features),
            "chr": self.chr_loss(data, embedding_features, labels),
            "chr_auxiliary": self.chr_auxiliary_loss(data, auxiliary_features, labels),
            "latent": self.latent_loss(latent_features),
            "reconstruction": self.reconstruction_loss(data, reconstructed_data)
        }

    def total_loss(self, data, embedding_features, auxiliary_features, latent_features, reconstructed_data, labels):
        return (
            self.embedding_weight * self.embedding_loss(data, embedding_features) +
            self.auxiliary_weight * self.auxiliary_loss(data, auxiliary_features) +
            self.chr_weight * self.chr_loss(data, embedding_features, labels) +
            self.chr_auxiliary_weight * self.chr_auxiliary_loss(data, auxiliary_features, labels) +
            self.latent_weight * self.latent_loss(latent_features) +
            self.reconstruction_weight *
            self.reconstruction_loss(data, reconstructed_data)
        )

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

    def chr_loss(self, data, embedding_features, labels):
        data = data.view(data.size(0), -1)
        pos_indices = self.get_positive_indices_batch(
            data, labels, k=self.k)  # 默认 labels=None
        embedding_sim_features = self.get_embeddings_for_loss(
            embedding_features, pos_indices)
        logits = self.batch_logits(
            embedding_features, embedding_sim_features, a=1.8956058664239412, b=0.8006378441176886)

        if self.curr_epoch < self.warmup_epochs:
            loss = ntxent_loss(logits, tau=self.temparature,
                               item_weights=self.embedding_weight)
        elif self.curr_epoch < self.warmup_epochs + self.main_epochs:
            loss = refined_ntxent_loss(
                logits, tau=0.15, item_weights=self.embedding_weight)
        else:
            loss = ntxent_loss(logits, tau=self.temparature,
                               item_weights=self.embedding_weight)
        return loss
    
    def chr_auxiliary_loss(self, data, auxiliary_feature, labels):
        data = data.view(data.size(0), -1)
        pos_indices = self.get_positive_indices_batch(
            data, labels, k=self.k)  # 默认 labels=None
        embedding_sim_features = self.get_embeddings_for_loss(
            auxiliary_feature, pos_indices)
        logits = self.batch_logits(
            auxiliary_feature, embedding_sim_features, a=1.8956058664239412, b=0.8006378441176886)

        if self.curr_epoch < self.warmup_epochs:
            loss = ntxent_loss(logits, tau=self.temparature,
                               item_weights=self.embedding_weight)
        elif self.curr_epoch < self.warmup_epochs + self.main_epochs:
            loss = refined_ntxent_loss(
                logits, tau=0.15, item_weights=self.embedding_weight)
        else:
            loss = ntxent_loss(logits, tau=self.temparature,
                               item_weights=self.embedding_weight)
        return loss
    