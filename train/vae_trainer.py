import torch
import torch.nn.functional as F

import loss
from train import BaseTrainer


class VaeTrainer(BaseTrainer):
    def training_step(self, batch):
        """单个批次的训练步骤"""
        self.optimizer.zero_grad()

        # 前向传播
        latent_features, mean, logvar = self.model.encode_for_train(batch)
        reconstructed = self.model.decode_for_train(latent_features)

        # 损失计算
        losses = self.compute_losses(
            batch, mean, logvar, reconstructed
        )

        # 反向传播
        losses["total"].backward()
        self.optimizer.step()

        # 返回标量值
        return {k: v.item() for k, v in losses.items()}

    def compute_losses(self, inputs, mean, logvar, reconstructed):
        """计算各损失分量"""
        return {
            "total": self.total_loss(inputs, mean, logvar, reconstructed),
            "embedding": torch.tensor(0),
            "latent": self.latent_loss(mean, logvar),
            "reconstruction": self.reconstruction_loss(inputs, reconstructed)
        }

    # 损失计算组件（可扩展）
    def total_loss(self, inputs, mean, logvar, reconstructed):
        """计算总损失"""
        return (
            self.latent_weight * self.latent_loss(mean, logvar) +
            self.reconstruction_weight *
            self.reconstruction_loss(inputs, reconstructed)
        )

    def latent_loss(self, mean, logvar):
        return loss.kl_loss(mean, logvar)

    def plot_embedding(self, suffix):
        return

    def setup_loss_weights(self):
        """初始化损失权重系数"""
        self.embedding_weight = 0
        self.latent_weight = 1e-2
        self.reconstruction_weight = 1

    def reconstruction_loss(self, inputs, reconstructed):
        # 检查 inputs 是否在 [0, 1] 范围内
        if (inputs < 0).any() or (inputs > 1).any():
            print("Inputs 中有值不在 [0, 1] 范围内！")
            print("Inputs 的最小值:", inputs.min().item())
            print("Inputs 的最大值:", inputs.max().item())
            print("Inputs 的非法值:", inputs[(inputs < 0) | (inputs > 1)])

        # 检查 reconstructed 是否在 [0, 1] 范围内
        if (reconstructed < 0).any() or (reconstructed > 1).any():
            print("Reconstructed 中有值不在 [0, 1] 范围内！")
            print("Reconstructed 的最小值:", reconstructed.min().item())
            print("Reconstructed 的最大值:", reconstructed.max().item())
            print("Reconstructed 的非法值:", reconstructed[(reconstructed < 0) | (reconstructed > 1)])
        return F.binary_cross_entropy(inputs, reconstructed, reduction='sum')
