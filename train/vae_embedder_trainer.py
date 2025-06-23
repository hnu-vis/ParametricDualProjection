import torch

from train import ConstractiveTrainer


class SampleTrainer(ConstractiveTrainer):
    def setup_loss_weights(self):
        """初始化损失权重系数"""
        self.embedding_weight = 1
        self.auxiliary_weight = 1e-2
        self.latent_weight = 1e-2
        self.reconstruction_weight = 0.1


class FeatureTrainer(ConstractiveTrainer):
    def setup_loss_weights(self):
        """初始化损失权重系数"""
        self.embedding_weight = 1
        self.auxiliary_weight = 1e-2
        self.latent_weight = 1e-2
        self.reconstruction_weight = 0.1
        
    def compute_dist(self, X, Y):
        # 计算每个样本的均值
        X_mean = X.mean(dim=1, keepdim=True)
        Y_mean = Y.mean(dim=1, keepdim=True)

        # 中心化数据
        X_centered = X - X_mean
        Y_centered = Y - Y_mean

        # 计算协方差矩阵
        covariance_matrix = torch.mm(X_centered, Y_centered.T)

        # 计算标准差
        X_std = torch.sqrt(torch.sum(X_centered ** 2, dim=1, keepdim=True))
        Y_std = torch.sqrt(torch.sum(Y_centered ** 2, dim=1, keepdim=True))

        # 计算皮尔逊相关性矩阵
        correlation_matrix = covariance_matrix / (X_std * Y_std.T)

        distances = 1 - correlation_matrix.abs()

        return distances
