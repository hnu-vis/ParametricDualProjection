import torch

from train import ConstractiveTrainer


class SampleTrainer(ConstractiveTrainer):
    def __init__(self, dataset, model, log_dir, device = ..., dtype = torch.float32):
        super().__init__(dataset, model, log_dir, device, dtype)

    def setup_loss_weights(self):
        self.embedding_weight = 1
        self.auxiliary_weight = 1e-2
        self.latent_weight = 1e-2
        self.reconstruction_weight = 0.1


class FeatureTrainer(ConstractiveTrainer):
    def __init__(self, dataset, model, log_dir, device = ..., dtype = torch.float32):
        super().__init__(dataset, model, log_dir, device, dtype)
        
    def setup_loss_weights(self):
        self.embedding_weight = 1
        self.auxiliary_weight = 1e-2
        self.latent_weight = 1e-2
        self.reconstruction_weight = 0.1
        
    def compute_dist(self, X, Y):
        X_mean = X.mean(dim=1, keepdim=True)
        Y_mean = Y.mean(dim=1, keepdim=True)

        X_centered = X - X_mean
        Y_centered = Y - Y_mean

        covariance_matrix = torch.mm(X_centered, Y_centered.T)

        X_std = torch.sqrt(torch.sum(X_centered ** 2, dim=1, keepdim=True))
        Y_std = torch.sqrt(torch.sum(Y_centered ** 2, dim=1, keepdim=True))

        correlation_matrix = covariance_matrix / (X_std * Y_std.T)

        distances = 1 - correlation_matrix.abs()

        return distances
