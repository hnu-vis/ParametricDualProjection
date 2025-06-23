import torch

from train import ConstractiveTrainer

default_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
default_dtype = torch.float64


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
