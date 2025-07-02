from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from .configs.base_config import BaseConfig


class BaseEmbedder(nn.Module, ABC):
    def __init__(self, config: BaseConfig):
        super().__init__()
        self.config = config
        self.input_dim = config.get_input_dim()
        self.latent_dim = config.get_latent_dim()
        self.output_dim = config.get_output_dim()

        self.presenting_dims = config.presenting_dims
        self.embedding_dims = config.embedding_dims

        self.compressor = None
        self.embedder = None

    def get_weighted_auxiliary_feature(self, embedding_features):
        if self.embedding_features is None:
            raise ValueError("Before calling this method, please call the `prepare` method to initialize `self.embedding_features`")

        flattened_embedding = self.embedding_features.reshape(-1, self.output_dim)
        flattened_input = embedding_features.reshape(-1, self.output_dim)

        distances = torch.cdist(flattened_input, flattened_embedding, p=2)

        k_nearest = 5
        _, nearest_indices = distances.topk(k_nearest, dim=1, largest=False)

        nearest_auxiliary_feature = self.auxiliary_features.reshape(-1, self.auxiliary_features.size(-1))[nearest_indices]

        nearest_distances = distances.gather(1, nearest_indices)
        weights = 1.0 / (nearest_distances + 1e-6)
        weights /= weights.sum(dim=1, keepdim=True)
        weighted_auxiliary_feature = (nearest_auxiliary_feature * weights.unsqueeze(-1)).sum(dim=1)

        return weighted_auxiliary_feature

    @abstractmethod
    def prepare(self, train_data):
        pass

    @abstractmethod
    def encode(self, data):
        pass

    @abstractmethod
    def decode(self, embedding_features):
        pass

    @abstractmethod
    def forward(self, data):
        pass
