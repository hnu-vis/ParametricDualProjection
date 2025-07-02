import torch

from .base_embedder import BaseEmbedder
from .configs.base_config import BaseConfig
from .modules.advanced.flow import Flow
from .modules.advanced.vae import AeMlp


class MlpEmbedder(BaseEmbedder):
    def __init__(self, config: BaseConfig):
        super().__init__(config=config)

        self.compressor = AeMlp(self.presenting_dims)
        self.embedder = Flow(self.embedding_dims, shuffle=True)

    def prepare(self, train_data):
        latent_features = self.compressor.encode(train_data)
        embedding_features = self.embedder.encode(latent_features)

        self.embedding_features = embedding_features[..., :self.output_dim]
        self.auxiliary_features = embedding_features[..., self.output_dim:]

    def encode(self, data):
        latent_features = self.compressor.encode(data)
        embedding_features = self.embedder.encode(latent_features)

        return embedding_features[..., :self.output_dim]

    def decode(self, embedding_features):
        weighted_auxiliary_features = self.get_weighted_auxiliary_feature(
            embedding_features)

        reconstructed_features = torch.cat(
            (embedding_features, weighted_auxiliary_features), dim=-1)

        latent_features = self.embedder.decode(reconstructed_features)
        reconstructed_data = self.compressor.decode(latent_features)

        return reconstructed_data
    
    def decode_test(self, embedding_features):
        latent_features = self.embedder.decode(embedding_features)
        reconstructed_data = self.compressor.decode(latent_features)

        return reconstructed_data

    def forward(self, data):
        latent_features = self.compressor.encode(data)
        embedding_features = self.embedder.encode(latent_features)

        reconstructed_features = self.embedder.decode(embedding_features)
        reconstructed_data = self.compressor.decode(reconstructed_features)

        auxiliary_features = embedding_features[..., self.output_dim:]
        embedding_features = embedding_features[..., :self.output_dim]

        return embedding_features, auxiliary_features, latent_features, reconstructed_data
