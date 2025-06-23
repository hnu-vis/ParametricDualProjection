import torch

from .base_embedder import BaseEmbedder
from .configs import BaseConfig
from .modules.advanced.flow import Flow
from .modules.advanced.vae import AeCnn


class CnnEmbedder(BaseEmbedder):
    def __init__(self, config: BaseConfig, conv_dim=2):
        super().__init__(config=config, conv_dim=conv_dim)
        self.config = config
        self.input_dim = config.get_input_dim()
        self.latent_dim = config.get_latent_dim()
        self.output_dim = config.get_output_dim()

        self.presenting_dims = config.presenting_dims
        self.embedding_dims = config.embedding_dims

        self.compressor = AeCnn(
            1, self.input_dim, self.latent_dim, conv_dim)
        self.embedder = Flow(self.embedding_dims, shuffle=True)

    def prepare(self, train_data):
        # train_data = train_data.unsqueeze(dim=1)

        latent_features = self.compressor(train_data)
        embedding_features = self.embedder.encode(latent_features)

        self.embedding_features = embedding_features[..., :self.output_dim]
        self.auxiliary_features = embedding_features[..., self.output_dim:]

    def encode(self, data):
        # data = data.unsqueeze(dim=1)

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

        return reconstructed_data.squeeze()

    def forward(self, data):
        # data = data.unsqueeze(dim=1)

        latent_features = self.compressor.encode(data)
        embedding_features = self.embedder.encode(latent_features)

        reconstructed_features = self.embedder.decode(embedding_features)
        reconstructed_data = self.compressor.decode(reconstructed_features)
        # reconstructed_data = self.compressor.decode(latent_features)

        return embedding_features[..., :self.output_dim], latent_features, reconstructed_data
