import torch

from .base_embedder import BaseEmbedder
from .configs import BaseConfig
from .modules.advanced.flow import Flow


class FlowEmbedder(BaseEmbedder):
    def __init__(self, config: BaseConfig):
        super().__init__()
        self.config = config
        self.input_dim = config.get_input_dim()  # 输入维度
        self.latent_dim = config.get_latent_dim()  # 潜在空间维度
        self.output_dim = config.get_output_dim()  # 输出维度
        self.noise_dim = self.input_dim - self.latent_dim

        self.presenting_dims = config.presenting_dims  # 表示学习过程的维度列表
        self.embedding_dims = config.embedding_dims  # 嵌入学习过程的维度列表

        # 表示学习模块（input_data -> [latent_feature, gaussian_noise]）
        self.presenting_flow = Flow(self.presenting_dims, shuffle=True)

        # 嵌入学习模块（latent_feature -> [embedding_feature, auxiliary_feature]）
        self.embedding_flow = Flow(self.embedding_dims, shuffle=True)

    def prepare(self, input_data):
        # 表示学习：input_data -> [latent_feature, gaussian_noise]
        input_data = self.presenting_flow.encode(input_data)

        # 分离潜在特征和高斯噪声
        latent_feature = input_data[..., :self.latent_dim]  # 潜在特征
        # gaussian_noise = input_data[..., self.latent_dim:]  # 高斯噪声

        # 嵌入学习：latent_feature -> [embedding_feature, auxiliary_feature]
        latent_feature = self.embedding_flow.encode(latent_feature)

        # 保存嵌入特征和辅助特征
        self.embedding_feature = latent_feature[..., :self.output_dim]  # 嵌入特征
        self.auxiliary_feature = latent_feature[..., self.output_dim:]  # 辅助特征

    def encode(self, input_data):
        # 表示学习：input_data -> [latent_feature, gaussian_noise]
        input_data = self.presenting_flow.encode(input_data)

        # 提取潜在特征
        latent_feature = input_data[..., :self.latent_dim]

        # 嵌入学习：latent_feature -> [embedding_feature, auxiliary_feature]
        latent_feature = self.embedding_flow.encode(latent_feature)

        return latent_feature[..., :self.output_dim]

    def decode(self, embedding_feature):
        weighted_auxiliary_feature = self.get_weighted_auxiliary_feature(
            embedding_feature)

        # 拼接加权辅助特征到嵌入特征
        reconstructed_feature = torch.cat(
            (embedding_feature, weighted_auxiliary_feature), dim=-1)

        # 反向嵌入学习
        reconstructed_feature = self.embedding_flow.decode(
            reconstructed_feature)

        # 从高斯分布采样噪声
        sampled_noise = torch.randn(
            (reconstructed_feature.size(0), self.input_dim - self.latent_dim),
            device=reconstructed_feature.device,
            dtype=reconstructed_feature.dtype
        )
        reconstructed_data = torch.cat(
            [reconstructed_feature, sampled_noise], dim=-1)

        # 反向表示学习
        reconstructed_data = self.presenting_flow.decode(reconstructed_data)

        return reconstructed_data

    def encode_for_train(self, input_data):
        # 表示学习：input_data -> [latent_feature, gaussian_noise]
        input_data = self.presenting_flow.encode(input_data)

        # 分离潜在特征和高斯噪声
        gaussian_noise = input_data[..., self.latent_dim:]  # 高斯噪声
        latent_feature = input_data[..., :self.latent_dim]  # 潜在特征

        # 嵌入学习：latent_feature -> [embedding_feature, auxiliary_feature]
        latent_feature = self.embedding_flow.encode(latent_feature)

        # 辅助特征
        self.auxiliary_feature = latent_feature[..., self.output_dim:]
        embedding_feature = latent_feature[..., :self.output_dim]  # 嵌入特征

        return embedding_feature, gaussian_noise

    def decode_for_train(self, embedding_feature):
        # 拼接辅助特征
        reconstructed_feature = torch.cat(
            [embedding_feature, self.auxiliary_feature], dim=-1)

        # 反向嵌入学习
        reconstructed_feature = self.embedding_flow.decode(
            reconstructed_feature)

        # 从高斯分布采样噪声
        sampled_noise = torch.randn(
            (reconstructed_feature.size(0), self.noise_dim),
            device=reconstructed_feature.device,
            dtype=reconstructed_feature.dtype
        )

        reconstructed_data = torch.cat(
            [reconstructed_feature, sampled_noise], dim=-1)

        # 反向表示学习
        reconstructed_data = self.presenting_flow.decode(reconstructed_data)

        return reconstructed_data
