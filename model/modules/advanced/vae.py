import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base.cnn import (ConvBlock, ConvTransposeBlock, compute_conv_output_dim,
                        compute_conv_transpose_output_padding)
from ..base.mlp import MLP


class AeMlp(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.encoder = MLP(dims)
        self.decoder = MLP(dims[::-1])

    def encode(self, x):
        z = self.encoder(x)
        return z

    def decode(self, z):
        x = self.decoder(z)
        return x

    def forward(self, x):
        z = self.encode(x)
        x_rec = self.decode(z)
        return z, x_rec


class AeCnn(nn.Module):
    def __init__(
        self,
        in_channels: int,
        in_dim: int,
        latent_dim: int = 64,
        conv_dim: int = 2,  # 1 for 1D, 2 for 2D
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.conv_dim = conv_dim

        # 编码器部分
        self.conv_block1 = ConvBlock(
            in_channels=in_channels, out_channels=32,
            kernel_size=3, stride=2, padding=1, conv_dim=conv_dim
        )

        self.encoder_out_dim1 = compute_conv_output_dim(in_dim, 3, 2, 1)

        self.conv_block2 = ConvBlock(
            in_channels=32, out_channels=64,
            kernel_size=3, stride=2, padding=1, conv_dim=conv_dim
        )

        self.encoder_out_dim2 = compute_conv_output_dim(
            self.encoder_out_dim1, 3, 2, 1)

        self.conv_block3 = ConvBlock(
            in_channels=64, out_channels=128,
            kernel_size=3, stride=2, padding=1, conv_dim=conv_dim
        )

        self.encoder_out_dim3 = compute_conv_output_dim(
            self.encoder_out_dim2, 3, 2, 1)

        # 编码器的全连接层
        if conv_dim == 1:
            self.encode_fc1 = nn.Linear(self.encoder_out_dim3 * 128, 1024)
        else:  # conv_dim == 2
            self.encode_fc1 = nn.Linear(
                self.encoder_out_dim3 * self.encoder_out_dim3 * 128, 1024)
        self.encode_fc2 = nn.Linear(1024, latent_dim)

        # 解码器部分
        self.decode_fc2 = nn.Linear(latent_dim, 1024)
        if conv_dim == 1:
            self.decode_fc1 = nn.Linear(1024, 128 * self.encoder_out_dim3)
        else:  # conv_dim == 2
            self.decode_fc1 = nn.Linear(
                1024, 128 * self.encoder_out_dim3 * self.encoder_out_dim3)

        self.decoder_output_padding3 = compute_conv_transpose_output_padding(
            in_size=self.encoder_out_dim3, target_size=self.encoder_out_dim2,
            kernel_size=3, stride=2, padding=1
        )

        self.conv_transpose_block3 = ConvTransposeBlock(
            in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1,
            output_padding=self.decoder_output_padding3, conv_dim=conv_dim
        )

        self.decoder_output_padding2 = compute_conv_transpose_output_padding(
            in_size=self.encoder_out_dim2, target_size=self.encoder_out_dim1,
            kernel_size=3, stride=2, padding=1
        )

        self.conv_transpose_block2 = ConvTransposeBlock(
            in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=1,
            output_padding=self.decoder_output_padding2, conv_dim=conv_dim
        )

        self.decoder_output_padding1 = compute_conv_transpose_output_padding(
            in_size=self.encoder_out_dim1, target_size=in_dim,
            kernel_size=3, stride=2, padding=1
        )

        self.conv_transpose_block1 = ConvTransposeBlock(
            in_channels=32, out_channels=in_channels, kernel_size=3, stride=2, padding=1,
            output_padding=self.decoder_output_padding1, relu=False, conv_dim=conv_dim
        )

    def encode(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)

        x = x.view(x.size(0), -1)

        x = F.relu(self.encode_fc1(x))
        z = self.encode_fc2(x)

        return z

    def decode(self, z):
        x = F.relu(self.decode_fc2(z))
        x = F.relu(self.decode_fc1(x))

        if self.conv_dim == 1:
            x = x.view(x.size(0), 128, self.encoder_out_dim3)
        else:  # conv_dim == 2
            x = x.view(x.size(0), 128, self.encoder_out_dim3,
                       self.encoder_out_dim3)

        x = self.conv_transpose_block3(x)
        x = self.conv_transpose_block2(x)
        x = self.conv_transpose_block1(x)

        return x

    def forward(self, x):
        z = self.encode(x)
        x_rec = self.decode(z)
        return z, x_rec


class AutoVae(nn.Module):
    def __init__(
        self,
        in_channels: int,
        in_dim: int,
        latent_dim: int = 64,
        conv_dim: int = 2,
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.conv_dim = conv_dim

        self.conv_block1 = ConvBlock(
            in_channels=in_channels, out_channels=32,
            kernel_size=3, stride=2, padding=1, conv_dim=conv_dim
        )

        self.encoder_out_dim1 = compute_conv_output_dim(in_dim, 3, 2, 1)

        self.conv_block2 = ConvBlock(
            in_channels=32, out_channels=64,
            kernel_size=3, stride=2, padding=1, conv_dim=conv_dim
        )

        self.encoder_out_dim2 = compute_conv_output_dim(
            self.encoder_out_dim1, 3, 2, 1)

        self.conv_block3 = ConvBlock(
            in_channels=64, out_channels=128,
            kernel_size=3, stride=2, padding=1, conv_dim=conv_dim
        )

        self.encoder_out_dim3 = compute_conv_output_dim(
            self.encoder_out_dim2, 3, 2, 1)

        if conv_dim == 1:
            self.encode_fc1 = nn.Linear(self.encoder_out_dim3 * 128, 1024)
        else:  # conv_dim == 2
            self.encode_fc1 = nn.Linear(
                self.encoder_out_dim3 * self.encoder_out_dim3 * 128, 1024)

        self.fc_mean = nn.Linear(1024, latent_dim)
        self.fc_logvar = nn.Linear(1024, latent_dim)

        self.decode_fc2 = nn.Linear(latent_dim, 1024)
        if conv_dim == 1:
            self.decode_fc1 = nn.Linear(1024, 128 * self.encoder_out_dim3)
        else:  # conv_dim == 2
            self.decode_fc1 = nn.Linear(
                1024, 128 * self.encoder_out_dim3 * self.encoder_out_dim3)

        self.decoder_output_padding3 = compute_conv_transpose_output_padding(
            in_size=self.encoder_out_dim3, target_size=self.encoder_out_dim2,
            kernel_size=3, stride=2, padding=1
        )

        self.conv_transpose_block3 = ConvTransposeBlock(
            in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1,
            output_padding=self.decoder_output_padding3, conv_dim=conv_dim
        )

        self.decoder_output_padding2 = compute_conv_transpose_output_padding(
            in_size=self.encoder_out_dim2, target_size=self.encoder_out_dim1,
            kernel_size=3, stride=2, padding=1
        )

        self.conv_transpose_block2 = ConvTransposeBlock(
            in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=1,
            output_padding=self.decoder_output_padding2, conv_dim=conv_dim
        )

        self.decoder_output_padding1 = compute_conv_transpose_output_padding(
            in_size=self.encoder_out_dim1, target_size=in_dim,
            kernel_size=3, stride=2, padding=1
        )

        self.conv_transpose_block1 = ConvTransposeBlock(
            in_channels=32, out_channels=in_channels, kernel_size=3, stride=2, padding=1,
            output_padding=self.decoder_output_padding1, relu=False, conv_dim=conv_dim
        )

    def encode(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)

        x = x.view(x.size(0), -1)

        x = F.relu(self.encode_fc1(x))

        mean = self.fc_mean(x)
        logvar = self.fc_logvar(x)

        return mean, logvar

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def decode(self, z):
        x = F.relu(self.decode_fc2(z))
        x = F.relu(self.decode_fc1(x))

        if self.conv_dim == 1:
            x = x.view(x.size(0), 128, self.encoder_out_dim3)
        else:  # conv_dim == 2
            x = x.view(x.size(0), 128, self.encoder_out_dim3,
                       self.encoder_out_dim3)

        x = self.conv_transpose_block3(x)
        x = self.conv_transpose_block2(x)
        x = self.conv_transpose_block1(x)

        return x

    def forward(self, x):
        mean, logvar = self.encode(x)

        z = self.reparameterize(mean, logvar)

        x_reconstructed = self.decode(z)

        return x_reconstructed, mean, logvar
