import torch.nn as nn


def compute_conv_output_dim(in_size, kernel_size, stride, padding):
    return (in_size + 2 * padding - kernel_size) // stride + 1


def compute_conv_transpose_output_padding(in_size, target_size, kernel_size, stride, padding):
    output_size = (in_size - 1) * stride + kernel_size - 2 * padding
    return target_size - output_size


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
        conv_dim: int = 2,
        relu: bool = True,
    ):
        super().__init__()

        conv_layer = nn.Conv2d if conv_dim == 2 else nn.Conv1d
        norm_layer = nn.BatchNorm2d if conv_dim == 2 else nn.BatchNorm1d

        self.conv = conv_layer(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.batch_norm = norm_layer(out_channels)
        self.relu = nn.ReLU() if relu else nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.batch_norm(x)
        x = self.relu(x)
        return x


class ConvTransposeBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
        output_padding: int,
        conv_dim: int = 2,
        relu: bool = True,
    ):
        super().__init__()

        conv_layer = nn.ConvTranspose2d if conv_dim == 2 else nn.ConvTranspose1d
        norm_layer = nn.BatchNorm2d if conv_dim == 2 else nn.BatchNorm1d

        self.conv_transpose = conv_layer(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding
        )
        self.batch_norm = norm_layer(out_channels)
        self.relu = nn.ReLU() if relu else nn.Identity()

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.batch_norm(x)
        x = self.relu(x)
        return x


class ResBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
        conv_dim: int = 2,
    ):
        super().__init__()

        self.conv1 = ConvBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            conv_dim=conv_dim,
            relu=True,
        )

        self.conv2 = ConvBlock(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            conv_dim=conv_dim,
            relu=False,
        )

        self.shortcut = nn.Identity()
        if in_channels != out_channels:
            self.shortcut = ConvBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=stride,
                padding=0,
                conv_dim=conv_dim,
                relu=False,
            )

    def forward(self, x):
        identity = self.shortcut(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = x + identity
        x = nn.ReLU()(x)
        return x


class ResTransposeBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
        output_padding: int,
        conv_dim: int = 2,
    ):
        super().__init__()

        self.conv_transpose1 = ConvTransposeBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            conv_dim=conv_dim,
            relu=True,
        )

        self.conv_transpose2 = ConvTransposeBlock(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            conv_dim=conv_dim,
            relu=False,
        )

        self.shortcut = nn.Identity()
        if in_channels != out_channels:
            self.shortcut = ConvTransposeBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=stride,
                padding=0,
                output_padding=0,
                conv_dim=conv_dim,
                relu=False,
            )

    def forward(self, x):
        identity = self.shortcut(x)
        x = self.conv_transpose1(x)
        x = self.conv_transpose2(x)
        x = x + identity
        x = nn.ReLU()(x)
        return x
