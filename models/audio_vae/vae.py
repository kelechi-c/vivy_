import torch
from torch import nn


# snake actviation
@torch.jit.script
def snake(x: torch.Tensor, alpha):
    x_shape = x.shape
    min_add = 1e-9

    x = x.reshape(x_shape[0], x_shape[1], -1)
    x = x + (alpha + min_add).reciprocal() * torch.sin(alpha * x).pow(2)
    x = x.reshape(x_shape)

    return x


class SnakeBlock(nn.Module):
    # snake activation to maintain the periodictiy of the music/audio signals
    def __init__(self, channels):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, channels, 1))

    def forward(self, input_x: torch.Tensor):
        output = snake(input_x, self.alpha)

        return output


# residual block
class ResLayer(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.reslayer = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=3), SnakeBlock(channels)
        )

    def forward(self, x_tensor: torch.Tensor) -> torch.Tensor:
        res_copy = x_tensor

        x = self.reslayer(x_tensor)

        return x + res_copy


# downsampling block
class Downsampler(nn.Module):
    def __init__(self, out_channels):
        self.down_layer = nn.Sequential(
            ResLayer(),
            nn.Conv1d(1, 64, kernel_size=3, dilation=1),
            nn.SiLU(),
            nn.Conv1d(64, 128, kernel_size=6, dilation=3),
        )

    def forward(self, x_wave: torch.Tensor):
        x_wave = self.downsampler(x_wave)

        return x_wave


# encoder/downsampler
class Encoder(nn.Module):
    def __init__(self, resolutions=[1, 2, 4, 8]):
        layers = []
        for res in range(resolutions):
            layers +=
        self.downsampler = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3, dilation=1),
            nn.SiLU(),
            nn.Conv1d(64, 128, kernel_size=6, dilation=3),
        )

    def forward(self, x_wave: torch.Tensor):
        x_wave = self.downsampler(x_wave)

        return x_wave


class MusicAutoEncoder(nn.Module):
    """
    This is a fully convolutional variatonal autoencoder
    """

    def __init__(
        self, in_channels, out_channels, downratio, latent_dim, config, pretransform
    ):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.downsample_ratio = downratio
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.config = config
        self.pretransform = pretransform
        self.latent_dim = latent_dim

    def encode(self, audio):
        output = self.encoder(audio)
        return output

    def decode(self, audio):
        output = self.decoder(audio)

        return output

    def forward(self, input):
        return input
