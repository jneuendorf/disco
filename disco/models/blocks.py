"""Common blocks used in models."""
from collections.abc import Callable, Sequence

import numpy as np
from torch import nn


class Encoder(nn.Module):
    """A simple encoder model."""

    def __init__(
        self,
        channels: Sequence[int],
        in_channels: int = 1,
    ) -> None:
        """Initialize the encoder.
        Args:
            in_channels: The number of input channels.
            channels: The number of channels in each hidden layer.
        Returns:
            None
        """
        super().__init__()
        self.channels = channels
        channels = [in_channels, *channels]
        module = [
            nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                ),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(),
            )
            for in_channels, out_channels in zip(channels[:-1], channels[1:])
        ]
        self.encoder = nn.Sequential(*module)

    def forward(self, x):
        """Forward pass of the encoder."""
        x = self.encoder(x)
        return x

    def out_size(self, img_size):
        """Return the output size of the encoder for a given image size."""
        out_img_size = img_size // 2 ** len(self.channels)
        return out_img_size ** 2 * self.channels[-1]


class Decoder(nn.Module):
    """A simple decoder model."""

    def __init__(
        self,
        channels: Sequence[int],
        out_channels: int = 1,
        create_conv: Callable[[], nn.Conv2d] | None = None,
    ) -> None:
        """Args:
            channels: The number of channels in each hidden layer.
            out_channels: The number of output channels.
            create_conv: A function that creates the final convolutional layer
                for potentially resizing the output.
        """
        super().__init__()

        self.last_conv = (
            nn.Conv2d(
                channels[-1],
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
            )
            if create_conv is None
            else create_conv()
        )

        modules = [
            nn.Sequential(
                nn.ConvTranspose2d(
                    in_channels,
                    out_channels,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                ),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(),
            )
            for in_channels, out_channels in zip(channels[:-1], channels[1:])
        ]
        modules.append(
            nn.Sequential(
                nn.ConvTranspose2d(
                    channels[-1],
                    channels[-1],
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                ),
                nn.BatchNorm2d(channels[-1]),
                nn.LeakyReLU(),
                self.last_conv,
                nn.Sigmoid(),
            )
        )
        self.model = nn.Sequential(*modules)

    def forward(self, x):
        """Forward pass of the decoder."""
        x = self.model(x)
        return x

    def print_convolutions(
        self,
        channels: Sequence[int],
        out_channels: int,
        input_img_size: tuple[int, int],
    ):
        print("DECODER:")
        h_in, w_in = input_img_size
        for c_in, c_out in zip(channels[:-1], channels[1:]):
            h_out, w_out = conv_transpose_size(
                h_in, w_in,
                kernel_size=3, stride=2, padding=1, output_padding=1,
            )
            print(f"{c_in, h_in, w_in} -> {c_out, h_out, w_out}")
            h_in, w_in = h_out, w_out

        # last ConvTranspose2d
        h_out, w_out = conv_transpose_size(
            h_in, w_in,
            kernel_size=3, stride=2, padding=1, output_padding=1,
        )
        print(f"{channels[-1], h_in, w_in} -> {channels[-1], h_out, w_out}")
        h_in, w_in = h_out, w_out

        # last Conv2d
        assert self.last_conv.kernel_size[0] == self.last_conv.kernel_size[1]
        assert self.last_conv.padding[0] == self.last_conv.padding[1]
        h_out, w_out = conv_size(
            h_in, w_in,
            kernel_size=self.last_conv.kernel_size[0],
            padding=self.last_conv.padding[0],
        )
        print(f"{channels[-1], h_in, w_in} -> {out_channels, h_out, w_out}")


# -------------------------------------------------------------------
# HELPER FUNCTIONS

def conv_size(h_in, w_in, kernel_size,
              stride=1, dilation=1, padding=0):
    x = 2 * padding - dilation * (kernel_size - 1) - 1
    return np.array([
        (h_in + x) / stride + 1,
        (w_in + x) / stride + 1,
    ]).astype(int)


def conv_transpose_size(h_in, w_in, kernel_size,
               padding=0, dilation=1, stride=1, output_padding=0):
    x = -2 * padding + dilation * (kernel_size - 1) + output_padding + 1
    return np.array([
        (h_in - 1) * stride + x,
        (w_in - 1) * stride + x,
    ])
