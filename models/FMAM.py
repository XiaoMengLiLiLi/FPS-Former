import torch
import torch.nn as nn
from typing import Tuple
from einops import rearrange
from einops.layers.torch import Rearrange
from torch.nn import functional as F
import numpy as np
import cv2
import math
from einops import rearrange
from functools import reduce
from operator import __add__

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FMAM(nn.Module):
    def __init__(self, in_channels=64, pyramid_levels=3):
        """
        Constructs a Laplacian pyramid from an input tensor.

        Args:
            in_channels    (int): Number of input channels.
            pyramid_levels (int): Number of pyramid levels.

        Input:
            x : (B, in_channels, H, W)
        Output:
            Fused frequency attention map : (B, in_channels, in_channels)
        """
        super().__init__()
        self.in_channels = in_channels
        self.pyramid_levels = pyramid_levels
        sigma = 1.6
        s_value = 2 ** (1 / 3)

        self.sigma_kernels = [
            self.get_gaussian_kernel(2 * i + 3, sigma * s_value ** i)
            for i in range(pyramid_levels)
        ]

    def get_gaussian_kernel(self, kernel_size, sigma):
        kernel_weights = cv2.getGaussianKernel(ksize=kernel_size, sigma=sigma)
        kernel_weights = kernel_weights * kernel_weights.T
        kernel_weights = np.repeat(kernel_weights[None, ...], self.in_channels, axis=0)[:, None, ...]

        return torch.from_numpy(kernel_weights).float().to(device)

    def forward(self, x):
        G = x

        # Level 1
        L0 = Rearrange('b d h w -> b d (h w)')(G)
        L0_att = F.softmax(L0, dim=2) @ L0.transpose(1, 2)  # L_k * L_v
        L0_att = F.softmax(L0_att, dim=-1)

        # Next Levels
        attention_maps = [L0_att]
        pyramid = [G]

        for kernel in self.sigma_kernels:
            # ---------------------------------------------------------------------------------------------------------------Programming implementation of Pytorch conv2d(padding='same' ---Xiaomeng-----------------------------------------------------)
            kernel_size = kernel.shape[-1]
            k_new = (kernel_size, kernel_size)
            conv_padding = reduce(__add__, [(k // 2 + (k - 2 * (k // 2)) - 1, k // 2) for k in k_new[::-1]])
            pad = nn.ZeroPad2d(conv_padding)
            G = pad(G)
            G = F.conv2d(input=G, weight=kernel, bias=None, groups=self.in_channels)
            # ---------------------------------------------------------------------------------------------------------------Programming implementation of Pytorch conv2d(padding='same' ---Xiaomeng-----------------------------------------------------)
            pyramid.append(G)

        for i in range(1, self.pyramid_levels):
            L = torch.sub(pyramid[i - 1], pyramid[i])
            L = Rearrange('b d h w -> b d (h w)')(L)
            L_att = F.softmax(L, dim=2) @ L.transpose(1, 2)
            attention_maps.append(L_att)

        return sum(attention_maps)