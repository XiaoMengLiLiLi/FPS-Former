import torch.nn as nn
import torch
import torch.nn.functional as F
from einops import rearrange
from torch import einsum
import math
import warnings
from torch.nn.init import _calculate_fan_in_and_fan_out
from collections import defaultdict, Counter
import numpy as np
from tqdm import tqdm
import random
from models.FMAM import FMAM


def uniform(a, b, shape, device='cuda'):
    return (b - a) * torch.rand(shape, device=device) + a


class AsymmetricTransform:

    def Q(self, *args, **kwargs):
        raise NotImplementedError('Query transform not implemented')

    def K(self, *args, **kwargs):
        raise NotImplementedError('Key transform not implemented')


class LSH:
    def __call__(self, *args, **kwargs):
        raise NotImplementedError('LSH scheme not implemented')

    def compute_hash_agreement(self, q_hash, k_hash):
        return (q_hash == k_hash).min(dim=-1)[0].sum(dim=-1)


class XBOXPLUS(AsymmetricTransform):

    def set_norms(self, x):
        self.x_norms = x.norm(p=2, dim=-1, keepdim=True)
        self.MX = torch.amax(self.x_norms, dim=-2, keepdim=True)

    def X(self, x):
        device = x.device
        ext = torch.sqrt((self.MX ** 2).to(device) - (self.x_norms ** 2).to(device))
        zero = torch.tensor(0.0, device=x.device).repeat(x.shape[:-1], 1).unsqueeze(-1)
        return torch.cat((x, ext, zero), -1)


def lsh_clustering(x, n_rounds, r=1):
    salsh = SALSH(n_rounds=n_rounds, dim=x.shape[-1], r=r, device=x.device)
    x_hashed = salsh(x).reshape((n_rounds,) + x.shape[:-1])
    return x_hashed.argsort(dim=-1)


class SALSH(LSH):
    def __init__(self, n_rounds, dim, r, device='cuda'):
        super(SALSH, self).__init__()
        self.alpha = torch.normal(0, 1, (dim, n_rounds), device=device)
        self.beta = uniform(0, r, shape=(1, n_rounds), device=device)
        self.dim = dim
        self.r = r

    def __call__(self, vecs):
        projection = vecs @ self.alpha
        projection_shift = projection + self.beta
        projection_rescale = projection_shift / self.r
        return projection_rescale.permute(2, 0, 1)


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


def variance_scaling_(tensor, scale=1.0, mode='fan_in', distribution='normal'):
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    if mode == 'fan_in':
        denom = fan_in
    elif mode == 'fan_out':
        denom = fan_out
    elif mode == 'fan_avg':
        denom = (fan_in + fan_out) / 2

    variance = scale / denom

    if distribution == "truncated_normal":
        trunc_normal_(tensor, std=math.sqrt(variance) / .87962566103423978)
    elif distribution == "normal":
        tensor.normal_(std=math.sqrt(variance))
    elif distribution == "uniform":
        bound = math.sqrt(3 * variance)
        tensor.uniform_(-bound, bound)
    else:
        raise ValueError(f"invalid distribution {distribution}")


def lecun_normal_(tensor):
    variance_scaling_(tensor, mode='fan_in', distribution='truncated_normal')


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, *args, **kwargs):
        x = self.norm(x)
        return self.fn(x, *args, **kwargs)


class GELU(nn.Module):
    def forward(self, x):
        return F.gelu(x)


def batch_scatter(output, src, dim, index):
    """
    :param output: [b,n,c]
    :param src: [b,n,c]
    :param dim: int
    :param index: [b,n]
    :return: output: [b,n,c]
    """
    b, k, c = src.shape
    index = index[:, :, None].expand(-1, -1, c)
    output, src, index = map(lambda t: rearrange(t, 'b k c -> (b c) k'), (output, src, index))
    output.scatter_(dim, index, src)
    output = rearrange(output, '(b c) k -> b k c', b=b)
    return output


def batch_gather(x, index, dim):
    """
    :param x: [b,n,c]
    :param index: [b,n//2]
    :param dim: int
    :return: output: [b,n//2,c]
    """
    b, n, c = x.shape
    index = index[:, :, None].expand(-1, -1, c)
    x, index = map(lambda t: rearrange(t, 'b n c -> (b c) n'), (x, index))
    output = torch.gather(x, dim, index)
    output = rearrange(output, '(b c) n -> b n c', b=b)
    return output


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim * mult, 1, 1, bias=False),
            GELU(),
            nn.Conv2d(dim * mult, dim * mult, 3, 1, 1, bias=False, groups=dim * mult),
            GELU(),
            nn.Conv2d(dim * mult, dim, 1, 1, bias=False),
        )

    def forward(self, x):
        """
        x: [b,h,w,c]
        return out: [b,h,w,c]
        """
        out = self.net(x.permute(0, 3, 1, 2))
        return out.permute(0, 2, 3, 1)


class FPSAttn(nn.Module):
    def __init__(self, heads=4, n_rounds=2, channels=64, patch_size=(8,8), r=1, pyramid_levels=3):
        super(FPSAttn, self).__init__()
        self.heads = heads
        self.n_rounds = n_rounds
        inner_dim = channels * 3
        self.to_q = nn.Linear(channels, inner_dim, bias=False)
        self.to_k = nn.Linear(channels, inner_dim, bias=False)
        self.to_v = nn.Linear(channels, inner_dim, bias=False)
        self.to_out = nn.Linear(inner_dim, channels, bias=False)

        self.xbox_plus = XBOXPLUS()
        self.clustering_params = {
            'r': r,
            'n_rounds': self.n_rounds
        }
        self.q_attn_size = patch_size[0] * patch_size[1]
        self.k_attn_size = patch_size[0] * patch_size[1]

        self.patch_size = patch_size

        self.FMAM = FMAM(in_channels=channels, pyramid_levels=pyramid_levels)
        self.queries = nn.Conv2d(channels, channels, 1)
        self.conv_dw = nn.Conv3d(channels, channels, kernel_size=(2, 1, 1), bias=False, groups=channels)

    def forward(self, input):
        """
        :param input: [b,n,c]
        :return: output: [b,n,c]
        """
        # --------------------------------------------------------------------------------------------------------Spatial attention-------------------------------------------------------------------------------------------------------------------------------------------------

        w_size = self.patch_size
        input = input.permute(0,2,3,1)
        input_inti = input

        b, h, w, c = input.shape
        input = rearrange(input, 'b (nh hh) (nw ww) c-> b (nh nw) (hh ww c)', hh=w_size[0], ww=w_size[1])
        N_input = input.shape[1]
        input = input.view(b * N_input, -1, c)

        B, N, C_inp = input.shape
        query = self.to_q(input)
        key = self.to_k(input)
        value = self.to_v(input)
        input_hash = input.view(B, N, self.heads, C_inp // self.heads)
        x_hash = rearrange(input_hash, 'b t h e -> (b h) t e')
        bs, x_seqlen, dim = x_hash.shape
        with torch.no_grad():
            self.xbox_plus.set_norms(x_hash)
            Xs = self.xbox_plus.X(x_hash)
            x_positions = lsh_clustering(Xs, **self.clustering_params)
            x_positions = x_positions.reshape(self.n_rounds, bs, -1)

        del Xs

        C = query.shape[-1]
        query = query.view(B, N, self.heads, C // self.heads)
        key = key.view(B, N, self.heads, C // self.heads)
        value = value.view(B, N, self.heads, C // self.heads)

        query = rearrange(query, 'b t h e -> (b h) t e')  # [bs, q_seqlen,c]
        key = rearrange(key, 'b t h e -> (b h) t e')
        value = rearrange(value, 'b s h d -> (b h) s d')

        bs, q_seqlen, dim = query.shape
        bs, k_seqlen, dim = key.shape
        v_dim = value.shape[-1]

        x_rev_positions = torch.argsort(x_positions, dim=-1)
        x_offset = torch.arange(bs, device=query.device).unsqueeze(-1) * x_seqlen
        x_flat = (x_positions + x_offset).reshape(-1)

        s_queries = query.reshape(-1, dim).index_select(0, x_flat).reshape(-1, self.q_attn_size, dim)
        s_keys = key.reshape(-1, dim).index_select(0, x_flat).reshape(-1, self.k_attn_size, dim)
        s_values = value.reshape(-1, v_dim).index_select(0, x_flat).reshape(-1, self.k_attn_size, v_dim)

        inner = s_queries @ s_keys.transpose(2, 1)
        norm_factor = 1
        inner = inner / norm_factor

        # free memory
        del x_positions

        # softmax denominator
        dots_logsumexp = torch.logsumexp(inner, dim=-1, keepdim=True)
        # softmax
        dots = torch.exp(inner - dots_logsumexp)
        # dropout

        # n_rounds outs
        bo = (dots @ s_values).reshape(self.n_rounds, bs, q_seqlen, -1)

        # undo sort
        x_offset = torch.arange(bs * self.n_rounds, device=query.device).unsqueeze(-1) * x_seqlen
        x_rev_flat = (x_rev_positions.reshape(-1, x_seqlen) + x_offset).reshape(-1)
        o = bo.reshape(-1, v_dim).index_select(0, x_rev_flat).reshape(self.n_rounds, bs, q_seqlen, -1)

        slogits = dots_logsumexp.reshape(self.n_rounds, bs, -1)
        logits = torch.gather(slogits, 2, x_rev_positions)

        # free memory
        del x_rev_positions

        # weighted sum multi-round attention
        probs = torch.exp(logits - torch.logsumexp(logits, dim=0, keepdim=True))
        out = torch.sum(o * probs.unsqueeze(-1), dim=0)
        out = rearrange(out, '(b h) t d -> b t h d', h=self.heads)
        out = out.reshape(B, N, -1)
        out = self.to_out(out)

        out = out.view(b, N_input, -1)
        F_spa = rearrange(out, 'b (nh nw) (hh ww c) -> b (nh hh) (nw ww) c', nh=h // (w_size[0]), hh=w_size[0], ww=w_size[1])

        # out = out.permute(0, 3, 1, 2)


        # --------------------------------------------------------------------------------------------------Frequency attention-------------------------------------------------------------------------------------------------------------------------------------------
        x = input_inti
        b, h, w, c = x.shape
        x_fre = x.permute(0,3,1,2)
        freq_context = self.FMAM(x_fre)
        queries = F.softmax(self.queries(x_fre).reshape(b, c, h * w), dim=1)
        freq_attention = (freq_context.transpose(1, 2) @ queries).reshape(b,c,h,w)

        # Attention Aggregation: Efficient Frequency Attention (EF-Att) Block
        spa_attention = F_spa.permute(0, 3, 1, 2)
        attention = torch.cat([spa_attention[:, :, None, ...], freq_attention[:, :, None, ...]], dim=2)
        out = self.conv_dw(attention)[:, :, 0, ...]
        return out