# https://github.com/openai/improved-diffusion/blob/main/improved_diffusion/gaussian_diffusion.py

# https://github.com/lucidrains/denoising-diffusion-pytorch/blob/7706bdfc6f527f58d33f84b7b522e61e6e3164b3/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py

# https://github.com/openai/guided-diffusion/blob/0ba878e517b276c45d1195eb29f6f5f72659a05b/guided_diffusion/nn.py


import math

import torch

import torch.nn as nn

import numpy as np

from einops import repeat, rearrange

def make_beta_schedule(schedule, n_timestep, linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
    if schedule == "linear":
        betas = (
                torch.linspace(linear_start ** 0.5, linear_end ** 0.5, n_timestep, dtype=torch.float64) ** 2
        )

    elif schedule == "cosine":
        timesteps = (
                torch.arange(n_timestep + 1, dtype=torch.float64) / n_timestep + cosine_s
        )
        alphas = timesteps / (1 + cosine_s) * np.pi / 2
        alphas = torch.cos(alphas).pow(2)
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = np.clip(betas, a_min=0, a_max=0.999)

    elif schedule == "sqrt_linear":
        betas = torch.linspace(linear_start, linear_end, n_timestep, dtype=torch.float64)
    elif schedule == "sqrt":
        betas = torch.linspace(linear_start, linear_end, n_timestep, dtype=torch.float64) ** 0.5
    else:
        raise ValueError(f"schedule '{schedule}' unknown.")
    return betas.numpy()

def extract_into_tensor(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def checkpoint(func, inputs, params, flag):
    return func(*inputs)

def timestep_embedding(timesteps, dim, max_period=10000, repeat_only=False):
    if not repeat_only:
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=timesteps.device)
        args = timesteps[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    else:
        embedding = repeat(timesteps, 'b -> b d', d=dim)
    return embedding

def zero_module(module):
    for p in module.parameters():
        p.detach().zero_()
    return module

def normalization(channels, norm_channel=32):
    return GroupNorm32(norm_channel, channels)

class GroupNorm32(nn.GroupNorm):
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)

def conv_nd(dims, *args, **kwargs):
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")

def linear(*args, **kwargs):
    return nn.Linear(*args, **kwargs)

def avg_pool_nd(dims, *args, **kwargs):
    if dims == 1:
        return nn.AvgPool1d(*args, **kwargs)
    elif dims == 2:
        return nn.AvgPool2d(*args, **kwargs)
    elif dims == 3:
        return nn.AvgPool3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")

def from_4d_to_5d(inp, b, c, t, h, w):
    out = rearrange(inp, '(b t) c h w -> b c t h w', b=b, c=c, t=t, h=h, w=w)
    return out

def from_5d_to_4d(inp, b, c, t, h, w):
    out = rearrange(inp, 'b c t h w -> (b t) c h w')
    return out

def from_4d_to_3d(inp, b, c, t, h, w):
    out = rearrange(inp, '(b t) c h w -> (b h w) t c', b=b, c=c, t=t, h=h, w=w)
    return out

def from_3d_to_4d(inp, b, c, t, h, w):
    out = rearrange(inp, '(b h w) t c -> (b t) c h w', b=b, c=c, t=t, h=h, w=w)
    return out

class SpatialTemporalConv(nn.Module):
    def __init__(self, num_feat, num_frames=1):
        super().__init__()

        self.num_frames = num_frames


        self.temporal_conv = conv_nd(3, num_feat, num_feat, (3, 1, 1), padding=(1, 0, 0))

        self.temporal_alpha_logit = nn.Parameter(torch.zeros(1))

    def forward(self, inp, t=None):
        bt, c, h, w = inp.shape
        b = bt // t if t else bt // self.num_frames
        ori = inp
        inp = from_4d_to_5d(inp, b, c, t, h, w)
        res = self.temporal_conv(inp)
        res = from_5d_to_4d(res, b, c, t, h, w)
        alpha = torch.sigmoid(self.temporal_alpha_logit).to(inp.dtype)
        out = alpha * res + (1-alpha) *ori


        return out
