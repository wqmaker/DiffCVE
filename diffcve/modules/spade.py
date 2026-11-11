"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import re

import torch

import torch.nn as nn

from diffcve.modules.diffusion.utils import normalization

class SPADE(nn.Module):
    def __init__(self, norm_nc, label_nc, config_text='spadeinstance3x3'):
        super().__init__()

        assert config_text.startswith('spade')
        parsed = re.search(r'spade(\D+)(\d)x\d', config_text)
        param_free_norm_type = str(parsed.group(1))
        ks = int(parsed.group(2))

        self.param_free_norm = normalization(norm_nc)


        nhidden = 128

        pw = ks // 2
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(label_nc, nhidden, kernel_size=ks, padding=pw),
            nn.ReLU()
        )
        self.mlp_gamma = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)
        self.mlp_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)

    def forward(self, x_dic, segmap_dic, size=None):

        if size is None:


            key = str(int(x_dic.shape[-1])) if torch.onnx.is_in_onnx_export() else str(x_dic.size(-1))
            segmap = segmap_dic[key]
            x = x_dic
        else:
            x = x_dic[str(size)]
            segmap = segmap_dic[str(size)]


        normalized = self.param_free_norm(x)


        actv = self.mlp_shared(segmap)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)


        out = normalized * (1 + gamma) + beta

        return out
