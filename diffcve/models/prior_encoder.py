import torch

import torch.nn as nn

class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(
            in_ch, out_ch,
            kernel_size=3,
            stride=2,
            padding=1
        )

    def forward(self, x):
        return self.conv(x)

class ResidualConvBlock(nn.Module):
    def __init__(self, ch, use_residual=True):
        super().__init__()
        self.use_residual = use_residual

        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(max(1, ch//8), ch)
        self.act = nn.SiLU()

    def forward(self, x):
        identity = x

        out = self.norm1(x)
        out = self.act(out)
        out = self.conv1(out)

        if self.use_residual:
            out = out + identity

        return out

class ResidualMotionEncoder(nn.Module):
    def __init__(
        self,
        res_in_channels=1,
        mv_in_channels=2,
        res_initial_ch=12,
        mv_initial_ch=20,
        output_ch=256,
        use_residual=True,
    ):
        super().__init__()
        self.use_residual = use_residual

        self.res_conv = nn.Conv2d(res_in_channels, res_initial_ch, 3, 1, 1)
        self.mv_conv = nn.Conv2d(mv_in_channels, mv_initial_ch, 3, 1, 1)

        self.act = nn.SiLU()

        merged_ch = res_initial_ch + mv_initial_ch

        self.merge_conv = nn.Conv2d(merged_ch, merged_ch, 3, 1, 1)
        self.merge_norm = nn.GroupNorm(max(1, merged_ch//8), merged_ch)

        self.down1 = DownBlock(merged_ch, merged_ch * 2)
        self.conv1 = ResidualConvBlock(merged_ch * 2, use_residual)

        self.down2 = DownBlock(merged_ch * 2, merged_ch * 4)
        self.conv2 = ResidualConvBlock(merged_ch * 4, use_residual)

        self.down3 = DownBlock(merged_ch * 4, merged_ch * 8)
        self.conv3 = ResidualConvBlock(output_ch, use_residual)


    def forward(self, res, mv, return_features=False):
        h_res = self.res_conv(res)
        h_mv = self.mv_conv(mv)

        h = torch.cat([h_res, h_mv], dim=1)
        h = self.merge_norm(h)
        h = self.act(h)
        h = self.merge_conv(h)

        fea_list = []

        h = self.down1(h)
        h = self.conv1(h)
        if return_features:
            fea_list.append(h)

        h = self.down2(h)
        h = self.conv2(h)
        if return_features:
            fea_list.append(h)

        h = self.down3(h)
        h = self.conv3(h)

        if return_features:
            return h, fea_list

        return h
