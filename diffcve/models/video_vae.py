import torch
from torch import nn
from diffcve.modules.vae_blocks import Encoder, PriorConditionedVideoDecoder
from diffcve.modules.distributions import DiagonalGaussianDistribution


class VideoAutoencoder(nn.Module):
    def __init__(self, ddconfig, embed_dim, fusion_w=1.0):
        super().__init__()
        self.encoder = Encoder(**ddconfig)
        self.decoder = PriorConditionedVideoDecoder(**ddconfig)
        self.decoder.fusion_w = fusion_w
        self.quant_conv = nn.Conv2d(2 * ddconfig["z_channels"], 2 * embed_dim, 1)
        self.post_quant_conv = nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        self.embed_dim = embed_dim

    def encode(self, x, return_features=True):
        if return_features:
            h, encoder_features = self.encoder(x, return_features=True)
        else:
            h = self.encoder(x)
        posterior = DiagonalGaussianDistribution(self.quant_conv(h))
        return (posterior, encoder_features) if return_features else posterior

    def decode(self, z, encoder_features, prior_features, qp):
        return self.decoder(self.post_quant_conv(z), encoder_features, prior_features, qp)
