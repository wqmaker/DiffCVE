from functools import partial

import numpy as np
import torch
from torch import nn
from peft import LoraConfig, get_peft_model
from tqdm import tqdm

from diffcve.utils import default, exists, instantiate_from_config
from diffcve.modules.diffusion.utils import make_beta_schedule, extract_into_tensor
from diffcve.modules.distributions import DiagonalGaussianDistribution
from diffcve.modules.text_encoder import load_text_conditions


class ConditionedDenoiser(nn.Module):
    def __init__(self, network_config, conditioning_key="crossattn"):
        super().__init__()
        if conditioning_key != "crossattn":
            raise ValueError("DiffCVE requires cross-attention conditioning")
        self.conditioning_key = conditioning_key
        self.network = instantiate_from_config(network_config)

    def forward(self, x, t, c_crossattn, struct_cond=None, prior_cond=None):
        return self.network(
            x, t, context=torch.cat(c_crossattn, 1),
            struct_cond=struct_cond, prior_cond=prior_cond,
        )


class VideoDiffusion(nn.Module):
    def __init__(self, denoiser_config, prior_encoder_config, prior_condition_config,
                 structure_encoder_config, timesteps=1000, linear_start=0.00085,
                 linear_end=0.012, num_frames=5, scale_factor=0.18215,
                 conditioning_key="crossattn", parameterization="eps",
                 text_conditions_path="checkpoints/text_conditions.pt",
                 cond_stage_config=None, text_prompts=None, text_condition_qps=None):
        super().__init__()
        if parameterization != "eps":
            raise ValueError("DiffCVE requires noise prediction (eps)")
        self.parameterization = parameterization
        self.num_frames = num_frames
        self.scale_factor = scale_factor
        self.v_posterior = 0.0
        self.denoiser = ConditionedDenoiser(denoiser_config, conditioning_key)
        self.prior_encoder = instantiate_from_config(prior_encoder_config)
        self.prior_condition_encoder = instantiate_from_config(prior_condition_config)
        self.structure_encoder = instantiate_from_config(structure_encoder_config)
        self.add_lora_adapters()
        self.text_conditions = load_text_conditions(
            text_conditions_path, cond_stage_config, text_prompts, text_condition_qps,
        )
        self.register_schedule(timesteps=timesteps, linear_start=linear_start, linear_end=linear_end)

    @property
    def device(self):
        return self.betas.device

    def add_lora_adapters(self):
        config = LoraConfig(
            r=32, init_lora_weights="gaussian",
            target_modules=["attn1.to_k", "attn1.to_q", "attn1.to_v", "attn1.to_out.0",
                            "attn2.to_k", "attn2.to_q", "attn2.to_v", "attn2.to_out.0",
                            "conv", "conv1", "conv2", "conv_shortcut", "conv_out",
                            "proj_in", "proj_out", "ff.net.2", "ff.net.0.proj"],
        )
        self.denoiser = get_peft_model(self.denoiser, config)

    def register_schedule(self, given_betas=None, beta_schedule="linear", timesteps=1000,
                          linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
        if exists(given_betas):
            betas = given_betas
        else:
            betas = make_beta_schedule(beta_schedule, timesteps, linear_start=linear_start, linear_end=linear_end,
                                       cosine_s=cosine_s)
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.linear_start = linear_start
        self.linear_end = linear_end
        assert alphas_cumprod.shape[0] == self.num_timesteps, 'alphas have to be defined for each timestep'

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev))


        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod - 1)))


        posterior_variance = (1 - self.v_posterior) * betas * (1. - alphas_cumprod_prev) / (
                    1. - alphas_cumprod) + self.v_posterior * betas

        self.register_buffer('posterior_variance', to_torch(posterior_variance))

        self.register_buffer('posterior_log_variance_clipped', to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))

    def predict_start_from_noise(self, x_t, t, noise):
        return (
                extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
                extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract_into_tensor(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        return (extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise)

    def add_noise_with_schedule(self, x_start, t, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        return (extract_into_tensor(sqrt_alphas_cumprod.to(noise.device), t, x_start.shape) * x_start +
                extract_into_tensor(sqrt_one_minus_alphas_cumprod.to(noise.device), t, x_start.shape) * noise)

    def sample_latent(self, encoder_posterior):
        if isinstance(encoder_posterior, DiagonalGaussianDistribution):
            z = encoder_posterior.sample()
        elif isinstance(encoder_posterior, torch.Tensor):
            z = encoder_posterior
        else:
            raise TypeError(type(encoder_posterior))
        return self.scale_factor * z

    def apply_model(self, x_noisy, t, cond, struct_cond, diff_prior_c):
        if not isinstance(cond, dict):
            cond = {"c_crossattn": cond if isinstance(cond, list) else [cond]}
        return self.denoiser(x_noisy, t, **cond, struct_cond=struct_cond, prior_cond=diff_prior_c)

    def predict_noise(self, x, t, cond, struct_cond, prior):
        cond_prior, diffusion_prior = self.prior_condition_encoder(prior, t)
        struct_features = self.structure_encoder(struct_cond, t, cond_prior)
        return self.apply_model(x, t, cond, struct_features, diffusion_prior)


class MultiStepVideoEnhancer(VideoDiffusion):
    @torch.no_grad()
    def sample(self, cond, struct_cond, prior, x_T, timesteps=50, verbose=True):
        if not 2 <= timesteps <= len(self.betas):
            raise ValueError("Multi-step inference requires at least two respaced timesteps")
        img = x_T
        cond = cond[:img.shape[0]]
        iterator = reversed(range(timesteps))
        if verbose:
            iterator = tqdm(iterator, desc="Sampling t", total=timesteps)
        for i in iterator:
            ts = torch.full((img.shape[0],), i, device=self.device, dtype=torch.long)
            original_t = torch.full_like(ts, self.original_timesteps[i])
            noise_pred = self.predict_noise(img, original_t, cond, struct_cond, prior)
            x_recon = self.predict_start_from_noise(img, ts, noise_pred)
            mean, _, log_variance = self.q_posterior(x_recon, img, ts)
            noise = torch.randn(img.shape, device=img.device)
            nonzero_mask = (1 - (ts == 0).float()).reshape(-1, 1, 1, 1)
            img = mean + nonzero_mask * (0.5 * log_variance).exp() * noise
        return img


class SingleStepVideoEnhancer(VideoDiffusion):
    def __init__(self, vae_config, **kwargs):
        super().__init__(**kwargs)
        self.vae = instantiate_from_config(vae_config)

    @torch.no_grad()
    def sample(self, cond, struct_cond, prior, x_T, timesteps=1, verbose=True):
        if timesteps != 1:
            raise ValueError("The distilled checkpoint supports exactly one sampling step")
        ts = torch.full((x_T.shape[0],), 99, device=self.device, dtype=torch.long)
        noise_pred = self.predict_noise(x_T, ts, cond[:x_T.shape[0]], struct_cond, prior)
        return self.predict_start_from_noise(x_T, ts, noise_pred)
