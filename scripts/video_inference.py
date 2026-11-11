import argparse
import json
import math
import os
import random
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from omegaconf import OmegaConf
from torch.nn import functional as F

from diffcve.utils import instantiate_from_config, load_checkpoint, resolve_path
from diffcve.modules.text_encoder import qp_interpolation_weights
from scripts.tiling import build_patch_grid, merge_patch_grid
from scripts.color_correction import adaptive_instance_normalization, wavelet_reconstruction


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args(single_step):
    parser = argparse.ArgumentParser(description='DiffCVE video enhancement inference')
    config = 'single_step.yaml' if single_step else 'multi_step.yaml'
    parser.add_argument('--config', default=f'configs/{config}')
    parser.add_argument('--checkpoint', default='checkpoints/single_step.ckpt' if single_step else 'checkpoints/multi_step.ckpt')
    parser.add_argument('--vae-checkpoint', default='checkpoints/video_vae.ckpt', help='External VAE checkpoint (multi-step only)')
    parser.add_argument('--vae-config', default='configs/video_vae.yaml')
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--bitstream', help='Encoded video bitstream')
    source.add_argument('--bitstream-list', help='JSON list of bitstream jobs with bitstream, qp, fps, and optional name')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--codec-dir', default=str(resolve_path('codec')), help='Directory containing the streaming decoder')
    parser.add_argument('--ffmpeg', default='ffmpeg')
    parser.add_argument('--rgb-buffer-frames', type=int, default=5, help='Maximum queued RGB frames; positive integer')
    parser.add_argument('--prior-buffer-frames', type=int, default=32, help='Maximum queued MV/residual pairs; at least 32 for decoder lookahead')
    parser.add_argument('--qp', type=int, help='Expected bitstream QP')
    parser.add_argument('--fps', type=float, help='Bitstream frame rate')
    parser.add_argument('--sequence-name', help='Single-bitstream output name')
    parser.add_argument('--steps', type=int, default=1 if single_step else 50)
    parser.add_argument('--frames-per-segment', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--precision', choices=['full', 'autocast'], default='autocast')
    parser.add_argument('--device', default='cuda', help='Device within CUDA_VISIBLE_DEVICES, e.g. cuda:0')
    parser.add_argument('--worker-index', type=int, default=0, help='Worker index for partitioning sequences')
    parser.add_argument('--num-workers', type=int, default=1, help='Number of manually launched workers')
    parser.add_argument('--decoder-fusion-weight', type=float, default=1.0)
    parser.add_argument('--color-correction', choices=['adain', 'wavelet', 'nofix'], default='nofix')
    parser.add_argument('--max-frames', type=int, default=0, help='Limit real frames per sequence for smoke tests; 0 means all')
    text_source = parser.add_mutually_exclusive_group()
    text_source.add_argument('--text-conditions', help='Use cached QP embeddings from a .pt file')
    text_source.add_argument('--text-encoder-config', help='YAML containing cond_stage_config and text_prompts')
    parser.add_argument('--text-checkpoint', help='Override the OpenCLIP checkpoint in cond_stage_config')
    parser.add_argument('--prompt', help='Use this text for every selected QP in encoder mode')
    opt = parser.parse_args()
    if opt.rgb_buffer_frames < 1:
        parser.error('--rgb-buffer-frames must be a positive integer')
    if opt.prior_buffer_frames < 32:
        parser.error('--prior-buffer-frames must be at least 32 for decoder lookahead')
    if opt.num_workers < 1 or not 0 <= opt.worker_index < opt.num_workers:
        parser.error('--worker-index must be between 0 and --num-workers - 1')
    if opt.max_frames < 0:
        parser.error('--max-frames must be nonnegative')
    if (single_step and opt.steps != 1) or (not single_step and not 2 <= opt.steps <= 1000):
        parser.error('--steps must be 1 for single-step or 2..1000 for multi-step')
    if opt.bitstream and (opt.qp is None or opt.fps is None):
        parser.error('--bitstream requires --qp and --fps')
    if not opt.bitstream and (opt.qp is not None or opt.sequence_name is not None):
        parser.error('--qp and --sequence-name require --bitstream')
    return opt


@dataclass(frozen=True)
class BitstreamSource:
    path: Path
    decoder: Path
    fps: float


def bitstream_jobs(opt):
    if opt.bitstream:
        base = Path.cwd()
        entries = [dict(bitstream=opt.bitstream, qp=opt.qp, fps=opt.fps, name=opt.sequence_name)]
    else:
        manifest = Path(opt.bitstream_list).expanduser().resolve()
        base = manifest.parent
        entries = json.loads(manifest.read_text())
        if not isinstance(entries, list) or not entries:
            raise ValueError('Bitstream manifest must contain a nonempty JSON list')
    default_codec_dir = Path(opt.codec_dir).expanduser().resolve()
    jobs, names = [], set()
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get('bitstream'), str):
            raise ValueError('Every bitstream job requires a bitstream path')
        qp, fps = entry.get('qp'), entry.get('fps', opt.fps)
        if type(qp) is not int or not 0 <= qp <= 51:
            raise ValueError('Every bitstream job requires an integer QP in 0..51')
        if type(fps) not in (int, float) or not math.isfinite(fps) or not 0.1 <= fps <= 100:
            raise ValueError('Every bitstream job requires an fps value in 0.1..100')
        path = Path(entry['bitstream']).expanduser()
        path = (base / path).resolve() if not path.is_absolute() else path.resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        codec_dir = default_codec_dir
        if 'codec_dir' in entry:
            codec_dir = Path(entry['codec_dir']).expanduser()
            codec_dir = (base / codec_dir).resolve() if not codec_dir.is_absolute() else codec_dir.resolve()
        decoder = codec_dir / 'decoder'
        if not decoder.is_file() or not os.access(decoder, os.X_OK):
            raise PermissionError(f'Streaming decoder unavailable; run chmod +x on {decoder}')
        name = entry.get('name') or path.stem
        if not isinstance(name, str) or name in ('.', '..') or Path(name).name != name:
            raise ValueError('Output sequence names must be single directory names')
        if not name.endswith(f'_qp{qp}'):
            name += f'_qp{qp}'
        if name in names:
            raise ValueError(f'Duplicate bitstream output name: {name}')
        names.add(name)
        jobs.append((BitstreamSource(path, decoder, float(fps)), name, qp))
    return [job for index, job in enumerate(jobs) if index % opt.num_workers == opt.worker_index]


def enhance_bitstream(source, output_dir, qp, model, vae, opt, noise_schedule):
    from scripts.stream_adapter import iter_segments
    from codec.processor import VideoStream

    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f'Bitstream output directory must be empty: {output_dir}')
    decoded_dir, enhanced_dir = output_dir / 'Decoded', output_dir / 'Enhanced'
    decoded_dir.mkdir(parents=True, exist_ok=True)
    enhanced_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    with VideoStream(source.path, decoder=source.decoder, ffmpeg=opt.ffmpeg,
                     fps=source.fps, expected_qp=qp, prefetch=opt.rgb_buffer_frames,
                     prior_prefetch=opt.prior_buffer_frames) as video:
        print(f'{output_dir.name}: {video.frames} decoded frames on {model.device}')
        for segment in iter_segments(video, opt.frames_per_segment, opt.max_frames):
            images, residues, motions = segment.to_torch(model.device)
            if images.shape[-2:] != residues.shape[-2:] or images.shape[-2:] != motions.shape[-2:]:
                raise ValueError('Decoded images and priors have different spatial dimensions')
            precision = torch.autocast('cuda') if opt.precision == 'autocast' else nullcontext()
            with precision:
                outputs = enhance_segment(model, vae, images, residues, motions,
                                          segment.qp, opt, noise_schedule)
            for frame, output in zip(segment.frames, outputs[:segment.valid_count]):
                Image.fromarray(frame.rgb).save(decoded_dir / frame.filename())
                Image.fromarray(output.astype(np.uint8)).save(enhanced_dir / frame.filename())
                written += 1
            del images, residues, motions, outputs
        if not video.completed:
            raise RuntimeError('Bitstream decoding did not complete validation')
        expected = min(video.frames, opt.max_frames) if opt.max_frames else video.frames
        if written != expected or video.summary['qp'] != qp:
            raise RuntimeError('Decoded frame count or QP differs from the inference outputs')
        result = dict(bitstream=str(source.path), fps=source.fps, qp=qp,
                      width=video.width, height=video.height, decoded_frames=video.frames,
                      saved_frames=written, decode_verified=True, steps=opt.steps,
                      rgb_buffer_frames=opt.rgb_buffer_frames,
                      prior_buffer_frames=opt.prior_buffer_frames)
    (output_dir / 'inference.json').write_text(json.dumps(result, indent=2) + '\n')
    print(f'Saved {written} decoded and enhanced frames to {output_dir}')


def configure_text_conditioning(config, opt, jobs):
    params = config.model.params
    qps = sorted({qp for _, _, qp in jobs})
    params.text_condition_qps = qps
    if opt.text_conditions:
        params.text_conditions_path = opt.text_conditions
        params.cond_stage_config = None
        params.text_prompts = None
    if opt.text_encoder_config:
        text_config = OmegaConf.load(resolve_path(opt.text_encoder_config))
        params.cond_stage_config = text_config.cond_stage_config
        params.text_prompts = text_config.get('text_prompts')
    if params.get('cond_stage_config') is None:
        if opt.text_checkpoint or opt.prompt is not None:
            raise ValueError('--text-checkpoint and --prompt require a text encoder configuration')
        print(f'Text conditions: cache {params.get("text_conditions_path", "checkpoints/text_conditions.pt")}')
        return
    if opt.text_checkpoint:
        params.cond_stage_config.params.cond_model_path = opt.text_checkpoint
    if opt.prompt is not None:
        prompts = {qp: opt.prompt for qp in qps}
    else:
        prompts = {int(qp): text for qp, text in (params.get('text_prompts') or {}).items()}
        needed = {anchor for qp in qps for anchor in qp_interpolation_weights(qp, prompts)}
        prompts = {qp: prompts[qp] for qp in sorted(needed)}
    if not prompts or any(not isinstance(text, str) or not text.strip() for text in prompts.values()):
        raise ValueError('Text prompts must be nonempty strings')
    params.text_prompts = prompts
    print(f'Text conditions: encode anchors {sorted(prompts)} for QPs {qps}')


def prepare_schedule(model, steps):
    model.register_schedule(timesteps=1000, linear_start=0.00085, linear_end=0.012)
    original = model.sqrt_alphas_cumprod.clone(), model.sqrt_one_minus_alphas_cumprod.clone()
    selected, index = set(), 0.0
    for _ in range(steps):
        selected.add(round(index))
        index += 999 / (steps - 1)
    last_alpha, betas = 1.0, []
    for i, alpha in enumerate(model.alphas_cumprod):
        if i in selected:
            betas.append((1 - alpha / last_alpha).cpu().numpy())
            last_alpha = alpha
    model.register_schedule(given_betas=np.array(betas), timesteps=steps)
    model.original_timesteps = sorted(selected)
    return original


def pad_segment(images, residues, motions):
    height, width = images.shape[-2:]
    if height < 720:
        rows, cols = 1, 1
        pad_h, pad_w = (-height) % 64, (-width) % 64
    elif height < 1080:
        rows, cols = 1, 2
        pad_h, pad_w = (-height) % 64, 0
    else:
        rows, cols = (2, 3) if height < 1600 else (3, 4)
        pad_h = pad_w = 0
    if pad_h or pad_w:
        images, residues, motions = [F.pad(t, (0, pad_w, 0, pad_h), mode='reflect')
                                     for t in [images, residues, motions]]
    return images, residues, motions, rows, cols


@torch.no_grad()
def enhance_segment(model, vae, images, residues, motions, qp, opt, noise_schedule):
    height, width = images.shape[-2:]
    images, residues, motions, rows, cols = pad_segment(images, residues, motions)
    patches, metadata = build_patch_grid(*images.shape[-2:], rows, cols)
    outputs = []
    for hs, ws in patches:
        seed_everything(opt.seed)
        image_patch = images[:, :, hs, ws]
        posterior, encoder_features = vae.encode(image_patch)
        latent = model.sample_latent(posterior)
        prior, prior_features = model.prior_encoder(residues[:, :, hs, ws], motions[:, :, hs, ws], return_features=True)
        condition = model.text_conditions[qp].to(model.device)
        x_T = latent
        if noise_schedule is not None:
            sqrt_alpha, sqrt_one_minus_alpha = noise_schedule
            noise = torch.randn_like(latent)
            t = torch.full((latent.size(0),), 999, device=model.device, dtype=torch.long)
            x_T = model.add_noise_with_schedule(latent, t, sqrt_alpha, sqrt_one_minus_alpha, noise)
        samples = model.sample(cond=condition, struct_cond=latent, prior=prior,
                               x_T=x_T, timesteps=opt.steps)
        qp_tensor = torch.full((images.shape[0],), qp, dtype=torch.float32, device=model.device) / 51.0
        output = vae.decode(samples / model.scale_factor, encoder_features, prior_features, qp_tensor)
        outputs.append(output.float().cpu())
        del posterior, encoder_features, latent, prior, prior_features, condition, x_T, samples, qp_tensor, output
    merge_device = model.device if noise_schedule is not None else 'cpu'
    output = merge_patch_grid(images.shape, patches, outputs, metadata, device=merge_device).to(model.device)
    if opt.color_correction == 'adain':
        output = adaptive_instance_normalization(output, images)
    elif opt.color_correction == 'wavelet':
        output = wavelet_reconstruction(output, images)
    output = torch.clamp((output + 1.0) / 2.0, 0.0, 1.0)
    if not torch.isfinite(output).all():
        raise RuntimeError('Inference produced non-finite image values')
    return output[:, :, :height, :width].cpu().numpy().transpose(0, 2, 3, 1) * 255


def main(single_step=False):
    opt = parse_args(single_step)
    if not torch.cuda.is_available() or not opt.device.startswith('cuda'):
        raise RuntimeError('DiffCVE inference requires a CUDA device')
    device = torch.device('cuda:0' if opt.device == 'cuda' else opt.device)
    torch.cuda.set_device(device)
    jobs = bitstream_jobs(opt)
    if not jobs:
        return
    seed_everything(opt.seed)
    config = OmegaConf.load(resolve_path(opt.config))
    expected = 'SingleStepVideoEnhancer' if single_step else 'MultiStepVideoEnhancer'
    if config.model.target.rsplit('.', 1)[-1] != expected:
        raise ValueError(f'This entry point requires {expected}')
    configure_text_conditioning(config, opt, jobs)
    model = load_checkpoint(instantiate_from_config(config.model), opt.checkpoint).eval().to(device)
    if opt.frames_per_segment != model.num_frames:
        raise ValueError(f'Checkpoint expects {model.num_frames} frames per segment')
    model.denoiser = model.denoiser.merge_and_unload()
    model.eval().requires_grad_(False).to(device)
    if single_step:
        vae = model.vae
        noise_schedule = None
        model.register_schedule(timesteps=1000, linear_start=0.00085, linear_end=0.012)
    else:
        vae_config = OmegaConf.load(resolve_path(opt.vae_config))
        vae = load_checkpoint(instantiate_from_config(vae_config.model), opt.vae_checkpoint)
        vae.eval().requires_grad_(False).to(device)
        noise_schedule = prepare_schedule(model, opt.steps)

    model.to(device)
    vae.decoder.fusion_w = opt.decoder_fusion_weight
    missing_qps = {qp for _, _, qp in jobs} - model.text_conditions.keys()
    if missing_qps:
        raise ValueError(f'Missing text conditions for QPs: {sorted(missing_qps)}')
    outdir = Path(opt.output_dir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    for source, name, qp in jobs:
        enhance_bitstream(source, outdir / name, qp, model, vae, opt, noise_schedule)
