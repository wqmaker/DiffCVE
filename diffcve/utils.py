import importlib
from inspect import isfunction
from pathlib import Path
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]

def resolve_path(path):
    path = Path(path).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path

def exists(value):
    return value is not None

def default(value, fallback):
    return value if exists(value) else fallback() if isfunction(fallback) else fallback

def instantiate_from_config(config):
    module, name = config['target'].rsplit('.', 1)
    return getattr(importlib.import_module(module), name)(**config.get('params', {}))

def load_checkpoint(model, path):
    path = resolve_path(path)
    checkpoint = torch.load(path, map_location='cpu')
    state = checkpoint.get('state_dict', checkpoint)
    expected = model.state_dict()
    selected = {}
    prefixes = {
        'model.': 'denoiser.',
        'priorcond_stage_model.': 'prior_condition_encoder.',
        'structcond_stage_model.': 'structure_encoder.',
        'vaep.': 'vae.',
    }
    for key, value in state.items():
        target = key
        for prefix, replacement in prefixes.items():
            if target.startswith(prefix):
                target = replacement + target[len(prefix):]
                break
        target = target.replace('.diffusion_model.', '.network.')
        if target in expected:
            if target in selected:
                raise RuntimeError(f'{path}: duplicate inference tensor: {target}')
            selected[target] = value
    missing = set(expected) - selected.keys()
    if missing:
        raise RuntimeError(f'{path}: missing inference tensors: {sorted(missing)}')
    model.load_state_dict(selected, strict=True)
    print(f'Loaded {len(selected)} inference tensors from {path.name}; '
          f'skipped {len(state) - len(selected)} unused checkpoint tensors.')
    return model
