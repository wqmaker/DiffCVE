from dataclasses import dataclass
from typing import Tuple

import numpy as np

from codec.processor import Frame


def normalize_frame(frame):
    if frame.rgb is None:
        raise ValueError('model adapter requires output_format="rgb24"')
    # Preserve the original operation order and float32 arithmetic exactly.
    images = frame.rgb.astype(np.float32) / 255.0
    images = images[None].transpose(0, 3, 1, 2)
    images = (images - 0.5) / 0.5
    residues = (frame.res.astype(np.float32) / 255.0)[None, None]
    motions = frame.mv.astype(np.float32) / 4.0 / 32.0
    motions = motions[None].transpose(0, 3, 1, 2)
    return tuple(np.ascontiguousarray(np.clip(value, -1, 1)) for value in (images, residues, motions))


@dataclass
class Segment:
    frames: Tuple[Frame, ...]   # real frames only, for saving LQ and enhanced PNGs
    images: np.ndarray         # float32 [T,3,H,W], [-1,1]
    residues: np.ndarray       # float32 [T,1,H,W], [-1,1]
    motions: np.ndarray        # float32 [T,2,H,W], [-1,1]
    qp: int

    @property
    def valid_count(self):
        return len(self.frames)

    def to_torch(self, device='cpu'):
        import torch  # Optional: use the model server's existing torch environment.
        values = [torch.from_numpy(value) for value in (self.images, self.residues, self.motions)]
        values[0] = values[0].contiguous(memory_format=torch.channels_last)
        return tuple(value.to(device) for value in values)


def iter_segments(frames, segment_size=5, max_frames=0):
    if type(segment_size) is not int or segment_size < 1:
        raise ValueError('segment_size must be a positive integer')
    if type(max_frames) is not int or max_frames < 0:
        raise ValueError('max_frames must be a nonnegative integer')
    iterator = iter(frames)
    previous = None
    consumed = 0
    while True:
        chunk = []
        for _ in range(segment_size):
            if max_frames and consumed >= max_frames:
                break
            try:
                frame = next(iterator)
            except StopIteration:
                break
            if previous is not None and frame.index != previous + 1:
                raise ValueError('noncontiguous model frame indices')
            previous = frame.index
            chunk.append(frame)
            consumed += 1
        if not chunk:
            if max_frames:
                for _ in iterator:
                    pass
            return
        qp = chunk[0].qp
        if any(frame.qp != qp for frame in chunk):
            raise ValueError('mixed QP within a model segment')
        padded = chunk + [chunk[-1]] * (segment_size - len(chunk))
        normalised = [normalize_frame(frame) for frame in padded]
        values = [np.concatenate([value[i] for value in normalised], axis=0) for i in range(3)]
        yield Segment(tuple(chunk), *values, qp)
        if len(chunk) < segment_size:
            if max_frames:
                for _ in iterator:
                    pass
            return
