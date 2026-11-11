import argparse
from collections import deque
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import queue
import select
import shutil
import struct
import subprocess
import threading
import time
from typing import Optional

import numpy as np

API_VERSION = 1


class CodecError(RuntimeError):
    """Decode, protocol, FFmpeg, timeout or QP validation failure."""


@dataclass
class Frame:
    index: int                 # zero based, in decoded/display order
    qp: int                    # decoded QP, never inferred from the filename
    rgb: Optional[np.ndarray]  # uint8 [H,W,3], RGB; None in YUV-only mode
    mv: np.ndarray             # float32 [H,W,2], raw quarter-luma-pixel units
    res: np.ndarray            # float32 [H,W], decoded luma prediction residual
    yuv420p: Optional[np.ndarray] = None  # uint8 [H*3//2,W], Y then U then V

    def filename(self, digits=5):
        return 'im{:0{}d}.png'.format(self.index + 1, digits)


def _executable(value):
    text = os.fspath(value)
    resolved = shutil.which(text) if '/' not in text else text
    if not resolved or not os.access(resolved, os.X_OK) or not Path(resolved).is_file():
        raise FileNotFoundError('executable unavailable: ' + text)
    return str(Path(resolved).resolve())


def _read_exact(pipe, count, timeout=None):
    data = bytearray(count)
    view = memoryview(data)
    position = 0
    deadline = None if timeout is None else time.monotonic() + timeout
    while position < count:
        if deadline is not None:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not select.select([pipe], [], [], remaining)[0]:
                raise CodecError('timed out waiting for decoder metadata')
        chunk = os.read(pipe.fileno(), min(count-position, 1024*1024))
        if not chunk:
            raise CodecError('truncated decoder/FFmpeg stream: expected {} bytes, received {}'.format(count, position))
        view[position:position+len(chunk)] = chunk
        position += len(chunk)
    return data


def _json_record(pipe, timeout=None):
    length = struct.unpack('<I', _read_exact(pipe, 4, timeout))[0]
    if not 1 <= length <= 1024*1024:
        raise CodecError('invalid JSON record size')
    value = json.loads(_read_exact(pipe, length, timeout).decode('utf-8'))
    if not isinstance(value, dict):
        raise CodecError('metadata must be an object')
    return value


class VideoStream:
    def __init__(self, bitstream, *, decoder=None, ffmpeg='ffmpeg', fps=24,
                 threads=1, expected_qp=None, output_format='rgb24', prefetch=5,
                 prior_prefetch=32, timeout=120):
        self.bitstream = str(Path(bitstream).expanduser().resolve(strict=True))
        if not Path(self.bitstream).is_file():
            raise ValueError('bitstream must be a regular file')
        self.decoder_path = _executable(decoder if decoder is not None else Path(__file__).with_name('decoder'))
        if output_format not in ('rgb24', 'yuv420p'):
            raise ValueError('output_format must be rgb24 or yuv420p')
        self.output_format = output_format
        self.ffmpeg_path = _executable(ffmpeg) if output_format == 'rgb24' else None
        self.fps = float(fps)
        if not math.isfinite(self.fps) or not .1 <= self.fps <= 100:
            raise ValueError('fps must be between 0.1 and 100')
        if type(threads) is not int or not 1 <= threads <= 64:
            raise ValueError('threads must be an integer between 1 and 64')
        if type(prefetch) is not int or prefetch < 1:
            raise ValueError('prefetch must be a positive integer')
        if type(prior_prefetch) is not int or prior_prefetch < 32:
            raise ValueError('prior_prefetch must be an integer of at least 32 for decoder lookahead')
        if expected_qp is not None and (type(expected_qp) is not int or not 0 <= expected_qp <= 51):
            raise ValueError('expected_qp must be an integer between 0 and 51')
        if not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError('timeout must be finite and positive')
        self.expected_qp, self.threads, self.timeout = expected_qp, threads, timeout
        self.metadata, self.summary, self.qp = {}, None, None
        self.completed = False
        self._started = self._closed = False
        self._index = 0
        # The RGB queue imposes backpressure; these slots are allocated on demand.
        self._priors = queue.Queue(maxsize=prior_prefetch)
        self._images = queue.Queue(maxsize=prefetch)
        self._stop = threading.Event()
        self._threads, self._processes, self._owned_pipes = [], [], []
        self._logs = {}

    def __enter__(self):
        self._start()
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()
        return False

    def __iter__(self):
        return self

    def _thread(self, target):
        thread = threading.Thread(target=target, daemon=True)
        self._threads.append(thread)
        thread.start()

    def _diagnostics(self, label, process):
        tail = deque(maxlen=64)
        self._logs[label] = tail
        def drain():
            try:
                while True:
                    chunk = process.stderr.read(4096)
                    if not chunk:
                        break
                    tail.append(chunk)
            except (OSError, ValueError):
                pass
        self._thread(drain)

    def _put(self, target, item):
        while not self._stop.is_set():
            try:
                target.put(item, timeout=.1)
                return
            except queue.Full:
                pass

    def _take(self, target):
        try:
            kind, item = target.get(timeout=self.timeout)
        except queue.Empty:
            raise CodecError('timed out waiting for a frame (decoder/FFmpeg stalled)')
        if kind == 'error':
            raise CodecError(str(item)) from item
        return kind, item

    def _failure(self, error):
        details = '\n'.join(label + ': ' + b''.join(tail).decode('utf-8', errors='replace')[-8000:]
                            for label, tail in self._logs.items() if tail)
        return CodecError(str(error) + ('\n' + details if details else ''))

    def _start(self):
        if self._closed:
            raise CodecError('stream is closed')
        if self._started:
            return
        self._started = True
        yuv_read, yuv_write = os.pipe()
        try:
            command = [self.decoder_path, '--stream', '--input', self.bitstream,
                       '--yuv-fd', str(yuv_write), '--fps', str(self.fps)]
            decoder = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                       stdin=subprocess.DEVNULL, pass_fds=(yuv_write,), bufsize=0)
            self._processes.append(('decoder', decoder))
            os.close(yuv_write)
            yuv_write = None
            self._diagnostics('decoder', decoder)
            if _read_exact(decoder.stdout, 8, self.timeout) != b'CVPRIO1\n':
                raise CodecError('decoder does not provide stream protocol version 1')
            self.metadata = _json_record(decoder.stdout, self.timeout)
            m = self.metadata
            if (m.get('protocol_version') != 1 or m.get('pixel_format') != 'yuv420p' or
                    m.get('mv_dtype') != '<f4' or m.get('res_dtype') != '<f4' or
                    m.get('mv_units') != 'quarter_luma_pixels'):
                raise CodecError('unsupported stream metadata')
            self.width, self.height, self.frames = m['width'], m['height'], m['frames']
            if (type(self.width) is not int or type(self.height) is not int or type(self.frames) is not int or
                    not 16 <= self.width <= 16384 or not 16 <= self.height <= 16384 or
                    self.width % 2 or self.height % 2 or not 1 <= self.frames <= 10000000):
                raise CodecError('invalid dimensions or frame count')
            if self.output_format == 'rgb24':
                command = [self.ffmpeg_path, '-nostdin', '-hide_banner', '-loglevel', 'error',
                           '-f', 'rawvideo', '-pixel_format', 'yuv420p', '-video_size',
                           '{}x{}'.format(self.width, self.height), '-framerate', str(self.fps),
                           '-probesize', '32', '-analyzeduration', '0', '-i', 'pipe:0',
                           '-frames:v', str(self.frames), '-pix_fmt', 'rgb24', '-threads', str(self.threads),
                           '-f', 'rawvideo', 'pipe:1']
                converter = subprocess.Popen(command, stdin=yuv_read, stdout=subprocess.PIPE,
                                             stderr=subprocess.PIPE, bufsize=0)
                self._processes.append(('ffmpeg', converter))
                self._diagnostics('ffmpeg', converter)
                os.close(yuv_read)
                yuv_read = None
                images_pipe = converter.stdout
            else:
                images_pipe = os.fdopen(yuv_read, 'rb', buffering=0)
                self._owned_pipes.append(images_pipe)
                yuv_read = None
            self._thread(lambda: self._read_priors(decoder.stdout))
            self._thread(lambda: self._read_images(images_pipe))
        except BaseException as error:
            self.close()
            if isinstance(error, (KeyboardInterrupt, SystemExit)):
                raise
            raise self._failure(error) from error
        finally:
            if yuv_read is not None:
                os.close(yuv_read)
            if yuv_write is not None:
                os.close(yuv_write)

    def _read_priors(self, pipe):
        try:
            pixels = self.width * self.height
            for index in range(self.frames):
                if _read_exact(pipe, 4) != b'FRM1':
                    raise CodecError('missing or invalid prior frame record')
                actual, qp = struct.unpack('<II', _read_exact(pipe, 8))
                if actual != index or qp > 51:
                    raise CodecError('misaligned frame index or invalid QP')
                if self.expected_qp is not None and qp != self.expected_qp:
                    raise CodecError('decoded QP {} differs from expected QP {}'.format(qp, self.expected_qp))
                mv = np.frombuffer(_read_exact(pipe, pixels*8), dtype='<f4').reshape(self.height, self.width, 2).copy()
                res = np.frombuffer(_read_exact(pipe, pixels*4), dtype='<f4').reshape(self.height, self.width).copy()
                self._put(self._priors, ('frame', (index, qp, mv, res)))
            if _read_exact(pipe, 4) != b'END1':
                raise CodecError('missing verified stream footer')
            summary = _json_record(pipe)
            if (summary.get('checked_frames') != self.frames or summary.get('qp_verified_constant') is not True or
                    type(summary.get('qp')) is not int or not 0 <= summary['qp'] <= 51):
                raise CodecError('invalid verification footer')
            if pipe.read(1):
                raise CodecError('unexpected bytes after verification footer')
            self._put(self._priors, ('end', summary))
        except BaseException as error:
            self._put(self._priors, ('error', error))

    def _read_images(self, pipe):
        try:
            rgb = self.output_format == 'rgb24'
            size = self.width*self.height*3 if rgb else self.width*self.height*3//2
            shape = (self.height, self.width, 3) if rgb else (self.height*3//2, self.width)
            for index in range(self.frames):
                value = np.frombuffer(_read_exact(pipe, size), dtype=np.uint8).reshape(shape).copy()
                self._put(self._images, ('frame', (index, value)))
            if pipe.read(1):
                raise CodecError('unexpected extra decoded image data')
            self._put(self._images, ('end', None))
        except BaseException as error:
            self._put(self._images, ('error', error))

    def __next__(self):
        if self.completed:
            raise StopIteration
        try:
            self._start()
            kind, prior = self._take(self._priors)
            if kind == 'end':
                if self._index != self.frames or self._take(self._images)[0] != 'end':
                    raise CodecError('image/prior counts differ')
                for label, process in self._processes:
                    if process.wait(timeout=self.timeout) != 0:
                        raise CodecError(label + ' exited unsuccessfully')
                if self.qp != prior['qp']:
                    raise CodecError('stream footer QP differs from frame QP')
                self.summary = prior
                self.metadata.update(qp=prior['qp'], qp_verified_constant=True,
                                     qp_checked_frames=prior['checked_frames'])
                self.completed = True
                self.close()
                raise StopIteration
            image_kind, image = self._take(self._images)
            index, qp, mv, res = prior
            if image_kind != 'frame' or index != self._index or image[0] != index:
                raise CodecError('image and prior frame indices differ')
            if self.qp is not None and self.qp != qp:
                raise CodecError('decoded video is not constant-QP')
            self.qp = qp
            self.metadata['qp'] = qp
            self._index += 1
            rgb = image[1] if self.output_format == 'rgb24' else None
            yuv = image[1] if self.output_format == 'yuv420p' else None
            return Frame(index, qp, rgb, mv, res, yuv)
        except StopIteration:
            raise
        except BaseException as error:
            self.close()
            if isinstance(error, (KeyboardInterrupt, SystemExit)):
                raise
            raise self._failure(error) from error

    def close(self):
        if self._closed:
            return
        self._closed = True
        self._stop.set()
        for _, process in self._processes:
            if process.poll() is None:
                process.terminate()
        for _, process in self._processes:
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        for thread in self._threads:
            if thread is not threading.current_thread():
                thread.join(timeout=3)
        for _, process in self._processes:
            for pipe in (process.stdout, process.stderr):
                if pipe is not None:
                    pipe.close()
        for pipe in self._owned_pipes:
            pipe.close()


def main():
    parser = argparse.ArgumentParser(description='Consume a bitstream in memory; write no intermediate files.')
    parser.add_argument('--input', required=True)
    parser.add_argument('--decoder')
    parser.add_argument('--ffmpeg', default='ffmpeg')
    parser.add_argument('--fps', type=float, default=24)
    parser.add_argument('--expected-qp', type=int)
    parser.add_argument('--output-format', choices=('rgb24', 'yuv420p'), default='rgb24')
    parser.add_argument('--rgb-buffer-frames', type=int, default=5)
    parser.add_argument('--prior-buffer-frames', type=int, default=32)
    opt = parser.parse_args()
    with VideoStream(opt.input, decoder=opt.decoder, ffmpeg=opt.ffmpeg, fps=opt.fps,
                     expected_qp=opt.expected_qp, output_format=opt.output_format,
                     prefetch=opt.rgb_buffer_frames, prior_prefetch=opt.prior_buffer_frames) as stream:
        for frame in stream:
            pass  # The enhancement model consumes Frames here; the CLI only verifies the stream.
        print(json.dumps(dict(metadata=stream.metadata, summary=stream.summary, completed=stream.completed), indent=2))


if __name__ == '__main__':
    main()
