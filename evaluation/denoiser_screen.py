"""Reproduce the isolated DNS64 research screen on the existing test mixture.

Run from the project root in the optional separation environment. This uses
the upstream streaming implementation unchanged, except that EOF is drained
with zero input instead of calling flush(), which resets recurrent state.
Upstream Denoiser is CC-BY-NC 4.0; this is not a distributable product backend.
See dtln-findings.md for setup, limitations, and the failed screening result.
"""
from pathlib import Path
import sys, time, json, hashlib, subprocess
import numpy as np
import torch
from scipy.signal import resample_poly
from adaptive_audio.wav import read_wav, write_wav

UPSTREAM = Path('models/denoiser').resolve()
COMMIT = '8afd7c166699bb3c8b2d95b6dd706f71e1075df0'
revision = subprocess.check_output(['git', '-C', str(UPSTREAM), 'rev-parse', 'HEAD'], text=True).strip()
dirty = subprocess.check_output(['git', '-C', str(UPSTREAM), 'status', '--porcelain', '--untracked-files=no'], text=True)
if revision != COMMIT or dirty:
    raise ValueError('expected the clean pinned Denoiser source checkout')
sys.path.insert(0, str(UPSTREAM))
from denoiser.demucs import Demucs, DemucsStreamer

weights = Path('models/denoiser/dns64-a7761ff99a7d5bb6.th')
assert hashlib.sha256(weights.read_bytes()).hexdigest() == 'a7761ff99a7d5bb69c16a61b3089469805cf3afd82e128103f0f77e79dfead5f'
torch.set_num_threads(1)
model = Demucs(hidden=64, sample_rate=16000).eval()
model.load_state_dict(torch.load(weights, map_location='cpu', weights_only=True))

def render(lowband, dry=0, chunk=256):
    streams = [DemucsStreamer(model, dry=dry, num_frames=1) for _ in range(lowband.shape[1])]
    outputs = [[] for _ in streams]
    timings = []
    available = None
    with torch.inference_mode():
        for start in range(0, len(lowband), chunk):
            tick = time.perf_counter()
            for channel, stream in enumerate(streams):
                block = torch.from_numpy(lowband[start:start+chunk,channel].copy()).reshape(1,-1)
                result = stream.feed(block)
                if result.shape[1] and available is None:
                    available = min(start+chunk, len(lowband)) / 16
                outputs[channel].append(result.numpy().copy())
            timings.append(time.perf_counter()-tick)
        for channel, stream in enumerate(streams):
            # Feed zeros rather than reset recurrent state at EOF.
            outputs[channel].append(stream.feed(torch.zeros(1, stream.total_length)).numpy().copy())
    result = np.stack([np.concatenate(channel,axis=1).ravel()[:len(lowband)] for channel in outputs],axis=1)
    assert result.shape == lowband.shape
    return result, timings, available, streams[0].total_length / 16

test = np.random.default_rng(0).normal(0, 0.02, (4099, 2)).astype(np.float32)
dry, *_ = render(test, dry=1)
assert np.max(np.abs(dry-test)) == 0
one, *_ = render(test)
odd_chunks, *_ = render(test, chunk=113)
assert np.max(np.abs(one-odd_chunks)) < 1e-6
print('dry alignment and arbitrary block partition checks passed', flush=True)

for name in ['music', 'mixture', 'speech']:
    source = Path(f'audio/separation/dfn_synthetic/{name}.wav')
    audio, rate = read_wav(source)
    began = time.perf_counter()
    lowband = resample_poly(audio, 1, 3, axis=0)
    speech, timings, available, required = render(lowband)
    residual = resample_poly(lowband-speech, 3, 1, axis=0)[:len(audio)]
    preview = audio + (10**(-6/20)-1)*residual
    elapsed = time.perf_counter()-began
    assert np.isfinite(preview).all() and np.max(np.abs(preview)) < 1
    directory = Path(f'audio/separation/dns64_{name}')
    directory.mkdir(exist_ok=True)
    write_wav(directory/'reduced_preview.wav', preview, rate)
    write_wav(directory/'speech_estimate_16k.wav', speech, 16000)
    report = {'input':str(source), 'model':'Denoiser DNS64, CC-BY-NC 4.0 research only',
        'commit':COMMIT,
        'weights_sha256':hashlib.sha256(weights.read_bytes()).hexdigest(),
        'input_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
        'seconds':elapsed, 'hop_ms':16, 'hop_p99_ms':float(np.percentile(timings,99)*1000),
        'hop_max_ms':max(timings)*1000, 'missed':sum(t>0.016 for t in timings),
        'first_output_after_input_ms':available, 'required_input_ms':required,
        'output_level_change_db':float(10*np.log10(np.mean(preview.astype(np.float64)**2)/np.mean(audio.astype(np.float64)**2))),
        'torch':torch.__version__, 'note':'CPU one thread; offline external resampling; no Chrome latency measured'}
    (directory/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report), flush=True)
