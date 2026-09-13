import numpy as np

from adaptive_audio.classification import EventScores
from adaptive_audio.streaming import process_wav
from adaptive_audio.wav import read_wav, write_wav


def test_semantic_scores_protect_speech_and_reduce_action(tmp_path):
    rate = 8000
    time = np.arange(rate) / rate
    tone = np.sin(2 * np.pi * 440 * time)
    audio = np.concatenate([0.04 * tone, 0.4 * tone]).astype(np.float32)
    source = tmp_path / "source.wav"
    baseline = tmp_path / "baseline.wav"
    informed = tmp_path / "informed.wav"
    write_wav(source, audio, rate)
    labels = [
        EventScores(0, 1, {"speech": 1, "music": 0, "action": 0, "other": 0}),
        EventScores(1, 2, {"speech": 0, "music": 0, "action": 1, "other": 0}),
    ]
    process_wav(source, baseline, "balanced")
    process_wav(source, informed, "balanced", labels=labels)
    plain, _ = read_wav(baseline)
    semantic, _ = read_wav(informed)
    assert np.sqrt(np.mean(semantic[2000:6000] ** 2)) > np.sqrt(np.mean(plain[2000:6000] ** 2))
    assert np.sqrt(np.mean(semantic[10000:14000] ** 2)) < np.sqrt(np.mean(plain[10000:14000] ** 2))
    assert np.max(np.abs(semantic)) <= 10 ** (-1 / 20) + 2 / 32768
