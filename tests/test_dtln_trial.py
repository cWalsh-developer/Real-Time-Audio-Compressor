"""Model-free checks for timing and state in the isolated DTLN adapter."""

from types import SimpleNamespace

import numpy as np
import pytest

from evaluation.dtln_trial import BLOCK, HOP, ALIGNMENT, enhance


class IdentityStage:
    """An identity analysis/synthesis pair with recurrent state passthrough."""

    def __init__(self, stage):
        self.stage = stage

    def get_inputs(self):
        return [SimpleNamespace(name="audio", shape=[1, 1, 257 if self.stage == 1 else BLOCK]),
                SimpleNamespace(name="state", shape=[1, 2, 128, 2])]

    def run(self, _, inputs):
        audio = inputs["audio"]
        # Four overlapping blocks: a unity reconstruction needs 1/4 per block.
        return [np.ones_like(audio) if self.stage == 1 else audio / 4,
                inputs["state"]]


@pytest.mark.parametrize("length", [1, HOP, BLOCK, 16003])
def test_identity_pair_reconstructs_both_channels_and_flushes_tail(length):
    audio = np.random.default_rng(42).normal(0, 0.1, (length, 2)).astype(np.float32)
    result, timings = enhance(audio, (IdentityStage(1), IdentityStage(2)))
    assert result.shape == audio.shape
    np.testing.assert_allclose(result, audio, atol=1e-7)
    assert len(timings) == (length + ALIGNMENT + HOP - 1) // HOP


def test_live_output_is_delayed_and_does_not_use_future_input():
    audio = np.random.default_rng(1).normal(0, 0.1, (4096, 2)).astype(np.float32)
    other = audio.copy()
    other[2048:] *= -3
    stages = (IdentityStage(1), IdentityStage(2))
    first, _ = enhance(audio, stages, align=False)
    second, _ = enhance(other, stages, align=False)
    np.testing.assert_allclose(first[:ALIGNMENT], 0, atol=1e-7)
    np.testing.assert_allclose(first[ALIGNMENT:ALIGNMENT + len(audio)], audio, atol=1e-7)
    np.testing.assert_array_equal(first[:2048], second[:2048])


def test_recurrent_state_is_independent_per_channel():
    class Accumulator(IdentityStage):
        def run(self, _, inputs):
            state = inputs["state"] + float(np.mean(inputs["audio"]))
            return [inputs["audio"] / 4 + state.ravel()[0] * 0.01, state]

    audio = np.zeros((2048, 2), np.float32)
    audio[:, 0] = 0.1
    result, _ = enhance(audio, (IdentityStage(1), Accumulator(2)))
    np.testing.assert_array_equal(result[:, 1], 0)
    mono, _ = enhance(audio[:, :1], (IdentityStage(1), Accumulator(2)))
    np.testing.assert_array_equal(result[:, :1], mono)


@pytest.mark.parametrize("audio", [np.zeros(10), np.zeros((0, 2)), np.full((128, 2), np.nan)])
def test_rejects_invalid_input(audio):
    with pytest.raises(ValueError):
        enhance(audio, (IdentityStage(1), IdentityStage(2)))
