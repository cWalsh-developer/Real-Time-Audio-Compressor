import importlib.util

import pytest


pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("torch") is None or importlib.util.find_spec("onnxruntime") is None,
    reason="browser model export requires torch and onnxruntime",
)


def test_browser_export_matches_pytorch(tmp_path):
    torch = pytest.importorskip("torch")
    onnxruntime = pytest.importorskip("onnxruntime")
    from adaptive_audio.export_browser_model import export_model
    from adaptive_audio.training import SpectralMaskNet

    model = SpectralMaskNet().eval()
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save({"model": model.state_dict()}, checkpoint)
    output = export_model(checkpoint, tmp_path / "model.onnx")
    session = onnxruntime.InferenceSession(str(output), providers=["CPUExecutionProvider"])
    values = torch.rand(1, 257, 31)
    with torch.inference_mode():
        expected = model(values).numpy()
    actual = session.run(["speech_mask"], {"mixture_magnitude": values.numpy()})[0]
    assert actual.shape == expected.shape
    assert abs(actual - expected).max() < 1e-5