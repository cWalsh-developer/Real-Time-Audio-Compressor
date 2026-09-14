"""Train a compact spectral speech-mask baseline on prepared DnR windows."""

from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from .training_data import iter_batches


N_FFT = 512
HOP_LENGTH = 256


class SpectralMaskNet(nn.Module):
    """Predict a bounded speech magnitude mask from a mixture spectrogram."""

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(16, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, mixture_magnitude):
        features = torch.log1p(mixture_magnitude).unsqueeze(1)
        return self.layers(features).squeeze(1)


def _window(device):
    return torch.hann_window(N_FFT, device=device)


def _loss(model, mixture, speech, device):
    window = _window(device)
    mixture_stft = torch.stft(mixture, N_FFT, HOP_LENGTH, N_FFT, window, return_complex=True)
    speech_stft = torch.stft(speech, N_FFT, HOP_LENGTH, N_FFT, window, return_complex=True)
    mixture_magnitude = mixture_stft.abs()
    target_magnitude = speech_stft.abs()
    mask = model(mixture_magnitude)
    estimate_magnitude = mask * mixture_magnitude
    return F.l1_loss(estimate_magnitude, target_magnitude)


def _run_epoch(model, root, split, batch_size, device, optimizer=None, max_batches=None, seed=0):
    is_training = optimizer is not None
    model.train(is_training)
    losses = []
    batches = iter_batches(root, split, batch_size, shuffle=is_training, seed=seed)
    for batch_number, batch in enumerate(batches):
        mixture = torch.from_numpy(batch["mixture"]).to(device)
        speech = torch.from_numpy(batch["speech"]).to(device)
        if is_training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(is_training):
            loss = _loss(model, mixture, speech, device)
        if is_training:
            loss.backward()
            optimizer.step()
        losses.append(float(loss.detach().cpu()))
        if max_batches is not None and batch_number + 1 >= max_batches:
            break
    return sum(losses) / len(losses)


def train_model(root, output, *, epochs=5, batch_size=4, learning_rate=1e-3,
                device="auto", resume=None, max_batches=None, seed=0):
    """Train the baseline and save latest and per-epoch checkpoints."""
    if type(epochs) is not int or epochs < 1:
        raise ValueError("epochs must be a positive integer")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_device = torch.device(device)
    model = SpectralMaskNet().to(torch_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    start_epoch = 0
    history = []
    if resume is not None:
        checkpoint = torch.load(resume, map_location=torch_device, weights_only=True)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint["epoch"]
        history = checkpoint.get("history", [])
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for epoch in range(start_epoch, epochs):
        train_loss = _run_epoch(model, root, "train", batch_size, torch_device, optimizer, max_batches, seed + epoch)
        validation_loss = _run_epoch(model, root, "val", batch_size, torch_device, max_batches=max_batches)
        history.append({"epoch": epoch + 1, "train_loss": train_loss, "validation_loss": validation_loss})
        state = {"epoch": epoch + 1, "model": model.state_dict(), "optimizer": optimizer.state_dict(), "history": history}
        torch.save(state, output / "checkpoint_latest.pt")
        torch.save(state, output / f"checkpoint_epoch_{epoch + 1:03d}.pt")
    return history