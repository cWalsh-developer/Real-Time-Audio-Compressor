# Stem-first listening trial

This is an offline proof of the intended default route: estimate dialogue,
music, and effects from the mixed soundtrack; leave dialogue alone; reduce
music/effects; and remix. Netflix tab capture supplies only a mix, so these
stems are estimates. The original mix remains the remix reference, making a
zero-reduction preview exactly equal to bypass. Listen for speech leaking into
the music/effects stems, especially during simultaneous dialogue and action.

The trial uses [BandIt v2 Multi](https://github.com/kwatcharasupat/bandit-v2),
a cinematic three-stem separator, through the experimental
[bandit-infer](https://github.com/openmirlab/bandit-infer) wrapper. The pinned
wrapper commit is `d45cdec634bf1ee01cdd2acea74a2d100e639c8a`. The wrapper
downloads and checks the official model weights on first use (about 447 MB).
The code is Apache-2.0; the v2 weights have a separate CC-BY-SA-4.0 license.
Keep this model evaluation separate from any distribution decision.

Set up an isolated environment (PowerShell):

```powershell
python -m venv models/separation-venv
models/separation-venv/Scripts/python.exe -m pip install -e .
models/separation-venv/Scripts/python.exe -m pip install "git+https://github.com/openmirlab/bandit-infer.git@d45cdec634bf1ee01cdd2acea74a2d100e639c8a"
```

CPU inference works for a small trial. For the NVIDIA GPU, install a
[CUDA-enabled PyTorch build](https://pytorch.org/get-started/locally/) that
supports the GPU, then pass `--device cuda`. Check availability with
`models/separation-venv/Scripts/python.exe -c "import torch; print(torch.cuda.is_available())"`.

Use a 48 kHz stereo 16-bit PCM WAV, for example a short local film excerpt:

```powershell
models/separation-venv/Scripts/python.exe evaluation/separation_trial.py audio/eval/service_middle_original.wav audio/separation/service_middle --start 0 --duration 12 --reduction-db 6 --device cuda
```

The output folder contains the original segment, three estimated stems, a
fixed-reduction preview, and `report.json`. Compare original and preview at the
same playback volume, then solo the music/effects stems to hear leaked speech.
The fixed 6 dB reduction is an audition control, not the final event compressor.
The preview WAV may clip if imperfect stem estimates cause peaks over full
scale; the report records the unclipped peak so this is visible.

BandIt v2's published inference path uses **8-second, overlapping chunks**.
Even if its inference runs faster than playback, this noncausal context cannot
simply replace the extension's current AudioWorklet without multi-second sound
delay. The report measures compute throughput only, not live end-to-end
latency. A streaming, low-look-ahead model or a different playback architecture
will be needed before stem-first processing can become the live default.
