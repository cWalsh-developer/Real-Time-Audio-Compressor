# Chrome live-audio prototype

This unpacked Manifest V3 extension captures a tab's audio, reduces loud passages in real time, and replays it to the default speaker output. It does not record, upload, classify, or use a multi-second buffer. A level-sensitive AudioWorklet passes dialogue-range audio through unchanged and attenuates only higher-level passages. It is **not** the Python Balanced curve or the AI classifier. A green `CMP` badge means loudness reduction is active; green `AUD` means original audio is playing; amber `0` means capture is active but no audio was measured; `ERR` means capture failed.

## Load and test

1. Open `chrome://extensions` in Chrome, turn on **Developer mode**, and choose **Load unpacked**.
2. Select this `chrome-extension` folder.
3. Play an ordinary video tab with audible sound. Click the extension icon to start compression (`CMP`). Press **Alt+Shift+P** to switch between compressed (`CMP`) and original (`AUD`) audio at the same speaker volume. Click again to stop.
4. Repeat on Netflix while signed in and playing a title. For fullscreen, enter Netflix's video fullscreen **before** starting capture, then press **Alt+Shift+A** to activate the extension. Press it again to stop. Set either shortcut at `chrome://extensions/shortcuts` if Chrome has not assigned it.

Chrome may prevent a video player from entering fullscreen **after** tab capture begins. If you are already capturing, stop capture, enter fullscreen, then use the shortcut. If the player still exits fullscreen, try Chrome's **F11** window fullscreen as a fallback and report whether it behaves differently.

Chrome requires a user click before tab capture. When a tab is captured, Chrome stops its normal audio output, so the extension explicitly routes captured audio back to the speakers. The extension uses an offscreen document to keep the audio graph running after the click. It needs Chrome 116 or newer.

The reducer tracks a slowly changing background level and reacts when a passage rises substantially above it. A low-frequency sidechain reduces sustained bass and sub-bass energy, and when bass is a strong part of the program it also applies conservative extra attenuation to the upper band, which helps control guitar and similar theme-song content. The bass-conditioned upper-band detector is deliberately smoothed over musical timescales so individual guitar notes do not cause large gain swings. A fast peak detector catches brief, very loud transients such as gunfire; the output also has a 0.8 sample-peak cap. Sustained broadband reduction is limited to 10.5 dB, with up to 9 dB of bass-conditioned upper-band reduction, and audio is never boosted. Ordinary changes in speaking level should pass through, but this is still level-based processing, not speech or sound-effect recognition: a sufficiently loud voice can be reduced, and a quieter effect can pass through. Please compare dialogue, bass impact, theme music, guitar, effects, pumping, and lip-sync on the same title and speaker volume. The existing 2-second Python rolling buffer cannot be inserted into this audio-only path without making sound late relative to video.

Sources: [Chrome tabCapture](https://developer.chrome.com/docs/extensions/reference/api/tabCapture), [Chrome offscreen documents](https://developer.chrome.com/docs/extensions/reference/api/offscreen).

To repeat the browser audio-render check, install Playwright (`python -m pip install playwright`), install its Chromium build (`python -m playwright install chromium`), then run `python chrome-extension/verify_audio.py` from the project root.
