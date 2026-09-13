# Chrome live-audio prototype

This unpacked Manifest V3 extension captures a tab's audio, applies stronger live dynamic-range compression, and replays it to the default speaker output. It does not record, upload, classify, or buffer the audio. The live effect is a Web Audio compressor, **not** the Python Balanced curve or the AI classifier. A green `CMP` badge means compressed audio is playing; green `AUD` means original audio is playing; amber `0` means capture is active but no audio was measured; `ERR` means capture failed.

## Load and test

1. Open `chrome://extensions` in Chrome, turn on **Developer mode**, and choose **Load unpacked**.
2. Select this `chrome-extension` folder.
3. Play an ordinary video tab with audible sound. Click the extension icon to start compression (`CMP`). Press **Alt+Shift+P** to switch between compressed (`CMP`) and original (`AUD`) audio at the same speaker volume. Click again to stop.
4. Repeat on Netflix while signed in and playing a title. For fullscreen, enter Netflix's video fullscreen **before** starting capture, then press **Alt+Shift+A** to activate the extension. Press it again to stop. Set either shortcut at `chrome://extensions/shortcuts` if Chrome has not assigned it.

Chrome may prevent a video player from entering fullscreen **after** tab capture begins. If you are already capturing, stop capture, enter fullscreen, then use the shortcut. If the player still exits fullscreen, try Chrome's **F11** window fullscreen as a fallback and report whether it behaves differently.

Chrome requires a user click before tab capture. When a tab is captured, Chrome stops its normal audio output, so the extension explicitly routes captured audio back to the speakers. The extension uses an offscreen document to keep the audio graph running after the click. It needs Chrome 116 or newer.

The compressor uses a -24 dBFS threshold, 6 dB knee, 6:1 ratio, 5 ms attack, 300 ms release, and output gain to lift typical dialogue slightly above its original volume while reducing much louder passages. These are listening-test settings; it has no dialogue recognition or LUFS target. Please compare dialogue, loud effects, pumping, and lip-sync on the same title and speaker volume. The existing 2-second Python rolling buffer cannot be inserted into this audio-only path without making sound late relative to video.

Sources: [Chrome tabCapture](https://developer.chrome.com/docs/extensions/reference/api/tabCapture), [Chrome offscreen documents](https://developer.chrome.com/docs/extensions/reference/api/offscreen).
