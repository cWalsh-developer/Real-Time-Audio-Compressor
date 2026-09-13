# Chrome tab-audio feasibility probe

This unpacked Manifest V3 extension tests Chrome's tab-audio path. Clicking its toolbar icon captures the selected tab's audio, passes it through an `AudioContext` **unchanged**, and replays it to the default speaker output. It does not record, upload, classify, compress, or delay the audio. A green `AUD` badge means the extension is receiving nonzero audio; an amber `0` means capture is active but no audio was measured; `ERR` means capture failed. Click again to stop.

## Load and test

1. Open `chrome://extensions` in Chrome, turn on **Developer mode**, and choose **Load unpacked**.
2. Select this `chrome-extension` folder.
3. Play an ordinary video tab with audible sound. Click the extension icon. Confirm the audio continues, the badge changes to `AUD`, and the sound remains synchronized with the picture. Click again to stop.
4. Repeat on Netflix while signed in and playing a title. For fullscreen, enter Netflix's video fullscreen **before** starting capture, then press **Alt+Shift+A** to activate the extension. Press the shortcut again to stop. If that shortcut is unavailable, set one at `chrome://extensions/shortcuts`.

Chrome may prevent a video player from entering fullscreen **after** tab capture begins. If you are already capturing, stop capture, enter fullscreen, then use the shortcut. If the player still exits fullscreen, try Chrome's **F11** window fullscreen as a fallback and report whether it behaves differently.

Chrome requires a user click before tab capture. When a tab is captured, Chrome stops its normal audio output, so the extension explicitly routes captured audio back to the speakers. The extension uses an offscreen document to keep the audio graph running after the click. It needs Chrome 116 or newer.

This probe does **not** establish that Netflix or another protected service will expose usable audio. If Netflix yields `0` or `ERR`, capture availability must be resolved before porting the compressor. A later processing version also needs an audio/video synchronization design: putting the existing 2-second rolling buffer into this audio-only path would make sound two seconds late.

Sources: [Chrome tabCapture](https://developer.chrome.com/docs/extensions/reference/api/tabCapture), [Chrome offscreen documents](https://developer.chrome.com/docs/extensions/reference/api/offscreen).
