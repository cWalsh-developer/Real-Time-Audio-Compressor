"""Render the extension's actual AudioWorklet against dialogue/loud test signals."""

from pathlib import Path
from tempfile import TemporaryDirectory

from playwright.sync_api import sync_playwright


RENDER = """async () => {
  const rate = 48000;
  const amplitudes = [0.15, 0.30, 0.40, 0.50];
  const context = new OfflineAudioContext(2, rate * 8, rate);
  await context.audioWorklet.addModule('loudness-reducer.js');
  const buffer = context.createBuffer(2, rate * 8, rate);
  for (let channel = 0; channel < 2; channel++) {
    const samples = buffer.getChannelData(channel);
    for (let i = 0; i < samples.length; i++) {
      const time = i / rate;
      samples[i] = amplitudes[Math.floor(time / 2)]
        * Math.sin(2 * Math.PI * (channel ? 660 : 440) * time);
    }
  }
  const source = context.createBufferSource();
  const reducer = new AudioWorkletNode(context, 'loudness-reducer', {
    outputChannelCount: [2],
  });
  source.buffer = buffer;
  source.connect(reducer);
  reducer.connect(context.destination);
  source.start();
  const rendered = await context.startRendering();
  const original = buffer.getChannelData(0);
  const left = rendered.getChannelData(0);
  const right = rendered.getChannelData(1);
  const rms = (samples, start, end) => {
    let sum = 0;
    for (let i = start * rate; i < end * rate; i++) sum += samples[i] ** 2;
    return Math.sqrt(sum / ((end - start) * rate));
  };
  const changes = amplitudes.map((_, index) => 20 * Math.log10(
    rms(left, 2 * index + 1, 2 * index + 2)
    / rms(original, 2 * index + 1, 2 * index + 2),
  ));
  let dialogueDifference = 0;
  let stereoDifference = 0;
  for (let i = rate; i < 2 * rate; i++) {
    dialogueDifference = Math.max(dialogueDifference, Math.abs(left[i] - original[i]));
    stereoDifference += Math.abs(left[i] - right[i]);
  }
  return {changes, dialogueDifference, stereoDifference: stereoDifference / rate};
}"""


def main() -> None:
    extension = Path(__file__).resolve().parent
    workspace = extension.parent.resolve()
    with TemporaryDirectory(prefix="chrome-extension-check-", dir=workspace) as profile:
        assert Path(profile).resolve().parent == workspace
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch_persistent_context(
                profile,
                channel="chromium",
                headless=True,
                args=[
                    f"--disable-extensions-except={extension}",
                    f"--load-extension={extension}",
                ],
            )
            try:
                worker = browser.service_workers[0] if browser.service_workers else browser.wait_for_event(
                    "serviceworker", timeout=10000
                )
                page = browser.new_page()
                page.goto(worker.url.replace("background.js", "offscreen.html"))
                result = page.evaluate(RENDER)
            finally:
                browser.close()
    quiet, moderate, loud, loudest = result["changes"]
    assert abs(quiet) < 0.01 and abs(moderate) < 0.01, result
    assert -11 < loudest < -10 and loud < -6, result
    assert result["dialogueDifference"] < 1e-6, result
    assert result["stereoDifference"] > 0.01, result
    print("Rendered gain changes (dB):", [round(value, 2) for value in result["changes"]])
    print("Dialogue passthrough and stereo separation verified")


if __name__ == "__main__":
    main()
