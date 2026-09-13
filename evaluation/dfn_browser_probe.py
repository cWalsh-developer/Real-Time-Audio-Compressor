"""Render an isolated stereo DeepFilterNet3 AudioWorklet in Chromium."""

import argparse
import hashlib
import json
import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import quote

from playwright.sync_api import sync_playwright


ASSETS = {
    "df_bg.wasm": "440b5d12b6ea7d95008736f844221d7874ee15de5cb10d3015002470fdba0432",
    "DeepFilterNet3_onnx.tar.gz": "c94d91f70911001c946e0fabb4aa9adc37045f45a03b56008cb0c8244cb63616",
}


RENDER = """async ({audioUrl, attenuationDb}) => {
  const wasmBytes = await (await fetch('/models/dfn-assets/df_bg.wasm')).arrayBuffer();
  const modelBytes = await (await fetch('/models/dfn-assets/DeepFilterNet3_onnx.tar.gz')).arrayBuffer();
  const wasmModule = await WebAssembly.compile(wasmBytes);
  const sourceContext = new AudioContext({sampleRate: 48000});
  const audioBytes = await (await fetch(audioUrl)).arrayBuffer();
  const input = await sourceContext.decodeAudioData(audioBytes);
  await sourceContext.close();
  const context = new OfflineAudioContext(2, input.length, 48000);
  await context.audioWorklet.addModule('/evaluation/dfn_probe_worklet.js');
  const source = context.createBufferSource();
  source.buffer = input;
  const processor = new AudioWorkletNode(context, 'dfn-stereo-probe', {
    outputChannelCount: [2],
    processorOptions: {wasmModule, modelBytes, attenuationDb},
  });
  source.connect(processor);
  processor.connect(context.destination);
  source.start();
  const start = performance.now();
  const result = await context.startRendering();
  const renderSeconds = (performance.now() - start) / 1000;
  const samples = result.getChannelData(0);
  const right = result.getChannelData(1);
  let outputPower = 0;
  let inputPower = 0;
  let stereoDifference = 0;
  let peak = 0;
  const original = input.getChannelData(0);
  for (let i = 0; i < samples.length; i++) {
    outputPower += samples[i] ** 2;
    inputPower += original[i] ** 2;
    stereoDifference += Math.abs(samples[i] - right[i]);
    peak = Math.max(peak, Math.abs(samples[i]), Math.abs(right[i]));
  }
  let bestLag = 0;
  let bestCorrelation = -Infinity;
  for (let lag = 0; lag <= 4800; lag += 8) {
    let correlation = 0;
    for (let i = 24000; i < Math.min(samples.length, 144000); i += 8) {
      correlation += original[i - lag] * samples[i];
    }
    if (correlation > bestCorrelation) {
      bestCorrelation = correlation;
      bestLag = lag;
    }
  }
  const coarseLag = bestLag;
  for (let lag = Math.max(0, coarseLag - 8); lag <= coarseLag + 8; lag++) {
    let correlation = 0;
    for (let i = 24000; i < Math.min(samples.length, 144000); i += 8) {
      correlation += original[i - lag] * samples[i];
    }
    if (correlation > bestCorrelation) {
      bestCorrelation = correlation;
      bestLag = lag;
    }
  }
  return {
    audioSeconds: input.duration,
    renderSeconds,
    renderTimePerAudioSecond: renderSeconds / input.duration,
    outputLevelChangeDb: 10 * Math.log10(outputPower / inputPower),
    stereoDifference: stereoDifference / samples.length,
    peak,
    estimatedDelayMs: bestLag / 48,
  };
}"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="*", type=Path, help="48 kHz stereo WAV files inside the project")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    for name, expected in ASSETS.items():
        asset = root / "models" / "dfn-assets" / name
        if not asset.is_file():
            parser.error(f"missing asset: {asset}; see evaluation/dfn-browser-findings.md")
        if hashlib.sha256(asset.read_bytes()).hexdigest() != expected:
            parser.error(f"asset SHA-256 mismatch: {asset}")
    inputs = args.inputs or [
        Path("audio/separation/service_middle/original.wav"),
        Path("audio/separation/tos_197_209.wav"),
        Path("audio/separation/tos_208_220.wav"),
    ]
    urls = []
    for path in inputs:
        resolved = path.resolve()
        if not resolved.is_file():
            parser.error(f"missing audio file: {resolved}")
        try:
            relative = resolved.relative_to(root)
        except ValueError:
            parser.error(f"audio file must be inside the project: {resolved}")
        urls.append("/" + quote(relative.as_posix(), safe="/"))
    class QuietHandler(SimpleHTTPRequestHandler):
        def log_message(self, *_args: object) -> None:
            pass

    handler = partial(QuietHandler, directory=str(root))
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(channel="chromium", headless=True)
            try:
                page = browser.new_page()
                page.goto(f"http://127.0.0.1:{server.server_port}/")
                for url in urls:
                    for attenuation_db in (12, 20, 50):
                        result = page.evaluate(RENDER, {"audioUrl": url, "attenuationDb": attenuation_db})
                        print(url, attenuation_db, json.dumps(result))
            finally:
                browser.close()
    finally:
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    main()
