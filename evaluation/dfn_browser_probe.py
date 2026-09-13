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


RENDER = """async ({audioUrl, attenuationDb, mode, download, trace}) => {
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
    processorOptions: {wasmModule, modelBytes, attenuationDb, mode, trace, totalSamples: input.length},
  });
  const tracePromise = trace ? new Promise((resolve) => {
    processor.port.onmessage = (event) => {
      if (event.data.trace) resolve(event.data.trace);
    };
  }) : null;
  source.connect(processor);
  processor.connect(context.destination);
  source.start();
  const start = performance.now();
  const result = await context.startRendering();
  const eventTrace = trace ? await tracePromise : null;
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
  const residualWindows = [];
  for (let begin = 24000; begin + 4800 < samples.length; begin += 4800) {
    let power = 0;
    for (let i = begin; i < begin + 4800; i++) {
      const residual = original[i - bestLag] - samples[i];
      power += residual * residual;
    }
    residualWindows.push(10 * Math.log10(Math.max(power / 4800, 1e-12)));
  }
  residualWindows.sort((a, b) => a - b);
  if (download) {
    const wav = new ArrayBuffer(44 + samples.length * 4);
    const view = new DataView(wav);
    const writeAscii = (offset, value) => {
      for (let i = 0; i < value.length; i++) view.setUint8(offset + i, value.charCodeAt(i));
    };
    writeAscii(0, 'RIFF'); view.setUint32(4, wav.byteLength - 8, true);
    writeAscii(8, 'WAVE'); writeAscii(12, 'fmt ');
    view.setUint32(16, 16, true); view.setUint16(20, 1, true);
    view.setUint16(22, 2, true); view.setUint32(24, 48000, true);
    view.setUint32(28, 192000, true); view.setUint16(32, 4, true);
    view.setUint16(34, 16, true); writeAscii(36, 'data');
    view.setUint32(40, samples.length * 4, true);
    for (let i = 0; i < samples.length; i++) {
      view.setInt16(44 + i * 4, Math.round(Math.max(-1, Math.min(32767 / 32768, samples[i])) * 32768), true);
      view.setInt16(46 + i * 4, Math.round(Math.max(-1, Math.min(32767 / 32768, right[i])) * 32768), true);
    }
    const link = document.createElement('a');
    link.href = URL.createObjectURL(new Blob([wav], {type: 'audio/wav'}));
    link.download = 'preview.wav';
    document.body.appendChild(link);
    link.click();
    link.remove();
    setTimeout(() => URL.revokeObjectURL(link.href), 1000);
  }
  return {
    audioSeconds: input.duration,
    renderSeconds,
    renderTimePerAudioSecond: renderSeconds / input.duration,
    outputLevelChangeDb: 10 * Math.log10(outputPower / inputPower),
    stereoDifference: stereoDifference / samples.length,
    peak,
    estimatedDelayMs: bestLag / 48,
    residualP50Db: residualWindows[Math.floor(residualWindows.length * 0.5)],
    residualP90Db: residualWindows[Math.floor(residualWindows.length * 0.9)],
    residualAboveMinus15Fraction: residualWindows.filter((db) => db > -15).length / residualWindows.length,
    trace: eventTrace,
  };
}"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="*", type=Path, help="48 kHz stereo WAV files inside the project")
    parser.add_argument("--output-dir", type=Path, help="save residual-gated preview WAV files here")
    parser.add_argument("--preview-only", action="store_true", help="run only the event-gated preview")
    parser.add_argument("--trace", action="store_true", help="show last four seconds of event-gate state")
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
    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
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
                page = browser.new_page(accept_downloads=bool(args.output_dir))
                page.goto(f"http://127.0.0.1:{server.server_port}/")
                for url in urls:
                    trials = ((50, "residual"),) if args.preview_only else (
                        (12, "enhanced"), (20, "enhanced"), (50, "enhanced"), (50, "residual"))
                    for attenuation_db, mode in trials:
                        params = {"audioUrl": url, "attenuationDb": attenuation_db, "mode": mode,
                                  "download": bool(args.output_dir and mode == "residual"),
                                  "trace": bool(args.trace and mode == "residual")}
                        if params["download"]:
                            with page.expect_download() as pending:
                                result = page.evaluate(RENDER, params)
                            url_path = Path(url)
                            name = url_path.parent.name + "_" + url_path.stem + "_residual_preview.wav"
                            pending.value.save_as(args.output_dir / name)
                        else:
                            result = page.evaluate(RENDER, params)
                        timeline = result.pop("trace", None)
                        print(url, attenuation_db, mode, json.dumps(result))
                        if timeline:
                            print(json.dumps([point for point in timeline if point["time"] >= 8]))
            finally:
                browser.close()
    finally:
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    main()
