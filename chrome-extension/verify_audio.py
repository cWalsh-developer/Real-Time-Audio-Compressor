"""Render the extension's actual AudioWorklet against dialogue/loud test signals."""

from pathlib import Path
from tempfile import TemporaryDirectory
import math

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

BURST_RENDER = """async () => {
  const rate = 48000;
  const context = new OfflineAudioContext(1, rate * 3, rate);
  await context.audioWorklet.addModule('loudness-reducer.js');
  const buffer = context.createBuffer(1, rate * 3, rate);
  const input = buffer.getChannelData(0);
  for (let i = 0; i < input.length; i++) {
    const time = i / rate;
    const amplitude = time >= 2 && time < 2.12 ? 0.95 : 0.15;
    input[i] = amplitude * Math.sin(2 * Math.PI * 440 * time);
  }
  const source = context.createBufferSource();
  const reducer = new AudioWorkletNode(context, 'loudness-reducer', {
    outputChannelCount: [2],
  });
  source.buffer = buffer;
  source.connect(reducer);
  reducer.connect(context.destination);
  source.start();
  const output = (await context.startRendering()).getChannelData(0);
  let dialogueDifference = 0;
  for (let i = rate; i < 2 * rate; i++) {
    dialogueDifference = Math.max(dialogueDifference, Math.abs(output[i] - input[i]));
  }
  let before = 0;
  let after = 0;
  let peak = 0;
  for (let i = Math.round(2.03 * rate); i < Math.round(2.10 * rate); i++) {
    before += input[i] ** 2;
    after += output[i] ** 2;
    peak = Math.max(peak, Math.abs(output[i]));
  }
  return {change: 10 * Math.log10(after / before), peak, dialogueDifference};
}"""

BASS_RENDER = """async () => {
  const rate = 48000;
  const context = new OfflineAudioContext(1, rate * 4, rate);
  await context.audioWorklet.addModule('loudness-reducer.js');
  const buffer = context.createBuffer(1, rate * 4, rate);
  const input = buffer.getChannelData(0);
  for (let i = 0; i < input.length; i++) {
    const time = i / rate;
    const amplitude = time >= 2 ? 0.7 : 0.15;
    input[i] = amplitude * Math.sin(2 * Math.PI * 60 * time);
  }
  const source = context.createBufferSource();
  const reducer = new AudioWorkletNode(context, 'loudness-reducer');
  source.buffer = buffer;
  source.connect(reducer);
  reducer.connect(context.destination);
  source.start();
  const output = (await context.startRendering()).getChannelData(0);
  const rms = (samples, start, end) => {
    let sum = 0;
    for (let i = start * rate; i < end * rate; i++) sum += samples[i] ** 2;
    return Math.sqrt(sum / ((end - start) * rate));
  };
  return 20 * Math.log10(rms(output, 3, 4) / rms(input, 3, 4));
}"""

BASS_GUITAR_RENDER = """async () => {
  const rate = 48000;
  const context = new OfflineAudioContext(1, rate * 4, rate);
  await context.audioWorklet.addModule('loudness-reducer.js');
  const buffer = context.createBuffer(1, rate * 4, rate);
  const input = buffer.getChannelData(0);
  for (let i = 0; i < input.length; i++) {
    const time = i / rate;
    const amplitude = time >= 2 ? 0.7 : 0.15;
    input[i] = amplitude * (Math.sin(2 * Math.PI * 60 * time)
      + 0.55 * Math.sin(2 * Math.PI * 440 * time));
  }
  const source = context.createBufferSource();
  const reducer = new AudioWorkletNode(context, 'loudness-reducer');
  source.buffer = buffer;
  source.connect(reducer);
  reducer.connect(context.destination);
  source.start();
  const output = (await context.startRendering()).getChannelData(0);
  const coefficient = (samples, frequency, start, end) => {
    let inPhase = 0;
    let quadrature = 0;
    for (let i = start * rate; i < end * rate; i++) {
      const phase = 2 * Math.PI * frequency * i / rate;
      inPhase += samples[i] * Math.sin(phase);
      quadrature += samples[i] * Math.cos(phase);
    }
    return 2 * Math.hypot(inPhase, quadrature) / ((end - start) * rate);
  };
  return {
    bassChange: 20 * Math.log10(coefficient(output, 60, 3, 4) / coefficient(input, 60, 3, 4)),
    guitarChange: 20 * Math.log10(coefficient(output, 440, 3, 4) / coefficient(input, 440, 3, 4)),
  };
}"""

CONSISTENCY_RENDER = """async () => {
  const rate = 48000;
  const context = new OfflineAudioContext(1, rate * 6, rate);
  await context.audioWorklet.addModule('loudness-reducer.js');
  const buffer = context.createBuffer(1, rate * 6, rate);
  const input = buffer.getChannelData(0);
  for (let i = 0; i < input.length; i++) {
    const time = i / rate;
    const bass = time >= 2 ? 0.7 : 0.15;
    const guitar = time < 2 ? 0.15 : 0.25 + 0.35 * (0.5 + 0.5 * Math.sin(2 * Math.PI * 0.8 * time));
    input[i] = bass * Math.sin(2 * Math.PI * 60 * time)
      + guitar * Math.sin(2 * Math.PI * 440 * time);
  }
  const source = context.createBufferSource();
  const reducer = new AudioWorkletNode(context, 'loudness-reducer');
  source.buffer = buffer;
  source.connect(reducer);
  reducer.connect(context.destination);
  source.start();
  const output = (await context.startRendering()).getChannelData(0);
  const rms = (samples, start, end) => {
    let sum = 0;
    for (let i = start * rate; i < end * rate; i++) sum += samples[i] ** 2;
    return Math.sqrt(sum / ((end - start) * rate));
  };
  const changes = [];
  for (let start = 3; start < 6; start += 0.5) {
    changes.push(20 * Math.log10(rms(output, start, start + 0.5) / rms(input, start, start + 0.5)));
  }
  return changes;
}"""

VOICE_RENDER = """async () => {
  const rate = 48000;
  const context = new OfflineAudioContext(1, rate * 6, rate);
  await context.audioWorklet.addModule('loudness-reducer.js');
  const buffer = context.createBuffer(1, rate * 6, rate);
  const input = buffer.getChannelData(0);
  for (let i = 0; i < input.length; i++) {
    const time = i / rate;
    const amplitude = time < 2 ? 0.15 : time < 4 ? 0.40 : 0.50;
    const syllable = 0.72 + 0.28 * Math.sin(2 * Math.PI * 5 * time) ** 2;
    input[i] = amplitude * syllable * (
      Math.sin(2 * Math.PI * 150 * time)
      + 0.55 * Math.sin(2 * Math.PI * 300 * time)
      + 0.25 * Math.sin(2 * Math.PI * 900 * time));
  }
  const source = context.createBufferSource();
  const reducer = new AudioWorkletNode(context, 'loudness-reducer');
  source.buffer = buffer;
  source.connect(reducer);
  reducer.connect(context.destination);
  source.start();
  const output = (await context.startRendering()).getChannelData(0);
  const rms = (samples, start, end) => {
    let sum = 0;
    for (let i = start * rate; i < end * rate; i++) sum += samples[i] ** 2;
    return Math.sqrt(sum / ((end - start) * rate));
  };
  return [20 * Math.log10(rms(output, 2.5, 3.5) / rms(input, 2.5, 3.5)),
    20 * Math.log10(rms(output, 4.5, 5.5) / rms(input, 4.5, 5.5))];
}"""

GRACE_RENDER = """async () => {
  const rate = 48000;
  const context = new OfflineAudioContext(1, rate * 6, rate);
  await context.audioWorklet.addModule('loudness-reducer.js');
  const buffer = context.createBuffer(1, rate * 6, rate);
  const input = buffer.getChannelData(0);
  for (let i = 0; i < input.length; i++) {
    const time = i / rate;
    let amplitude = 0.15;
    if (time >= 2 && time < 2.7) amplitude = 0.7;
    if (time >= 2.7 && time < 3.3) amplitude = 0.3;
    if (time >= 3.3 && time < 4.2) amplitude = 0.7;
    input[i] = amplitude * (Math.sin(2 * Math.PI * 60 * time)
      + 0.55 * Math.sin(2 * Math.PI * 440 * time));
  }
  const source = context.createBufferSource();
  const reducer = new AudioWorkletNode(context, 'loudness-reducer');
  source.buffer = buffer;
  source.connect(reducer);
  reducer.connect(context.destination);
  source.start();
  const output = (await context.startRendering()).getChannelData(0);
  const rms = (samples, start, end) => {
    let sum = 0;
    for (let i = start * rate; i < end * rate; i++) sum += samples[i] ** 2;
    return Math.sqrt(sum / ((end - start) * rate));
  };
  const changes = [];
  for (const start of [2.4, 2.9, 3.6]) {
    changes.push(20 * Math.log10(rms(output, start, start + 0.3) / rms(input, start, start + 0.3)));
  }
  return changes;
}"""

QUICK_GUITAR_RENDER = """async () => {
  const rate = 48000;
  const context = new OfflineAudioContext(1, rate * 5, rate);
  await context.audioWorklet.addModule('loudness-reducer.js');
  const buffer = context.createBuffer(1, rate * 5, rate);
  const input = buffer.getChannelData(0);
  for (let i = 0; i < input.length; i++) {
    const time = i / rate;
    const guitar = time >= 2.5 && time < 2.55 ? 0.9 : 0.25;
    const bass = time >= 2 ? 0.7 : 0.15;
    input[i] = bass * Math.sin(2 * Math.PI * 60 * time)
      + guitar * Math.sin(2 * Math.PI * 440 * time);
  }
  const source = context.createBufferSource();
  const reducer = new AudioWorkletNode(context, 'loudness-reducer');
  source.buffer = buffer;
  source.connect(reducer);
  reducer.connect(context.destination);
  source.start();
  const output = (await context.startRendering()).getChannelData(0);
  const coefficient = (samples, frequency, start, end) => {
    let inPhase = 0;
    let quadrature = 0;
    for (let i = start * rate; i < end * rate; i++) {
      const phase = 2 * Math.PI * frequency * i / rate;
      inPhase += samples[i] * Math.sin(phase);
      quadrature += samples[i] * Math.cos(phase);
    }
    return 2 * Math.hypot(inPhase, quadrature) / ((end - start) * rate);
  };
  return 20 * Math.log10(coefficient(output, 440, 2.5, 2.55)
    / coefficient(input, 440, 2.5, 2.55));
}"""

AI_WORKER_RENDER = """async () => {
  const worker = new Worker('ai-worker.js');
  try {
    const ready = await new Promise((resolve, reject) => {
      worker.onmessage = (event) => event.data.type === 'READY' ? resolve() : event.data.type === 'ERROR' ? reject(new Error(event.data.error)) : null;
      worker.postMessage({type: 'LOAD'});
    });
    await ready;
    const score = await new Promise((resolve, reject) => {
      worker.onmessage = (event) => event.data.type === 'SCORE' ? resolve(event.data.speechRatio) : event.data.type === 'ERROR' ? reject(new Error(event.data.error)) : null;
      const values = new Float32Array(257).fill(0.1);
      worker.postMessage({type: 'INFER', values}, [values.buffer]);
    });
    return score;
  } finally {
    worker.terminate();
  }
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
                burst = page.evaluate(BURST_RENDER)
                bass = page.evaluate(BASS_RENDER)
                bassGuitar = page.evaluate(BASS_GUITAR_RENDER)
                consistency = page.evaluate(CONSISTENCY_RENDER)
                voice = page.evaluate(VOICE_RENDER)
                grace = page.evaluate(GRACE_RENDER)
                quickGuitar = page.evaluate(QUICK_GUITAR_RENDER)
                aiScore = page.evaluate(AI_WORKER_RENDER)
            finally:
                browser.close()
    quiet, moderate, loud, loudest = result["changes"]
    assert all(abs(change) < 0.01 for change in result["changes"][:3]), result
    assert -2 < loudest < 0.5, result
    assert result["dialogueDifference"] < 1e-6, result
    assert result["stereoDifference"] > 0.01, result
    assert -8 < burst["change"] < -4 and burst["peak"] <= 0.8, burst
    assert burst["dialogueDifference"] < 1e-6, burst
    assert -16 < bass < -2, bass
    assert bassGuitar["bassChange"] < -2 and bassGuitar["guitarChange"] < bassGuitar["bassChange"] - 0.5, bassGuitar
    assert max(consistency) - min(consistency) < 3, consistency
    assert all(-2 < change < 0.5 for change in voice), voice
    assert all(math.isfinite(change) for change in grace) and max(grace) - min(grace) < 2.5, grace
    assert quickGuitar < -10, quickGuitar
    assert math.isfinite(aiScore) and 0 <= aiScore <= 1, aiScore
    print("Rendered gain changes (dB):", [round(value, 2) for value in result["changes"]])
    print("Burst change (dB):", round(burst["change"], 2))
    print("Bass change (dB):", round(bass, 2))
    print("Bass/guitar changes (dB):", {key: round(value, 2) for key, value in bassGuitar.items()})
    print("Consistency changes (dB):", [round(value, 2) for value in consistency])
    print("Voice changes (dB):", [round(value, 2) for value in voice])
    print("Grace changes (dB):", [round(value, 2) for value in grace])
    print("Quick guitar change (dB):", round(quickGuitar, 2))
    print("AI worker speech score:", round(aiScore, 4))
    print("Dialogue passthrough and stereo separation verified")


if __name__ == "__main__":
    main()
