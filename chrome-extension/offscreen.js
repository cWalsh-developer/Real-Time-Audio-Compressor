let activeCapture;
const COMPRESSION = Object.freeze({
  threshold: -30,
  knee: 6,
  ratio: 6,
  attack: 0.005,
  release: 0.3,
  outputGain: 0.3,
});

async function stopCapture(tabId) {
  if (!activeCapture || (tabId !== undefined && activeCapture.tabId !== tabId)) return;
  const capture = activeCapture;
  activeCapture = undefined;
  clearInterval(capture.timer);
  capture.stream.getTracks().forEach((track) => track.stop());
  await capture.context.close();
  chrome.runtime.sendMessage({target: "background", type: "STOPPED", tabId: capture.tabId}).catch(() => {});
}

async function startCapture(tabId, streamId) {
  await stopCapture();
  const stream = await navigator.mediaDevices.getUserMedia({
    audio: {mandatory: {chromeMediaSource: "tab", chromeMediaSourceId: streamId}},
    video: false,
  });
  const context = new AudioContext({latencyHint: "interactive"});
  try {
    const source = context.createMediaStreamSource(stream);
    const dry = context.createGain();
    const wet = context.createGain();
    const compressor = context.createDynamicsCompressor();
    compressor.threshold.value = COMPRESSION.threshold;
    compressor.knee.value = COMPRESSION.knee;
    compressor.ratio.value = COMPRESSION.ratio;
    compressor.attack.value = COMPRESSION.attack;
    compressor.release.value = COMPRESSION.release;
    dry.gain.value = 0;
    wet.gain.value = COMPRESSION.outputGain;
    const analyser = context.createAnalyser();
    analyser.fftSize = 2048;
    source.connect(dry);
    source.connect(compressor);
    dry.connect(analyser);
    compressor.connect(wet);
    wet.connect(analyser);
    analyser.connect(context.destination);
    await context.resume();

    const samples = new Float32Array(analyser.fftSize);
    const timer = setInterval(() => {
      analyser.getFloatTimeDomainData(samples);
      let sumSquares = 0;
      for (const sample of samples) sumSquares += sample * sample;
      const rms = Math.sqrt(sumSquares / samples.length);
      chrome.runtime.sendMessage({target: "background", type: "LEVEL", tabId, rms, processed: activeCapture?.processed ?? false}).catch(() => {});
    }, 500);
    activeCapture = {tabId, stream, context, timer, dry, wet, processed: true};
    stream.getAudioTracks().forEach((track) => {
      track.addEventListener("ended", () => { void stopCapture(tabId); }, {once: true});
    });
  } catch (error) {
    stream.getTracks().forEach((track) => track.stop());
    await context.close();
    throw error;
  }
}

function toggleProcessing(tabId) {
  if (!activeCapture || activeCapture.tabId !== tabId) throw new Error("Start capture on this tab first");
  const capture = activeCapture;
  capture.processed = !capture.processed;
  const now = capture.context.currentTime;
  for (const [gain, target] of [[capture.dry.gain, capture.processed ? 0 : 1], [capture.wet.gain, capture.processed ? COMPRESSION.outputGain : 0]]) {
    gain.cancelScheduledValues(now);
    gain.setTargetAtTime(target, now, 0.02);
  }
  return capture.processed;
}

chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
  if (message.target !== "offscreen") return;
  const operation = message.type === "START"
    ? startCapture(message.tabId, message.streamId)
    : message.type === "STOP" ? stopCapture(message.tabId)
      : message.type === "TOGGLE" ? Promise.resolve().then(() => toggleProcessing(message.tabId))
        : Promise.reject(new Error("Unknown command"));
  operation.then((processed) => sendResponse({ok: true, processed}))
    .catch((error) => sendResponse({ok: false, error: error.message}));
  return true;
});
