let activeCapture;
let aiWorker;

function startAiWorker(worklet) {
  try {
    aiWorker = new Worker("ai-worker.js");
    aiWorker.onmessage = (event) => {
      if (event.data.type === "SCORE") worklet.port.postMessage({type: "AI_SCORE", speechRatio: event.data.speechRatio});
    };
    aiWorker.postMessage({type: "LOAD"});
  } catch (error) {
    console.warn("Adaptive Audio AI unavailable", error);
  }
}

async function stopCapture(tabId) {
  if (!activeCapture || (tabId !== undefined && activeCapture.tabId !== tabId)) return;
  const capture = activeCapture;
  activeCapture = undefined;
  clearInterval(capture.timer);
  aiWorker?.terminate();
  aiWorker = undefined;
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
    await context.audioWorklet.addModule("loudness-reducer.js");
    const source = context.createMediaStreamSource(stream);
    const dry = context.createGain();
    const wet = context.createGain();
    const loudnessReducer = new AudioWorkletNode(context, "loudness-reducer", {
      outputChannelCount: [2],
    });
    dry.gain.value = 0;
    wet.gain.value = 1;
    const analyser = context.createAnalyser();
    analyser.fftSize = 512;
    source.connect(dry);
    source.connect(loudnessReducer);
    dry.connect(analyser);
    loudnessReducer.connect(wet);
    wet.connect(analyser);
    analyser.connect(context.destination);
    startAiWorker(loudnessReducer);
    await context.resume();

    const samples = new Float32Array(analyser.fftSize);
    const timer = setInterval(() => {
      analyser.getFloatTimeDomainData(samples);
      let sumSquares = 0;
      for (const sample of samples) sumSquares += sample * sample;
      const rms = Math.sqrt(sumSquares / samples.length);
      chrome.runtime.sendMessage({target: "background", type: "LEVEL", tabId, rms, processed: activeCapture?.processed ?? false}).catch(() => {});
      const frequencies = new Float32Array(analyser.frequencyBinCount + 1);
      analyser.getFloatFrequencyData(frequencies);
      for (let index = 0; index < analyser.frequencyBinCount; index++) {
        frequencies[index] = 10 ** (frequencies[index] / 20);
      }
      frequencies[analyser.frequencyBinCount] = 0;
      aiWorker?.postMessage({type: "INFER", values: frequencies}, [frequencies.buffer]);
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
  for (const [gain, target] of [[capture.dry.gain, capture.processed ? 0 : 1], [capture.wet.gain, capture.processed ? 1 : 0]]) {
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
