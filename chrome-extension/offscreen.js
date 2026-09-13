let activeCapture;

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
    const analyser = context.createAnalyser();
    analyser.fftSize = 2048;
    source.connect(analyser);
    analyser.connect(context.destination);
    await context.resume();

    const samples = new Float32Array(analyser.fftSize);
    const timer = setInterval(() => {
      analyser.getFloatTimeDomainData(samples);
      let sumSquares = 0;
      for (const sample of samples) sumSquares += sample * sample;
      const rms = Math.sqrt(sumSquares / samples.length);
      chrome.runtime.sendMessage({target: "background", type: "LEVEL", tabId, rms}).catch(() => {});
    }, 500);
    activeCapture = {tabId, stream, context, timer};
    stream.getAudioTracks().forEach((track) => {
      track.addEventListener("ended", () => { void stopCapture(tabId); }, {once: true});
    });
  } catch (error) {
    stream.getTracks().forEach((track) => track.stop());
    await context.close();
    throw error;
  }
}

chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
  if (message.target !== "offscreen") return;
  const operation = message.type === "START"
    ? startCapture(message.tabId, message.streamId)
    : message.type === "STOP" ? stopCapture(message.tabId) : Promise.reject(new Error("Unknown command"));
  operation.then(() => sendResponse({ok: true}))
    .catch((error) => sendResponse({ok: false, error: error.message}));
  return true;
});
