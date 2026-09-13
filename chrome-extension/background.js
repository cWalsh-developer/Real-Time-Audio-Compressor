const OFFSCREEN_PATH = "offscreen.html";
let creatingOffscreen;

async function ensureOffscreen() {
  const url = chrome.runtime.getURL(OFFSCREEN_PATH);
  const contexts = await chrome.runtime.getContexts({
    contextTypes: ["OFFSCREEN_DOCUMENT"],
    documentUrls: [url],
  });
  if (contexts.length) return;
  if (!creatingOffscreen) {
    creatingOffscreen = chrome.offscreen.createDocument({
      url: OFFSCREEN_PATH,
      reasons: ["USER_MEDIA", "AUDIO_PLAYBACK"],
      justification: "Capture the selected tab's audio and replay it through the user's speakers.",
    }).finally(() => { creatingOffscreen = undefined; });
  }
  await creatingOffscreen;
}

async function setBadge(tabId, text, color, title) {
  await chrome.action.setBadgeText({tabId, text});
  await chrome.action.setBadgeBackgroundColor({tabId, color});
  await chrome.action.setTitle({tabId, title});
}

chrome.action.onClicked.addListener(async (tab) => {
  if (tab.id === undefined) return;
  try {
    const captured = await chrome.tabCapture.getCapturedTabs();
    const alreadyActive = captured.some((item) => item.tabId === tab.id && item.status === "active");
    if (alreadyActive) {
      await chrome.runtime.sendMessage({target: "offscreen", type: "STOP", tabId: tab.id});
      await setBadge(tab.id, "", "#555555", "Click to start tab audio capture");
      return;
    }

    await ensureOffscreen();
    const streamId = await chrome.tabCapture.getMediaStreamId({targetTabId: tab.id});
    const result = await chrome.runtime.sendMessage({
      target: "offscreen", type: "START", tabId: tab.id, streamId,
    });
    if (!result?.ok) throw new Error(result?.error || "Audio capture did not start");
    await setBadge(tab.id, "...", "#777777", "Listening for tab audio");
  } catch (error) {
    console.error("Adaptive Audio capture failed", error);
    await setBadge(tab.id, "ERR", "#b42318", `Capture failed: ${error.message}`);
  }
});

chrome.runtime.onMessage.addListener((message) => {
  if (message.target !== "background") return;
  if (message.type === "LEVEL") {
    const audible = message.rms > 0.0001;
    void setBadge(
      message.tabId,
      audible ? "AUD" : "0",
      audible ? "#137333" : "#b26a00",
      audible ? "Tab audio is reaching the extension" : "Capture is active; no audio detected",
    );
  } else if (message.type === "STOPPED") {
    void setBadge(message.tabId, "", "#555555", "Click to start tab audio capture");
  }
});
