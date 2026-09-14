let session;

self.onmessage = async (event) => {
  const {type, values} = event.data || {};
  if (type === "LOAD") {
    try {
      importScripts("vendor/ort.min.js");
      ort.env.wasm.wasmPaths = new URL("vendor/", self.location.href).href;
      session = await ort.InferenceSession.create("spectral-mask.onnx", {
        executionProviders: ["wasm"],
        graphOptimizationLevel: "all",
      });
      self.postMessage({type: "READY"});
    } catch (error) {
      self.postMessage({type: "ERROR", error: error.message});
    }
    return;
  }
  if (type !== "INFER" || !session || !values?.length) return;
  try {
    const input = new Float32Array(values);
    const tensor = new ort.Tensor("float32", input, [1, input.length, 1]);
    const output = await session.run({mixture_magnitude: tensor});
    const mask = output.speech_mask.data;
    let total = 0;
    let speech = 0;
    for (let index = 0; index < input.length; index++) {
      total += input[index];
      speech += input[index] * mask[index];
    }
    self.postMessage({type: "SCORE", speechRatio: speech / Math.max(total, 1e-8)});
  } catch (error) {
    self.postMessage({type: "ERROR", error: error.message});
  }
};