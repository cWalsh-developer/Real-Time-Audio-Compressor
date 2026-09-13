// Isolated DeepFilterNet3 stereo feasibility probe. Not loaded by the extension.
import * as dfn from "../models/mezon-noise-suppression/src/df3/df.js";

class DfnStereoProbe extends AudioWorkletProcessor {
  constructor(options) {
    super();
    dfn.initSync({module: options.processorOptions.wasmModule});
    const model = new Uint8Array(options.processorOptions.modelBytes);
    const attenuationDb = options.processorOptions.attenuationDb ?? 50;
    this.handles = [dfn.df_create(model, attenuationDb), dfn.df_create(model, attenuationDb)];
    this.frameLength = dfn.df_get_frame_length(this.handles[0]);
    this.capacity = this.frameLength * 4;
    this.inputRings = [new Float32Array(this.capacity), new Float32Array(this.capacity)];
    this.outputRings = [new Float32Array(this.capacity), new Float32Array(this.capacity)];
    this.inWrite = 0;
    this.inRead = 0;
    this.inCount = 0;
    this.outWrite = 0;
    this.outRead = 0;
    this.outCount = 0;
    this.frames = [new Float32Array(this.frameLength), new Float32Array(this.frameLength)];
    this.port.postMessage({frameLength: this.frameLength});
  }

  process(inputs, outputs) {
    const input = inputs[0];
    const output = outputs[0];
    if (!output.length) return true;
    const blockLength = output[0].length;
    for (let i = 0; i < blockLength; i++) {
      for (let channel = 0; channel < 2; channel++) {
        this.inputRings[channel][this.inWrite] = input[Math.min(channel, input.length - 1)]?.[i] || 0;
      }
      this.inWrite = (this.inWrite + 1) % this.capacity;
      this.inCount++;
    }
    while (this.inCount >= this.frameLength) {
      for (let i = 0; i < this.frameLength; i++) {
        for (let channel = 0; channel < 2; channel++) {
          this.frames[channel][i] = this.inputRings[channel][this.inRead];
        }
        this.inRead = (this.inRead + 1) % this.capacity;
      }
      this.inCount -= this.frameLength;
      const processedLeft = dfn.df_process_frame(this.handles[0], this.frames[0]);
      const processedRight = dfn.df_process_frame(this.handles[1], this.frames[1]);
      for (let i = 0; i < this.frameLength; i++) {
        this.outputRings[0][this.outWrite] = processedLeft[i];
        this.outputRings[1][this.outWrite] = processedRight[i];
        this.outWrite = (this.outWrite + 1) % this.capacity;
      }
      this.outCount += this.frameLength;
    }
    for (let i = 0; i < blockLength; i++) {
      for (let channel = 0; channel < output.length; channel++) {
        output[channel][i] = this.outCount > 0
          ? this.outputRings[Math.min(channel, 1)][this.outRead]
          : 0;
      }
      if (this.outCount > 0) {
        this.outRead = (this.outRead + 1) % this.capacity;
        this.outCount--;
      }
    }
    return true;
  }
}

registerProcessor("dfn-stereo-probe", DfnStereoProbe);
