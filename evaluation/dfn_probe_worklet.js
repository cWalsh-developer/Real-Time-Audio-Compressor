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
    this.mode = options.processorOptions.mode ?? "enhanced";
    this.reductionGain = 10 ** (-(options.processorOptions.reductionDb ?? 6) / 20);
    this.dryDelaySamples = 1888;
    this.dryHistory = [new Float32Array(4096), new Float32Array(4096)];
    this.drySamples = new Float32Array(2);
    this.enhancedSamples = new Float32Array(2);
    this.inputSampleNumber = 0;
    this.outputSampleNumber = 0;
    this.residualMeanSquare = 0;
    this.speechMeanSquare = 0;
    this.residualSmoothing = Math.exp(-1 / (sampleRate * 0.1));
    this.gain = 1;
    this.gainAttack = Math.exp(-1 / (sampleRate * 0.015));
    this.gainRelease = Math.exp(-1 / (sampleRate * 0.5));
    this.holdSamples = 0;
    this.eventActive = false;
    this.traceEnabled = options.processorOptions.trace ?? false;
    this.totalSamples = options.processorOptions.totalSamples ?? 0;
    this.trace = [];
    this.port.postMessage({frameLength: this.frameLength});
  }

  process(inputs, outputs) {
    const input = inputs[0];
    const output = outputs[0];
    if (!output.length) return true;
    const blockLength = output[0].length;
    for (let i = 0; i < blockLength; i++) {
      for (let channel = 0; channel < 2; channel++) {
        const sample = input[Math.min(channel, input.length - 1)]?.[i] || 0;
        this.inputRings[channel][this.inWrite] = sample;
        this.dryHistory[channel][this.inputSampleNumber % this.dryHistory[channel].length] = sample;
      }
      this.inWrite = (this.inWrite + 1) % this.capacity;
      this.inCount++;
      this.inputSampleNumber++;
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
      const delayedIndex = this.outputSampleNumber - this.dryDelaySamples;
      let residualPower = 0;
      let speechPower = 0;
      for (let channel = 0; channel < 2; channel++) {
        this.drySamples[channel] = delayedIndex >= 0
          ? this.dryHistory[channel][delayedIndex % this.dryHistory[channel].length] : 0;
        this.enhancedSamples[channel] = this.outCount > 0
          ? this.outputRings[channel][this.outRead] : 0;
        const difference = this.drySamples[channel] - this.enhancedSamples[channel];
        residualPower += difference * difference / 2;
        speechPower += this.enhancedSamples[channel] * this.enhancedSamples[channel] / 2;
      }
      this.residualMeanSquare = this.residualSmoothing * this.residualMeanSquare
        + (1 - this.residualSmoothing) * residualPower;
      this.speechMeanSquare = this.residualSmoothing * this.speechMeanSquare
        + (1 - this.residualSmoothing) * speechPower;
      const residualDb = 10 * Math.log10(Math.max(this.residualMeanSquare, 1e-10));
      const speechDb = 10 * Math.log10(Math.max(this.speechMeanSquare, 1e-10));
      if (residualDb >= -15) {
        this.eventActive = true;
        this.holdSamples = Math.round(sampleRate * 0.6);
      } else if (this.holdSamples > 0) {
        this.holdSamples--;
      } else if (residualDb < -20) {
        this.eventActive = false;
      }
      const targetGain = this.eventActive ? this.reductionGain : 1;
      const coefficient = targetGain < this.gain ? this.gainAttack : this.gainRelease;
      this.gain = coefficient * this.gain + (1 - coefficient) * targetGain;
      if (this.traceEnabled && this.outputSampleNumber % 4800 === 0) {
        this.trace.push({time: this.outputSampleNumber / sampleRate, residualDb, speechDb,
          active: this.eventActive, holdMs: this.holdSamples / sampleRate * 1000, gain: this.gain});
      }
      for (let channel = 0; channel < output.length; channel++) {
        const index = Math.min(channel, 1);
        output[channel][i] = this.mode === "residual"
          ? this.drySamples[index] + (this.gain - 1) * (this.drySamples[index] - this.enhancedSamples[index])
          : this.enhancedSamples[index];
      }
      if (this.outCount > 0) {
        this.outRead = (this.outRead + 1) % this.capacity;
        this.outCount--;
      }
      this.outputSampleNumber++;
      if (this.traceEnabled && this.outputSampleNumber === this.totalSamples) {
        this.port.postMessage({trace: this.trace});
      }
    }
    return true;
  }
}

registerProcessor("dfn-stereo-probe", DfnStereoProbe);
