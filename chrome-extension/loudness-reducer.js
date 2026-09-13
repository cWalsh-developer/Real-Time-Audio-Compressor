class LoudnessReducer extends AudioWorkletProcessor {
  constructor() {
    super();
    this.meanSquare = 0;
    this.gain = 1;
    this.levelAttack = Math.exp(-1 / (sampleRate * 0.05));
    this.levelRelease = Math.exp(-1 / (sampleRate * 0.3));
    this.gainAttack = Math.exp(-1 / (sampleRate * 0.04));
    this.gainRelease = Math.exp(-1 / (sampleRate * 0.5));
  }

  process(inputs, outputs) {
    const compressed = inputs[0];
    const original = inputs[1];
    const output = outputs[0];
    if (!output.length) return true;

    for (let frame = 0; frame < output[0].length; frame++) {
      let power = 0;
      for (const channel of original) {
        const sample = channel[frame] || 0;
        power += sample * sample;
      }
      power /= Math.max(1, original.length);
      const levelCoefficient = power > this.meanSquare ? this.levelAttack : this.levelRelease;
      this.meanSquare = levelCoefficient * this.meanSquare + (1 - levelCoefficient) * power;

      const levelDb = 10 * Math.log10(Math.max(this.meanSquare, 1e-10));
      const reductionDb = Math.min(4, Math.max(0, levelDb + 17));
      const targetGain = 10 ** (-reductionDb / 20);
      const gainCoefficient = targetGain < this.gain ? this.gainAttack : this.gainRelease;
      this.gain = gainCoefficient * this.gain + (1 - gainCoefficient) * targetGain;

      for (let channel = 0; channel < output.length; channel++) {
        output[channel][frame] = (compressed[channel]?.[frame] || 0) * this.gain;
      }
    }
    return true;
  }
}

registerProcessor("loudness-reducer", LoudnessReducer);
