class LoudnessReducer extends AudioWorkletProcessor {
  constructor() {
    super();
    this.meanSquare = 0;
    this.baselineDb = -20;
    this.peakEnvelope = 0;
    this.gain = 1;
    this.levelSmoothing = Math.exp(-1 / (sampleRate * 0.03));
    this.baselineRise = 1 - Math.exp(-1 / (sampleRate * 5));
    this.baselineFall = 1 - Math.exp(-1 / (sampleRate * 1));
    this.peakRelease = Math.exp(-1 / (sampleRate * 0.08));
    this.gainAttack = Math.exp(-1 / (sampleRate * 0.005));
    this.gainRelease = Math.exp(-1 / (sampleRate * 0.4));
  }

  process(inputs, outputs) {
    const original = inputs[0];
    const output = outputs[0];
    if (!output.length) return true;

    for (let frame = 0; frame < output[0].length; frame++) {
      let power = 0;
      let peak = 0;
      for (const channel of original) {
        const sample = channel[frame] || 0;
        power += sample * sample;
        peak = Math.max(peak, Math.abs(sample));
      }
      power /= Math.max(1, original.length);
      this.meanSquare = this.levelSmoothing * this.meanSquare + (1 - this.levelSmoothing) * power;
      this.peakEnvelope = Math.max(peak, this.peakEnvelope * this.peakRelease);

      const levelDb = 10 * Math.log10(Math.max(this.meanSquare, 1e-10));
      if (levelDb < -10.5 || levelDb < this.baselineDb + 6) {
        const baselineTarget = Math.max(-30, Math.min(levelDb, -18));
        const coefficient = baselineTarget < this.baselineDb ? this.baselineFall : this.baselineRise;
        this.baselineDb += (baselineTarget - this.baselineDb) * coefficient;
      }
      const levelReduction = levelDb > -10.5
        ? Math.min(10.5, Math.max(0, (levelDb - this.baselineDb - 6) * 3.5))
        : 0;
      const peakDb = 20 * Math.log10(Math.max(this.peakEnvelope, 1e-5));
      const peakReduction = Math.min(6, Math.max(0, (peakDb + 3) * 3));
      const reductionDb = Math.max(levelReduction, peakReduction);
      const targetGain = 10 ** (-reductionDb / 20);
      const gainCoefficient = targetGain < this.gain ? this.gainAttack : this.gainRelease;
      this.gain = gainCoefficient * this.gain + (1 - gainCoefficient) * targetGain;

      for (let channel = 0; channel < output.length; channel++) {
        output[channel][frame] = (original[Math.min(channel, original.length - 1)]?.[frame] || 0)
          * this.gain * Math.min(1, 0.8 / Math.max(peak, 1e-5));
      }
    }
    return true;
  }
}

registerProcessor("loudness-reducer", LoudnessReducer);
