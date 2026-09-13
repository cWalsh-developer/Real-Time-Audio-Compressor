class LoudnessReducer extends AudioWorkletProcessor {
  constructor() {
    super();
    this.meanSquare = 0;
    this.baselineDb = -20;
    this.peakEnvelope = 0;
    this.sustainedGain = 1;
    this.transientGain = 1;
    this.heldReductionDb = 0;
    this.qualifyingSamples = 0;
    this.holdSamples = 0;
    this.qualifyFor = Math.round(sampleRate * 0.08);
    this.holdFor = Math.round(sampleRate * 0.35);
    this.levelSmoothing = Math.exp(-1 / (sampleRate * 0.03));
    this.baselineRise = 1 - Math.exp(-1 / (sampleRate * 5));
    this.baselineFall = 1 - Math.exp(-1 / (sampleRate * 1));
    this.peakRelease = Math.exp(-1 / (sampleRate * 0.08));
    this.sustainedAttack = Math.exp(-1 / (sampleRate * 0.05));
    this.sustainedRelease = Math.exp(-1 / (sampleRate * 0.25));
    this.transientAttack = Math.exp(-1 / (sampleRate * 0.003));
    this.transientRelease = Math.exp(-1 / (sampleRate * 0.12));
    this.reductionRelease = Math.exp(-1 / (sampleRate * 0.25));
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
      if (levelReduction >= 3) {
        this.qualifyingSamples++;
        if (this.qualifyingSamples >= this.qualifyFor) {
          this.heldReductionDb = Math.max(this.heldReductionDb, levelReduction);
          this.holdSamples = this.holdFor;
        }
      } else {
        this.qualifyingSamples = 0;
      }
      if (this.holdSamples > 0) {
        this.holdSamples--;
      } else {
        this.heldReductionDb *= this.reductionRelease;
      }
      const peakDb = 20 * Math.log10(Math.max(this.peakEnvelope, 1e-5));
      const peakReduction = Math.min(8, Math.max(0, (peakDb + 0.9) * 20));
      const sustainedTarget = 10 ** (-this.heldReductionDb / 20);
      const transientTarget = 10 ** (-peakReduction / 20);
      const sustainedCoefficient = sustainedTarget < this.sustainedGain
        ? this.sustainedAttack : this.sustainedRelease;
      const transientCoefficient = transientTarget < this.transientGain
        ? this.transientAttack : this.transientRelease;
      this.sustainedGain = sustainedCoefficient * this.sustainedGain
        + (1 - sustainedCoefficient) * sustainedTarget;
      this.transientGain = transientCoefficient * this.transientGain
        + (1 - transientCoefficient) * transientTarget;
      const gain = Math.min(this.sustainedGain, this.transientGain);

      for (let channel = 0; channel < output.length; channel++) {
        output[channel][frame] = (original[Math.min(channel, original.length - 1)]?.[frame] || 0)
          * gain * Math.min(1, 0.9 / Math.max(peak, 1e-5));
      }
    }
    return true;
  }
}

registerProcessor("loudness-reducer", LoudnessReducer);
