class LoudnessReducer extends AudioWorkletProcessor {
  constructor() {
    super();
    this.meanSquare = 0;
    this.bassMeanSquare = 0;
    this.upperMeanSquare = 0;
    this.bassEnvelope = 0;
    this.bassProgram = 0;
    this.lowpass = [];
    this.baselineDb = -20;
    this.peakEnvelope = 0;
    this.gain = 1;
    this.levelSmoothing = Math.exp(-1 / (sampleRate * 0.03));
    this.bassSmoothing = Math.exp(-1 / (sampleRate * 0.025));
    this.bassRelease = Math.exp(-1 / (sampleRate * 0.06));
    this.bassFilter = 1 - Math.exp(-2 * Math.PI * 140 / sampleRate);
    this.upperSmoothing = Math.exp(-1 / (sampleRate * 0.22));
    this.bassProgramSmoothing = Math.exp(-1 / (sampleRate * 0.18));
    this.upperGain = 1;
    this.reductionHoldDb = 0;
    this.reductionHoldSamples = 0;
    this.reductionHoldDuration = Math.round(sampleRate * 0.9);
    this.baselineRise = 1 - Math.exp(-1 / (sampleRate * 5));
    this.baselineFall = 1 - Math.exp(-1 / (sampleRate * 1));
    this.peakRelease = Math.exp(-1 / (sampleRate * 0.08));
    this.gainAttack = Math.exp(-1 / (sampleRate * 0.005));
    this.gainRelease = Math.exp(-1 / (sampleRate * 0.4));
    this.upperGainRelease = Math.exp(-1 / (sampleRate * 1.2));
  }

  process(inputs, outputs) {
    const original = inputs[0];
    const output = outputs[0];
    if (!output.length) return true;

    for (let frame = 0; frame < output[0].length; frame++) {
      let power = 0;
      let bassPower = 0;
      let upperPower = 0;
      let peak = 0;
      const bassValues = [];
      for (let channelIndex = 0; channelIndex < original.length; channelIndex++) {
        const channel = original[channelIndex];
        const sample = channel[frame] || 0;
        const previousBass = this.lowpass[channelIndex] || 0;
        const bass = previousBass + this.bassFilter * (sample - previousBass);
        this.lowpass[channelIndex] = bass;
        bassValues[channelIndex] = bass;
        power += sample * sample;
        bassPower += bass * bass;
        upperPower += (sample - bass) ** 2;
        peak = Math.max(peak, Math.abs(sample));
      }
      power /= Math.max(1, original.length);
      bassPower /= Math.max(1, original.length);
      this.meanSquare = this.levelSmoothing * this.meanSquare + (1 - this.levelSmoothing) * power;
      this.bassMeanSquare = this.bassSmoothing * this.bassMeanSquare + (1 - this.bassSmoothing) * bassPower;
      this.upperMeanSquare = this.upperSmoothing * this.upperMeanSquare + (1 - this.upperSmoothing) * upperPower / Math.max(1, original.length);
      this.bassEnvelope = Math.max(Math.sqrt(bassPower), this.bassEnvelope * this.bassRelease);
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
      const flattenReduction = levelDb > -14
        ? Math.min(12, Math.max(0, (levelDb + 14) * 2.2))
        : 0;
      const nightReduction = levelDb > -18
        ? Math.min(14, Math.max(0, (levelDb + 18) * 2.5))
        : 0;
      const peakDb = 20 * Math.log10(Math.max(this.peakEnvelope, 1e-5));
      const peakReduction = Math.min(6, Math.max(0, (peakDb + 3) * 3));
      const transientReduction = peakDb > -1 ? 12 : peakReduction;
      const bassDb = 10 * Math.log10(Math.max(this.bassMeanSquare, 1e-10));
      const bassLevelReduction = Math.min(7, Math.max(0, (bassDb + 18) * 1.5));
      const bassPeakDb = 20 * Math.log10(Math.max(this.bassEnvelope, 1e-5));
      const bassPeakReduction = Math.min(5, Math.max(0, (bassPeakDb + 6) * 1.5));
      const bassReduction = Math.max(bassLevelReduction, bassPeakReduction);
      const upperDb = 10 * Math.log10(Math.max(this.upperMeanSquare, 1e-10));
      const bassDominance = this.bassMeanSquare / Math.max(this.meanSquare, 1e-10);
      const bassProgramTarget = bassDb > -18 && bassDominance > 0.45 ? 1 : 0;
      this.bassProgram = this.bassProgramSmoothing * this.bassProgram
        + (1 - this.bassProgramSmoothing) * bassProgramTarget;
      const speechPresence = bassDominance < 0.45 && upperDb > -24;
      const upperReduction = this.bassProgram > 0.5 && !speechPresence
        ? Math.min(9, Math.max(0, (upperDb + 15) * 3))
        : 0;
      const requestedReduction = Math.max(levelReduction, flattenReduction, nightReduction, peakReduction, bassReduction);
      let reductionDb;
      if (speechPresence) {
        this.reductionHoldDb = 0;
        this.reductionHoldSamples = 0;
        reductionDb = Math.max(Math.min(1.5, levelReduction), transientReduction);
      } else {
        if (requestedReduction >= this.reductionHoldDb) {
          this.reductionHoldDb = requestedReduction;
          this.reductionHoldSamples = this.reductionHoldDuration;
        } else if (this.reductionHoldSamples > 0) {
          this.reductionHoldSamples--;
        } else {
          this.reductionHoldDb = requestedReduction;
        }
        reductionDb = Math.max(requestedReduction, this.reductionHoldDb);
      }
      const upperTargetGain = 10 ** (-upperReduction / 20);
      const upperCoefficient = upperTargetGain < this.upperGain ? this.gainAttack : this.upperGainRelease;
      this.upperGain = upperCoefficient * this.upperGain + (1 - upperCoefficient) * upperTargetGain;
      const targetGain = 10 ** (-reductionDb / 20);
      const gainCoefficient = targetGain < this.gain ? this.gainAttack : this.gainRelease;
      this.gain = gainCoefficient * this.gain + (1 - gainCoefficient) * targetGain;

      for (let channel = 0; channel < output.length; channel++) {
        const sourceChannel = original[Math.min(channel, original.length - 1)];
        const sample = sourceChannel?.[frame] || 0;
        const bass = bassValues[Math.min(channel, bassValues.length - 1)] || 0;
        output[channel][frame] = (bass + (sample - bass) * this.upperGain)
          * this.gain
          * Math.min(1, 0.8 / Math.max(peak, 1e-5));
      }
    }
    return true;
  }
}

registerProcessor("loudness-reducer", LoudnessReducer);
