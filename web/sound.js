/* sound.js — optional sound effects, OFF until the visitor turns them on.
   Everything is synthesized live with the Web Audio API, so there are no sound files:
     engine rumble   low-passed noise, on while the main engine fires in the simulation
     thruster hiss   band-passed noise, on while a side thruster fires
     countdown       short beeps, a higher one on LAUNCH
     touchdown       a falling low thud
     crash           a noise burst that sweeps down, plus a low boom */

export class Sound {
  constructor() {
    this.enabled = false;
    this.ctx = null;
  }

  async setEnabled(on) {
    this.enabled = on;
    if (on) {
      if (!this.ctx) this.build();
      await this.ctx.resume();
      this.master.gain.setTargetAtTime(0.7, this.ctx.currentTime, 0.05);
    } else if (this.ctx) {
      this.engine(false, false);
      this.master.gain.setTargetAtTime(0, this.ctx.currentTime, 0.03);
    }
  }

  build() {
    const ctx = (this.ctx = new (window.AudioContext || window.webkitAudioContext)());
    this.master = ctx.createGain();
    this.master.gain.value = 0;
    this.master.connect(ctx.destination);

    const noise = brown => {
      const len = ctx.sampleRate * 2, buf = ctx.createBuffer(1, len, ctx.sampleRate), d = buf.getChannelData(0);
      let last = 0;
      for (let i = 0; i < len; i++) {
        const w = Math.random() * 2 - 1;
        if (brown) { last = (last + 0.02 * w) / 1.02; d[i] = last * 3.5; } else d[i] = w;
      }
      return buf;
    };
    this.white = noise(false);
    const loop = (buf, type, freq, q) => {
      const src = ctx.createBufferSource(), filter = ctx.createBiquadFilter(), gain = ctx.createGain();
      src.buffer = buf; src.loop = true;
      filter.type = type; filter.frequency.value = freq; filter.Q.value = q;
      gain.gain.value = 0;
      src.connect(filter).connect(gain).connect(this.master);
      src.start();
      return gain;
    };
    this.rumble = loop(noise(true), "lowpass", 240, 0.7);
    this.hiss = loop(this.white, "bandpass", 2600, 0.9);
  }

  /** Called every frame with what the simulation's engines are doing right now. */
  engine(main, side) {
    if (!this.ctx) return;
    const t = this.ctx.currentTime;
    this.rumble.gain.setTargetAtTime(this.enabled && main ? 0.9 : 0, t, 0.04);
    this.hiss.gain.setTargetAtTime(this.enabled && side ? 0.05 : 0, t, 0.02);
  }

  tone(freq, dur, peak = 0.15, type = "sine", endFreq = freq) {
    if (!this.enabled || !this.ctx) return;
    const ctx = this.ctx, t = ctx.currentTime, osc = ctx.createOscillator(), gain = ctx.createGain();
    osc.type = type;
    osc.frequency.setValueAtTime(freq, t);
    osc.frequency.exponentialRampToValueAtTime(endFreq, t + dur);
    gain.gain.setValueAtTime(0.0001, t);
    gain.gain.exponentialRampToValueAtTime(peak, t + 0.008);
    gain.gain.exponentialRampToValueAtTime(0.0001, t + dur);
    osc.connect(gain).connect(this.master);
    osc.start(t);
    osc.stop(t + dur + 0.05);
  }

  burst(dur, peak, fromHz, toHz) {
    if (!this.enabled || !this.ctx) return;
    const ctx = this.ctx, t = ctx.currentTime;
    const src = ctx.createBufferSource(), filter = ctx.createBiquadFilter(), gain = ctx.createGain();
    src.buffer = this.white;
    filter.type = "lowpass";
    filter.frequency.setValueAtTime(fromHz, t);
    filter.frequency.exponentialRampToValueAtTime(toHz, t + dur);
    gain.gain.setValueAtTime(peak, t);
    gain.gain.exponentialRampToValueAtTime(0.0001, t + dur);
    src.connect(filter).connect(gain).connect(this.master);
    src.start(t);
    src.stop(t + dur + 0.05);
  }

  beep(final) { this.tone(final ? 1320 : 880, final ? 0.4 : 0.12, 0.12); }
  thud() { this.tone(110, 0.35, 0.6, "sine", 38); this.burst(0.18, 0.25, 900, 120); }
  crash() { this.burst(1.3, 0.9, 4000, 80); this.tone(70, 0.9, 0.7, "sine", 28); }
}
