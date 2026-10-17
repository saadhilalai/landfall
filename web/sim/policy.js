/* policy.js — a pilot's neural network, running in the browser.

   Each pilot is the same small network that was trained in Python: the 8 numbers the game
   shows (times 4 for the Final AI, which also sees its last 3 moments) go through two layers
   of 64 "neurons" and come out as 4 scores, one per button: nothing, left thruster, main
   engine, right thruster. A trained pilot presses the button with the highest score. The
   untrained pilot picks at random, weighted by its (meaningless) scores. */

const decode = b64 => {
  const bytes = Uint8Array.from(atob(b64), c => c.charCodeAt(0));
  return new Float32Array(bytes.buffer);
};

export class Pilot {
  constructor(spec) {
    this.memory = spec.memory;
    this.deterministic = spec.deterministic;
    this.layers = spec.layers.map(l => ({ inputs: l.inputs, outputs: l.outputs,
                                          weight: decode(l.weight), bias: decode(l.bias) }));
    this.frames = [];
  }

  reset(obs) {                       // like Gymnasium's FrameStack: start with copies of the first view
    this.frames = Array.from({ length: this.memory }, () => obs);
  }

  observe(obs) {
    this.frames.push(obs);
    this.frames.shift();
  }

  scores() {
    let x = new Float32Array(this.memory * 8);
    this.frames.forEach((f, k) => x.set(f, k * 8));          // oldest moment first
    this.layers.forEach((layer, k) => {
      const out = new Float32Array(layer.outputs);
      for (let o = 0; o < layer.outputs; o++) {
        let sum = layer.bias[o];
        const row = o * layer.inputs;
        for (let i = 0; i < layer.inputs; i++) sum += layer.weight[row + i] * x[i];
        out[o] = k < this.layers.length - 1 ? Math.tanh(sum) : sum;
      }
      x = out;
    });
    return x;
  }

  act(random) {                      // random: a function giving numbers in [0, 1), for the untrained pilot
    const s = this.scores();
    if (this.deterministic) return s.indexOf(Math.max(...s));
    const top = Math.max(...s);
    const weights = Array.from(s, v => Math.exp(v - top));
    let r = random() * weights.reduce((a, b) => a + b, 0);
    for (let a = 0; a < weights.length; a++) if ((r -= weights[a]) < 0) return a;
    return weights.length - 1;
  }
}

export async function loadPilots(url = "data/pilots.json") {
  const data = await fetch(url).then(r => r.json());
  return Object.fromEntries(Object.entries(data.pilots).map(([id, spec]) => [id, new Pilot(spec)]));
}
