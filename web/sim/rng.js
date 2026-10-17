/* rng.js — NumPy's random-number generator (SeedSequence + PCG64), rewritten in JavaScript.

   The game draws its terrain, its first push, its wind pattern and its engine "wobble" from
   NumPy's generator. Rewriting the same generator here means a landing number makes exactly
   the same random draws in the browser as it does in Python: the same terrain, the same
   push, the same gusts. Checked against NumPy by check_browser_sim.py. */

const M32 = 0xffffffff;
const MASK64 = (1n << 64n) - 1n;
const MASK128 = (1n << 128n) - 1n;
const PCG_MULT = (2549297995355413924n << 64n) + 4865540595714422341n;

// ---- SeedSequence: turns a seed number into a well-mixed starting state -------------------
const INIT_A = 0x43b0d7e5, MULT_A = 0x931e8875, INIT_B = 0x8b51f9dd, MULT_B = 0x58f38ded;
const MIX_MULT_L = 0xca01f9dd, MIX_MULT_R = 0x4973f715, POOL_SIZE = 4;

function seedWords(seed) {            // a non-negative integer -> its 32-bit words, lowest first
  let big = BigInt(seed);
  const words = [];
  do { words.push(Number(big & 0xffffffffn)); big >>= 32n; } while (big > 0n);
  return words;
}

function seedSequenceState(seed, nWords32) {
  let hashConst = INIT_A;
  const hashmix = value => {
    value = (value ^ hashConst) >>> 0;
    hashConst = Math.imul(hashConst, MULT_A) >>> 0;
    value = Math.imul(value, hashConst) >>> 0;
    return (value ^ (value >>> 16)) >>> 0;
  };
  const mix = (x, y) => {
    let r = (Math.imul(MIX_MULT_L, x) - Math.imul(MIX_MULT_R, y)) >>> 0;
    return (r ^ (r >>> 16)) >>> 0;
  };
  const entropy = seedWords(seed);
  const pool = [];
  for (let i = 0; i < POOL_SIZE; i++) pool.push(hashmix(i < entropy.length ? entropy[i] : 0));
  for (let src = 0; src < POOL_SIZE; src++)
    for (let dst = 0; dst < POOL_SIZE; dst++)
      if (src !== dst) pool[dst] = mix(pool[dst], hashmix(pool[src]));
  for (let src = POOL_SIZE; src < entropy.length; src++)
    for (let dst = 0; dst < POOL_SIZE; dst++) pool[dst] = mix(pool[dst], hashmix(entropy[src]));

  const out = [];
  let h = INIT_B;
  for (let i = 0; i < nWords32; i++) {
    let v = (pool[i % POOL_SIZE] ^ h) >>> 0;
    h = Math.imul(h, MULT_B) >>> 0;
    v = Math.imul(v, h) >>> 0;
    out.push((v ^ (v >>> 16)) >>> 0);
  }
  return out;
}

// ---- PCG64 + the two NumPy draws the game uses: uniform() and integers() --------------------
export class Generator {
  constructor(seed) {
    const w = seedSequenceState(seed, 8).map(BigInt);
    const u64 = k => w[2 * k] | (w[2 * k + 1] << 32n);       // little-endian pairs of 32-bit words
    const initState = (u64(0) << 64n) | u64(1);
    const initSeq = (u64(2) << 64n) | u64(3);
    this.inc = ((initSeq << 1n) | 1n) & MASK128;
    this.state = 0n;
    this._step();
    this.state = (this.state + initState) & MASK128;
    this._step();
    this.spare32 = null;                                      // PCG64 hands out 32-bit numbers in pairs
  }

  _step() { this.state = (this.state * PCG_MULT + this.inc) & MASK128; }

  next64() {
    this._step();
    const s = this.state;
    const x = ((s >> 64n) ^ s) & MASK64;
    const rot = s >> 122n;
    return ((x >> rot) | (x << ((64n - rot) & 63n))) & MASK64;
  }

  next32() {
    if (this.spare32 !== null) { const v = this.spare32; this.spare32 = null; return v; }
    const v = this.next64();
    this.spare32 = Number(v >> 32n);
    return Number(v & 0xffffffffn);
  }

  random() { return Number(this.next64() >> 11n) * (1 / 9007199254740992); }   // [0, 1), 53 bits

  uniform(low, high) { return low + (high - low) * this.random(); }

  integers(low, high) {        // like rng.integers(low, high): low <= n < high (range below 2^32)
    const rng = high - low - 1;
    const excl = rng + 1;
    const draw = () => {
      const m = BigInt(this.next32()) * BigInt(excl);
      return [Number(m >> 32n), Number(m & 0xffffffffn)];
    };
    let [value, leftover] = draw();
    if (leftover < excl) {
      const threshold = (M32 - rng) % excl;
      while (leftover < threshold) [value, leftover] = draw();
    }
    return low + value;
  }
}
