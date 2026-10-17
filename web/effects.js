/* effects.js — the polish layer: starfield, glass shine, rolling digits, count-up numbers,
   scroll reveals and the automatic quality switch. None of it changes what the AI does. */

const root = document.documentElement;
export const reduceMotion = () => root.classList.contains("reduce-motion");
export const isLite = () => root.dataset.fx === "lite";

/* ---- Starfield: two tiled star layers drifting slowly upward, moved by the GPU ---------------- */
function starTile(size, count, rMin, rMax, aMin, aMax) {
  const c = document.createElement("canvas");
  c.width = c.height = size;
  const g = c.getContext("2d");
  g.fillStyle = "#fff";
  for (let i = 0; i < count; i++) {
    g.globalAlpha = aMin + Math.random() * (aMax - aMin);
    g.beginPath();
    g.arc(Math.random() * size, Math.random() * size, rMin + Math.random() * (rMax - rMin), 0, Math.PI * 2);
    g.fill();
  }
  return c.toDataURL("image/png");
}

export function startStarfield() {
  const layers = [
    [".stars.far", 512, isLite() ? 70 : 110, 0.35, 0.8, 0.15, 0.5, 260],
    [".stars.near", 768, 45, 0.6, 1.2, 0.25, 0.7, 150],
  ];
  for (const [sel, size, count, rMin, rMax, aMin, aMax, seconds] of layers) {
    const el = document.querySelector(sel);
    if (!el) continue;
    el.style.backgroundImage = `url(${starTile(size, count, rMin, rMax, aMin, aMax)})`;
    el.style.setProperty("--tile", `${size}px`);
    el.style.setProperty("--dur", `${seconds}s`);
  }
}

/* ---- Liquid glass: the shine and bright edge follow the pointer --------------------------------- */
export function trackGlassLight() {
  if (matchMedia("(pointer: coarse)").matches) return;   // phones: a fixed highlight is enough
  let x = 0, y = 0, queued = false;
  const update = () => {
    queued = false;
    if (isLite()) return;
    for (const g of document.querySelectorAll(".glass, .glass-flat")) {
      const r = g.getBoundingClientRect();
      if (r.bottom < 0 || r.top > innerHeight || !r.width) continue;
      g.style.setProperty("--mx", `${(((x - r.left) / r.width) * 100).toFixed(1)}%`);
      g.style.setProperty("--my", `${(((y - r.top) / r.height) * 100).toFixed(1)}%`);
    }
  };
  addEventListener("pointermove", e => {
    x = e.clientX; y = e.clientY;
    if (!queued) { queued = true; requestAnimationFrame(update); }
  }, { passive: true });
}

/* ---- Rolling digits: each digit is a strip 0-9 that slides to its value ----------------------- */
const STRIP = Array.from({ length: 10 }, (_, d) => `<span>${d}</span>`).join("");

export class Roller {
  constructor(el) { this.el = el; this.text = null; this.shape = null; this.strips = []; }

  set(text) {
    if (text === this.text) return;
    this.text = text;
    const shape = text.replace(/\d/g, "0");
    if (shape !== this.shape) {                       // a digit appeared or disappeared: rebuild
      this.shape = shape;
      this.el.innerHTML = [...text].map(ch => (/\d/.test(ch)
        ? `<span class="rd"><span class="rs">${STRIP}</span></span>`
        : `<span class="rc">${ch === " " ? "&#8199;" : ch}</span>`)).join("");
      this.strips = [...this.el.querySelectorAll(".rs")];
    }
    let k = 0;
    for (const ch of text) if (/\d/.test(ch)) this.strips[k++].style.transform = `translateY(${-Number(ch) * 10}%)`;
  }
}

/* ---- Numbers that count up ------------------------------------------------------------------------- */
export function countUp(el, to, { decimals = 0, prefix = "", suffix = "", duration = 1500 } = {}) {
  const fmt = v => `${prefix}${v.toFixed(decimals)}${suffix}`;
  if (reduceMotion()) { el.innerHTML = fmt(to); return; }
  const t0 = performance.now();
  const step = now => {
    const k = Math.min(1, (now - t0) / duration), eased = 1 - Math.pow(1 - k, 4);
    el.innerHTML = fmt(to * eased);
    if (k < 1) requestAnimationFrame(step);
  };
  el.innerHTML = fmt(0);
  requestAnimationFrame(step);
}

/* ---- Fade and slide sections in as they scroll into view ---------------------------------------- */
export function revealOnScroll(selector, onReveal = () => {}) {
  const els = [...document.querySelectorAll(selector)];
  if (!("IntersectionObserver" in window)) { els.forEach(onReveal); return; }
  for (const el of els) {
    el.classList.add("reveal");
    const siblings = [...el.parentElement.children].filter(c => c.matches(selector));
    if (siblings.length > 1) el.style.transitionDelay = `${siblings.indexOf(el) * 90}ms`;
  }
  const io = new IntersectionObserver(entries => {
    for (const e of entries) {
      if (!e.isIntersecting) continue;
      e.target.classList.add("in");
      onReveal(e.target);
      io.unobserve(e.target);
    }
  }, { rootMargin: "0px 0px -8% 0px", threshold: 0.12 });
  els.forEach(el => io.observe(el));
}

/* ---- Automatic quality: if frames get slow, switch to the lighter effects ----------------------- */
export class FrameMonitor {
  constructor(onSlow) { this.onSlow = onSlow; this.samples = []; this.prev = 0; this.stats = []; }

  frame(now) {
    if (this.prev) {
      const dt = now - this.prev;
      if (dt < 250) this.samples.push(dt);          // ignore pauses (tab hidden, debugger)
    }
    this.prev = now;
    if (this.samples.length >= 90) {
      const sorted = [...this.samples].sort((a, b) => a - b);
      const median = sorted[sorted.length >> 1], p90 = sorted[Math.floor(sorted.length * 0.9)];
      this.stats.push({ fps: Math.round(1000 / median), median: +median.toFixed(1), p90: +p90.toFixed(1), fx: root.dataset.fx });
      if (median > 22 && !isLite()) { root.dataset.fx = "lite"; this.onSlow(); }
      this.samples = [];
    }
  }

  pause() { this.prev = 0; }
}

/* ---- Glow sprite for engine light (drawn once, then stamped: much cheaper than canvas shadows) ---- */
export function glowSprite(size = 128) {
  const c = document.createElement("canvas");
  c.width = c.height = size;
  const g = c.getContext("2d"), r = size / 2;
  const grad = g.createRadialGradient(r, r, 0, r, r, r);
  grad.addColorStop(0, "rgba(255,255,255,1)");
  grad.addColorStop(0.18, "rgba(255,255,255,0.65)");
  grad.addColorStop(0.45, "rgba(255,255,255,0.18)");
  grad.addColorStop(1, "rgba(255,255,255,0)");
  g.fillStyle = grad;
  g.fillRect(0, 0, size, size);
  return c;
}
