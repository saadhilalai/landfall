/* LANDFALL — front end. An S4D project by Saad Hilal.
   The server flies each landing and sends back a recording (50 snapshots per simulated
   second). This file draws that recording live on a canvas with webcast telemetry, and runs
   mission control (sliders with physics zones, landing check), the countdown, the result
   screen and the sections below. Effects live in effects.js, sound in sound.js. */

import { Sound } from "./sound.js";
import { reduceMotion, isLite, startStarfield, trackGlassLight, Roller, countUp, revealOnScroll,
         FrameMonitor, glowSprite } from "./effects.js";

// Saad's profiles for the footer. Paste the full URLs here; empty ones stay hidden.
const LINKS = { GitHub: "", LinkedIn: "" };

const $ = (sel, root = document) => root.querySelector(sel);
const el = (tag, attrs = {}, html = "") => {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
  if (html) node.innerHTML = html;
  return node;
};
const SVGNS = "http://www.w3.org/2000/svg";
const svg = (tag, attrs = {}) => {
  const node = document.createElementNS(SVGNS, tag);
  for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
  return node;
};
const wait = ms => new Promise(r => setTimeout(r, ms));
const MINUS = "−";
const signedSlot = (v, d = 1) => (v < -0.05 ? MINUS : " ") + Math.abs(v).toFixed(d);   // fixed sign slot
const pct = r => { const v = r * 100; return (v >= 99.5 && v < 100 ? v.toFixed(1) : Math.round(v)) + "%"; };

// ---- Loading screen: "AN S4D PROJECT", then the mission name ---------------------------------------
const loader = $("#loader");
const intro = (async () => {
  await wait(80);
  loader.classList.add("s1");
  await wait(reduceMotion() ? 150 : 750);
  loader.classList.add("s2");
  await wait(reduceMotion() ? 450 : 1450);
})();

startStarfield();
trackGlassLight();
const sound = new Sound();
const SITE = await fetch("data/site.json").then(r => r.json());
const P = SITE.physics;
const W = P.world.w, H = P.world.h, FPS = 50;
const PILOT = Object.fromEntries(SITE.pilots.map(p => [p.id, p]));
const GLOW = glowSprite();

const state = {
  pilot: SITE.first_flight.pilot,
  compare: false,
  pilot2: "baseline",
  sliders: { ...SITE.first_flight.sliders },
  landing: SITE.first_flight.landing,
  landingEdited: false,
  note: "",
};

// A link can carry a flight: ?pilot=final&vs=baseline&gravity=10&wind=16.25&turbulence=1.625&engine=100&landing=10000
const query = new URLSearchParams(location.search);
if (PILOT[query.get("pilot")]) state.pilot = query.get("pilot");
if (PILOT[query.get("vs")]) { state.compare = true; state.pilot2 = query.get("vs"); }
for (const [key, s] of Object.entries(SITE.sliders)) {
  const v = parseFloat(query.get(key));
  if (Number.isFinite(v)) state.sliders[key] = Math.min(s.max, Math.max(s.min, v));
}
if (/^\d{1,6}$/.test(query.get("landing") || "")) state.landing = parseInt(query.get("landing"), 10);
function shareURL() {
  const q = new URLSearchParams(location.search);
  for (const k of ["pilot", "vs", "landing", ...Object.keys(SITE.sliders)]) q.delete(k);
  q.set("pilot", state.pilot);
  if (state.compare) q.set("vs", state.pilot2);
  for (const [k, v] of Object.entries(state.sliders)) q.set(k, String(+v.toFixed(3)));
  q.set("landing", state.landing);
  history.replaceState(null, "", `?${q}${location.hash}`);
}

// ---- Sound toggle (off by default) -------------------------------------------------------------------
const soundBtn = $("#sound");
soundBtn.addEventListener("click", () => {
  const on = !sound.enabled;
  soundBtn.setAttribute("aria-pressed", String(on));
  $(".label", soundBtn).textContent = on ? "Sound on" : "Sound off";
  sound.setEnabled(on);
});

// =====================================================================================
// Physics: what the engines can do against gravity, wind and turbulence
// (all constants measured in the simulator by export_site_data.py)
// =====================================================================================
function physics(s) {
  const e = s.engine / 100, g = s.gravity;
  const thrust = P.main_thrust_accel * e;
  const sideways = Math.sqrt(Math.max(thrust * thrust - g * g, 0)) + P.side_thrust_accel * e;
  return {
    twr: thrust / g,
    gravityMax: thrust,
    engineMin: (g / P.main_thrust_accel) * 100,
    windMax: (sideways * P.total_mass) / P.gust_peak,
    turbulenceMax: (P.side_spin_accel * e) / (P.gust_peak * P.spin_per_torque),
  };
}

function impossibleFrom(key, ph) {
  const s = SITE.sliders[key];
  const range = { gravity: [ph.gravityMax, s.max], engine: [s.min, ph.engineMin],
                  wind: [ph.windMax, s.max], turbulence: [ph.turbulenceMax, s.max] }[key];
  const from = Math.max(range[0], s.min), to = Math.min(range[1], s.max);
  return from < to ? [from, to] : null;
}

function zoneOf(key, value, ph) {
  const red = impossibleFrom(key, ph);
  if (red && value >= red[0] - 1e-9 && value <= red[1] + 1e-9) return "impossible";
  const [lo, hi] = SITE.sliders[key].trained;
  return value >= lo - 1e-9 && value <= hi + 1e-9 ? "trained" : "beyond";
}
const ZONE_TEXT = { trained: "Trained", beyond: "Beyond training", impossible: "Physically impossible" };

// =====================================================================================
// Mission control
// =====================================================================================
const pilotsEl = $("#pilots"), aboutEl = $("#pilot-about"), compareEl = $("#compare"), pilot2El = $("#pilot2");
const landingEl = $("#landing"), launchBtn = $("#launch");

for (const p of SITE.pilots) {
  const b = el("button", { type: "button", class: "pilot glass-flat", role: "radio", "data-id": p.id }, p.name);
  b.addEventListener("click", () => { state.pilot = p.id; renderPilots(); updateCheck(); });
  pilotsEl.append(b);
  pilot2El.append(el("option", { value: p.id }, p.name));
}
pilot2El.value = state.pilot2;
compareEl.checked = state.compare;
pilot2El.disabled = !state.compare;
function renderPilots() {
  for (const b of pilotsEl.children) b.setAttribute("aria-checked", String(b.dataset.id === state.pilot));
  aboutEl.textContent = PILOT[state.pilot].about;
}
compareEl.addEventListener("change", () => { state.compare = compareEl.checked; pilot2El.disabled = !state.compare; updateCheck(); });
pilot2El.addEventListener("change", () => { state.pilot2 = pilot2El.value; updateCheck(); });
landingEl.addEventListener("input", () => {
  state.landing = Math.max(0, Math.min(999999, parseInt(landingEl.value || "0", 10)));
  state.landingEdited = true;
});

const sliderEls = {};
for (const [key, s] of Object.entries(SITE.sliders)) {
  const id = `s-${key}`;
  const ctl = el("div", { class: "ctl" });
  ctl.innerHTML = `
    <div class="ctl-head"><label for="${id}">${s.label}</label><output for="${id}"></output></div>
    <div class="ctl-track"><div class="zones" aria-hidden="true"></div>
      <input type="range" id="${id}" min="${s.min}" max="${s.max}" step="${s.step}"></div>
    <div class="ctl-zone"><span class="zone-glyph" aria-hidden="true"></span><span class="zone-name"></span></div>`;
  const input = $("input", ctl);
  input.value = state.sliders[key];
  input.addEventListener("input", () => { state.sliders[key] = parseFloat(input.value); updateCheck(); });
  $("#sliders").append(ctl);
  sliderEls[key] = { input, out: $("output", ctl), zones: $(".zones", ctl), glyph: $(".zone-glyph", ctl), name: $(".zone-name", ctl) };
}

const fmtSlider = (key, v) => {
  const s = SITE.sliders[key];
  const text = key === "turbulence" ? v.toFixed(2) : key === "engine" ? (v % 1 ? v.toFixed(2) : v.toFixed(0)) : v.toFixed(1);
  return `${text}${s.unit ? `<small>${s.unit}</small>` : ""}`;
};

function updateSliders(ph) {
  for (const [key, s] of Object.entries(SITE.sliders)) {
    const { input, out, zones, glyph, name } = sliderEls[key];
    const v = state.sliders[key];
    input.value = v;
    out.innerHTML = fmtSlider(key, v);
    // the zone line: trained (green, thick), beyond training (amber, thin), impossible (red, dotted)
    const at = x => ((x - s.min) / (s.max - s.min)) * 100;
    const red = impossibleFrom(key, ph);
    const cuts = [s.min, s.max, ...s.trained, ...(red || [])].filter(x => x >= s.min && x <= s.max).sort((a, b) => a - b);
    const parts = [];
    for (let i = 0; i < cuts.length - 1; i++) {
      const a = cuts[i], b = cuts[i + 1];
      if (b - a < 1e-9) continue;
      parts.push(`<span class="z-${zoneOf(key, (a + b) / 2, ph)}" style="left:${at(a)}%;width:${at(b) - at(a)}%"></span>`);
    }
    zones.innerHTML = parts.join("");
    const zone = zoneOf(key, v, ph);
    glyph.className = `zone-glyph zone-${zone}`;
    name.textContent = ZONE_TEXT[zone];
    input.setAttribute("aria-valuetext", `${v} ${s.unit}, ${ZONE_TEXT[zone]}`);
  }
}

function nearestTest(s) {
  let best = null;
  for (const t of SITE.tests) {
    const d = Math.hypot((s.gravity - t.sliders.gravity) / 3, (s.wind - t.sliders.wind) / 15,
                         (s.turbulence - t.sliders.turbulence) / 1.5, (s.engine - t.sliders.engine) / 30);
    if (!best || d < best.d) best = { t, d };
  }
  return best;
}

function updateCheck() {
  const s = state.sliders, ph = physics(s);
  updateSliders(ph);
  $("#twr").textContent = ph.twr.toFixed(2);
  let verdict, zone;
  if (ph.twr < 1) { zone = "impossible"; verdict = "<b>Impossible</b> The engine is weaker than gravity, so the lander can only fall."; }
  else if (s.wind > ph.windMax) { zone = "impossible"; verdict = "<b>Impossible</b> The strongest gusts push harder than the engines can push back."; }
  else if (s.turbulence > ph.turbulenceMax) { zone = "impossible"; verdict = "<b>Impossible</b> Turbulence twists harder than the side thrusters can correct."; }
  else if (Object.keys(SITE.sliders).some(k => zoneOf(k, s[k], ph) === "beyond")) { zone = "beyond"; verdict = "<b>Possible</b> Beyond training: the AI never practised this."; }
  else { zone = "trained"; verdict = "<b>Possible</b> Inside the conditions the AI practised."; }
  $("#verdict").innerHTML = `<span class="zone-glyph zone-${zone}" aria-hidden="true"></span><span>${verdict}</span>`;

  const near = nearestTest(s);
  const pilots = state.compare ? [state.pilot, state.pilot2] : [state.pilot];
  $("#expected").innerHTML = pilots.map(id => {
    const name = PILOT[id].name;
    if (zone === "impossible") {
      return `<div class="expect"><span class="expect-rate">0%</span><span class="expect-who">${name}</span>
        <span class="expect-note">Physics rules it out, whoever flies.</span></div>`;
    }
    if (!PILOT[id].model) {
      return `<div class="expect"><span class="expect-rate untested">n/a</span><span class="expect-who">${name}</span>
        <span class="expect-note">Never tested: it hasn't learned anything yet.</span></div>`;
    }
    const where = near.d < 0.15 ? `Measured: the Step 6 "${near.t.name}" test, ${near.t.landings} unseen landings.`
      : near.d < 0.5 ? `Closest Step 6 test: "${near.t.name}" (${near.t.landings} landings). Your settings are near it, not identical.`
      : `Untested here. The closest Step 6 test is "${near.t.name}", so treat this as a rough guide.`;
    return `<div class="expect"><span class="expect-rate${near.d >= 0.5 ? " untested" : ""}">${pct(near.t.landed[id])}</span>
      <span class="expect-who">${name}, expected landing rate</span><span class="expect-note">${where}</span></div>`;
  }).join("");
}

// =====================================================================================
// The stage: one pane per pilot, each a canvas + glass telemetry + result card
// =====================================================================================
const stage = $("#stage"), actions = $("#stage-actions"), consoleEl = $("#console");
const desktop = matchMedia("(min-width: 1081px)");
const GAUGE_R = 40, GAUGE_LEN = 2 * Math.PI * GAUGE_R * 0.75;   // 270-degree gauges

const gaugeHTML = (key, label, unit) => `
  <div class="gauge" data-g="${key}">
    <svg viewBox="0 0 92 92" aria-hidden="true">
      <circle class="gauge-track" cx="46" cy="46" r="${GAUGE_R}" stroke-dasharray="${GAUGE_LEN} 999" transform="rotate(135 46 46)"/>
      <circle class="gauge-arc" cx="46" cy="46" r="${GAUGE_R}" stroke-dasharray="${GAUGE_LEN} 999" stroke-dashoffset="${GAUGE_LEN}" transform="rotate(135 46 46)"/>
    </svg>
    <div><b class="roll" data-r="${key}"></b><span class="unit">${label}<br>${unit}</span></div>
  </div>`;

class Pane {
  constructor() {
    this.el = el("div", { class: "pane" });
    this.el.innerHTML = `
      <canvas aria-hidden="true"></canvas>
      <div class="flash"></div>
      <div class="pane-head"><span><span class="pane-pilot"></span><span class="pane-live">Live simulation</span></span><span class="pane-meta"></span></div>
      <div class="telemetry glass" aria-hidden="true">
        <div class="gauges">${gaugeHTML("speed", "Speed", "m/s")}${gaugeHTML("alt", "Altitude", "m")}</div>
        <dl class="readouts left">
          <div class="compact-only"><dt>Altitude</dt><dd><span class="roll" data-r="alt2"></span><small>m</small></dd></div>
          <div><dt>Vertical speed</dt><dd><span class="roll" data-r="vs"></span><small>m/s</small></dd></div>
          <div><dt>Horizontal speed</dt><dd><span class="roll" data-r="hs"></span><small>m/s</small></dd></div>
        </dl>
        <div class="clock"><span class="tplus">T+ 00:00.00</span><span class="phase">Ready</span></div>
        <dl class="readouts right">
          <div><dt>Tilt</dt><dd><span class="roll" data-r="tilt"></span><small>deg</small></dd></div>
          <div><dt>Fuel used</dt><dd><span class="roll" data-r="fuel"></span></dd></div>
          <div><dt>Wind</dt><dd><svg class="wind-arrow" viewBox="0 0 26 12"><path d="M1 6h20M16 1l6 5-6 5" fill="none" stroke="#fff" stroke-width="1.5"/></svg><span class="roll" data-r="wind"></span></dd></div>
        </dl>
      </div>
      <div class="outcome" hidden role="status">
        <div class="outcome-card glass">
          <h2 class="outcome-title"></h2><p class="outcome-sub"></p>
          <dl class="outcome-stats">
            <div><dt>Flight time</dt><dd data-o="time"></dd></div>
            <div><dt>Fuel used</dt><dd data-o="fuel"></dd></div>
            <div><dt>Landing speed</dt><dd data-o="speed"></dd></div>
          </dl>
        </div>
      </div>`;
    this.canvas = $("canvas", this.el);
    this.ctx = this.canvas.getContext("2d");
    this.clockEl = $(".tplus", this.el);
    this.phaseEl = $(".phase", this.el);
    this.windArrow = $(".wind-arrow", this.el);
    this.flashEl = $(".flash", this.el);
    this.rollers = {};
    for (const node of this.el.querySelectorAll("[data-r]")) this.rollers[node.dataset.r] = new Roller(node);
    this.arcs = {};
    for (const g of this.el.querySelectorAll("[data-g]")) this.arcs[g.dataset.g] = $(".gauge-arc", g);
    this.outcomeEl = $(".outcome", this.el);
    this.particles = [];
    this.streaks = [];
    this.flight = null;
  }

  load(flight, meta) {
    this.flight = flight;
    this.f = flight.frames;
    this.n = this.f.x.length;
    this.particles = [];
    this.lastStep = 0;
    this.lastTick = -1;
    this.touched = false;
    this.done = false;
    this.ended = false;
    this.playing = false;
    this.shake = null;
    this.outcomeEl.hidden = true;
    this.outcomeEl.classList.remove("show");
    this.el.classList.remove("ended");
    $(".pane-pilot", this.el).textContent = PILOT[flight.pilot].name;
    $(".pane-meta", this.el).textContent = meta;
    // terrain lookup for exhaust hitting the ground
    this.ter = flight.terrain;
    this.layout();
    this.setTelemetry(0, 0, true);
    this.setClock(-3);
    this.phaseEl.textContent = "Countdown";
    this.draw(0, 0, 0, performance.now());
  }

  layout() {
    const r = this.canvas.getBoundingClientRect();
    this.dpr = Math.min(window.devicePixelRatio || 1, isLite() ? 1.5 : 2);
    this.w = r.width; this.h = r.height;
    this.canvas.width = Math.round(r.width * this.dpr);
    this.canvas.height = Math.round(r.height * this.dpr);
    // Keep the world clear of the floating mission-control panel (desktop, last pane only).
    const c = consoleEl.getBoundingClientRect();
    const safeRight = desktop.matches && this === panes.at(-1) ? Math.max(0, r.right - c.left + 16) : 0;
    const usable = r.width - safeRight, compact = usable < 760;
    this.el.style.setProperty("--safe-right", `${safeRight}px`);
    this.el.style.setProperty("--usable", `${usable}px`);
    this.el.classList.toggle("compact", compact);
    const tele = $(".telemetry", this.el).getBoundingClientRect().height;
    stage.style.setProperty("--tele-h", `${Math.round(tele)}px`);
    const top = compact ? 92 : 124, bottom = tele + (compact ? 22 : 34);
    const floor = Math.max(0, Math.min(...this.flight.terrain.map(p => p[1])) - 0.5);
    this.s = Math.max(4, Math.min((usable - 24) / W, (r.height - top - bottom) / (H - floor)));
    this.ox = (usable - W * this.s) / 2;
    this.base = r.height - bottom + floor * this.s;
    this.skyTop = top - 40;
    const padY = this.Y(this.flight.pad[2]);
    this.streaks = Array.from({ length: Math.round(usable / (isLite() ? 26 : 16)) }, () => ({
      x: Math.random() * r.width, y: this.skyTop + Math.random() * Math.max(40, padY - 24 - this.skyTop),
      len: 18 + Math.random() * 46, k: 0.6 + Math.random() * 0.8,
    }));
  }

  X(x) { return this.ox + x * this.s; }
  Y(y) { return this.base - y * this.s; }

  groundAt(x) {
    const t = this.ter;
    if (x <= t[0][0]) return t[0][1];
    for (let k = 1; k < t.length; k++) {
      if (x <= t[k][0]) { const [x0, y0] = t[k - 1], [x1, y1] = t[k]; return y0 + ((y1 - y0) * (x - x0)) / (x1 - x0); }
    }
    return t[t.length - 1][1];
  }

  start(t0) { this.t0 = t0; this.prev = t0; this.playing = true; }

  setClock(sec) {
    const neg = sec < 0, a = Math.abs(sec);
    this.clockEl.textContent = `T${neg ? MINUS : "+"} 00:${String(Math.floor(a)).padStart(2, "0")}.${String(Math.floor((a % 1) * 100)).padStart(2, "0")}`;
  }

  // Exhaust, dust and touchdown for every simulated step we pass. Particles only appear on the
  // steps where the simulation actually fired that engine.
  simulateSteps(upTo, now) {
    const f = this.f, perStep = isLite() ? 2 : 4;
    for (let k = this.lastStep + 1; k <= upTo; k++) {
      const a = f.a[k], x = f.x[k], y = f.y[k], act = f.act[k];
      if (act === 2) {
        const dx = Math.sin(a), dy = -Math.cos(a);              // down along the body
        for (let i = 0; i < perStep; i++) {
          const sp = 6 + Math.random() * 5, spread = (Math.random() - 0.5) * 2.6;
          this.particles.push({ x: x + dx * 0.32, y: y + dy * 0.32,
            vx: f.vx[k] + dx * sp - dy * spread, vy: f.vy[k] + dy * sp + dx * spread,
            life: 0, max: 0.25 + Math.random() * 0.3, r: 0.05 + Math.random() * 0.08, hot: true });
        }
      } else if (act === 1 || act === 3) {
        const d = act - 2, ox = -d * Math.cos(a), oy = -d * Math.sin(a);
        const px = x + ox * 0.4 - Math.sin(a) * 0.567, py = y + oy * 0.4 + Math.cos(a) * 0.467;
        for (let i = 0; i < perStep / 2; i++) {
          this.particles.push({ x: px, y: py, vx: f.vx[k] + ox * (3 + Math.random() * 1.5), vy: f.vy[k] + oy * 3.5 + (Math.random() - 0.5),
            life: 0, max: 0.14 + Math.random() * 0.1, r: 0.035 + Math.random() * 0.03, hot: true });
        }
      }
      if (!this.touched && f.legs[k] > 0) {
        this.touched = true;
        this.onTouchdown(k, now);
      }
    }
    this.lastStep = Math.max(this.lastStep, upTo);
  }

  onTouchdown(k, now) {
    const f = this.f;
    sound.thud();
    this.shake = { t0: now, dur: 0.35, amp: 3.5 };
    for (const [lx, ly] of [[f.l0x[k], f.l0y[k]], [f.l1x[k], f.l1y[k]]]) {
      for (let i = 0; i < (isLite() ? 6 : 12); i++) {
        const dir = Math.random() < 0.5 ? -1 : 1;
        this.particles.push({ x: lx, y: ly - 0.2, vx: dir * (0.8 + Math.random() * 2.6), vy: Math.random() * 0.6,
          life: 0, max: 0.5 + Math.random() * 0.5, r: 0.03 + Math.random() * 0.04, dust: true });
      }
    }
  }

  onCrash(now) {
    const f = this.f, k = this.n - 1;
    sound.crash();
    this.shake = { t0: now, dur: 0.9, amp: 14 };
    if (!reduceMotion()) { this.flashEl.classList.remove("on"); void this.flashEl.offsetWidth; this.flashEl.classList.add("on"); }
    for (let i = 0; i < (isLite() ? 30 : 60); i++) {
      const ang = Math.random() * Math.PI, sp = 2 + Math.random() * 8;
      this.particles.push({ x: f.x[k], y: f.y[k], vx: Math.cos(ang) * sp * (Math.random() < 0.5 ? -1 : 1), vy: Math.sin(ang) * sp,
        life: 0, max: 0.7 + Math.random() * 1, r: 0.03 + Math.random() * 0.05, spark: true });
    }
  }

  frame(now) {
    const dt = Math.min(0.05, (now - this.prev) / 1000);
    this.prev = now;
    const t = (now - this.t0) / 1000 - 0.35;                  // a beat of stillness after LAUNCH
    const pos = Math.max(0, Math.min(t * FPS, this.n - 1));
    const i = Math.floor(pos), frac = pos - i;
    this.simulateSteps(i, now);
    if (pos >= this.n - 1 && !this.ended) {
      this.ended = true;
      if (this.flight.outcome === "crashed") this.onCrash(now);
    }
    this.setTelemetry(i, frac);
    this.setClock(Math.max(0, t));
    this.draw(i, frac, dt, now);
    if (t > (this.n - 1) / FPS + 0.9 && !this.done) {
      this.done = true;
      this.showOutcome();
    }
    return this.done && this.particles.length === 0;
  }

  lerp(key, i, frac) {
    const v = this.f[key];
    return i + 1 < this.n ? v[i] + (v[i + 1] - v[i]) * frac : v[i];
  }

  currentAct() {
    return this.playing && !this.ended ? this.f.act[Math.min(this.n - 1, this.lastStep)] : 0;
  }

  // Gauges move every frame; the numbers tick 12.5 times a second, like a flight computer.
  setTelemetry(i, frac, force = false) {
    const f = this.f;
    const vx = this.lerp("vx", i, frac), vy = this.lerp("vy", i, frac);
    const alt = Math.max(0, this.lerp("y", i, frac) - this.flight.pad[2] - P.rest_height);
    const speed = Math.hypot(vx, vy);
    this.arcs.speed.setAttribute("stroke-dashoffset", String(GAUGE_LEN * (1 - Math.min(speed / 12, 1))));
    this.arcs.alt.setAttribute("stroke-dashoffset", String(GAUGE_LEN * (1 - Math.min(alt / 11, 1))));
    const act = f.act[i];
    this.phaseEl.textContent = i >= this.n - 1 ? "Signal end" : f.legs[i] > 0 ? "Ground contact" :
      act === 2 ? "Main engine" : act ? "Steering thrusters" : "Coasting";
    const tick = Math.floor(i / 4);
    if (tick === this.lastTick && !force) return;
    this.lastTick = tick;
    let deg = (this.lerp("a", i, frac) * 180) / Math.PI;
    deg = ((deg + 180) % 360 + 360) % 360 - 180;
    const w = f.wind[i], R = this.rollers;
    R.speed.set(speed.toFixed(1));
    R.alt.set(alt.toFixed(1));
    R.alt2.set(alt.toFixed(1));
    R.vs.set(signedSlot(vy));
    R.hs.set(signedSlot(vx));
    R.tilt.set(signedSlot(deg, 0));
    R.fuel.set(String(Math.round(f.fuel[i])));
    R.wind.set(Math.abs(w).toFixed(1));
    this.windArrow.style.transform = `scaleX(${w < 0 ? -1 : 1})`;
    this.windArrow.style.opacity = Math.abs(w) < 0.05 ? "0.25" : "1";
  }

  draw(i, frac, dt, now) {
    const ctx = this.ctx, f = this.f, s = this.s, dpr = this.dpr;
    // camera shake (touchdown: small; crash: big; none with reduced motion)
    let sx = 0, sy = 0;
    if (this.shake && !reduceMotion()) {
      const age = (now - this.shake.t0) / 1000;
      if (age < this.shake.dur) {
        const k = this.shake.amp * Math.pow(1 - age / this.shake.dur, 2);
        sx = (Math.random() * 2 - 1) * k; sy = (Math.random() * 2 - 1) * k;
      }
    }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, this.w, this.h);                     // the starfield shows through the sky
    ctx.setTransform(dpr, 0, 0, dpr, sx * dpr, sy * dpr);

    // Wind, made visible: faint streaks moving with the gust pushing on the lander right now.
    const wind = f.wind[i];
    if (Math.abs(wind) > 0.05) {
      ctx.strokeStyle = `rgba(255,255,255,${Math.min(0.3, 0.05 + (Math.abs(wind) / 25) * 0.28)})`;
      ctx.lineWidth = 1;
      ctx.beginPath();
      for (const st of this.streaks) {
        if (!reduceMotion()) st.x += wind * 26 * st.k * dt;
        if (st.x > this.w + 60) st.x -= this.w + 120;
        if (st.x < -60) st.x += this.w + 120;
        ctx.moveTo(st.x, st.y);
        ctx.lineTo(st.x - Math.sign(wind) * st.len * Math.min(1, Math.abs(wind) / 10 + 0.3), st.y);
      }
      ctx.stroke();
    }

    // Ground: black below a thin white line (hides the stars), continued flat to the edges.
    const ter = this.flight.terrain;
    ctx.beginPath();
    ctx.moveTo(-40, this.Y(ter[0][1]));
    for (const [x, y] of ter) ctx.lineTo(this.X(x), this.Y(y));
    ctx.lineTo(this.w + 40, this.Y(ter[ter.length - 1][1]));
    ctx.lineTo(this.w + 40, this.h + 40);
    ctx.lineTo(-40, this.h + 40);
    ctx.closePath();
    ctx.fillStyle = "#000";
    ctx.fill();
    ctx.beginPath();
    ctx.moveTo(-40, this.Y(ter[0][1]));
    for (const [x, y] of ter) ctx.lineTo(this.X(x), this.Y(y));
    ctx.lineTo(this.w + 40, this.Y(ter[ter.length - 1][1]));
    ctx.strokeStyle = "#fff";
    ctx.lineWidth = 1.25;
    ctx.stroke();

    // Landing pad and flags
    const [x1, x2, py] = this.flight.pad;
    ctx.lineWidth = 3;
    ctx.beginPath(); ctx.moveTo(this.X(x1), this.Y(py)); ctx.lineTo(this.X(x2), this.Y(py)); ctx.stroke();
    ctx.lineWidth = 1;
    ctx.fillStyle = "#fff";
    for (const fx of [x1, x2]) {
      const top = py + 50 / 30;
      ctx.beginPath(); ctx.moveTo(this.X(fx), this.Y(py)); ctx.lineTo(this.X(fx), this.Y(top)); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(this.X(fx), this.Y(top)); ctx.lineTo(this.X(fx), this.Y(top - 10 / 30));
      ctx.lineTo(this.X(fx + 25 / 30), this.Y(top - 5 / 30)); ctx.closePath(); ctx.fill();
    }

    // Flight path so far
    ctx.fillStyle = "rgba(255,255,255,0.3)";
    for (let k = 0; k <= i; k += 4) ctx.fillRect(this.X(f.x[k]) - 0.75, this.Y(f.y[k]) - 0.75, 1.5, 1.5);

    const x = this.lerp("x", i, frac), y = this.lerp("y", i, frac), a = this.lerp("a", i, frac);
    const firing = f.act[i] === 2 && i < this.n - 1 && this.playing;
    const flicker = 0.85 + Math.random() * 0.3;

    // Engine light on the ground when firing close to it
    ctx.globalCompositeOperation = "lighter";
    if (firing) {
      const alt = y - this.groundAt(x);
      if (alt < 4.5) {
        const gy = this.Y(this.groundAt(x)), k = (1 - alt / 4.5) * 0.55 * flicker, gw = 3.4 * s;
        ctx.globalAlpha = k;
        ctx.drawImage(GLOW, this.X(x) - gw / 2, gy - gw * 0.12, gw, gw * 0.24);
        ctx.globalAlpha = 1;
      }
    }

    // Exhaust (white-hot, additive), dust and sparks
    const alive = [];
    for (const p of this.particles) {
      p.life += dt;
      if (p.life >= p.max) continue;
      if (p.spark) p.vy -= 9 * dt;
      const drag = p.dust ? 2.5 : 1.5;
      p.vx -= p.vx * drag * dt; p.vy -= p.vy * drag * dt;
      p.x += p.vx * dt; p.y += p.vy * dt;
      if (p.hot) {                       // exhaust hitting the ground spreads sideways as dust
        const g = this.groundAt(p.x);
        if (p.y < g + 0.04) {
          p.y = g + 0.04; p.hot = false; p.dust = true;
          p.vx = (p.vx >= 0 ? 1 : -1) * (1.5 + Math.random() * 3); p.vy = Math.random() * 0.5; p.max = p.life + 0.5 + Math.random() * 0.4;
        }
      }
      const k = 1 - p.life / p.max;
      const rad = Math.max(0.6, p.r * s * (p.spark ? 1 : 0.6 + k * 0.7));
      if (p.dust) {
        ctx.globalCompositeOperation = "source-over";
        ctx.fillStyle = `rgba(170,170,170,${0.6 * k})`;
      } else {
        ctx.globalCompositeOperation = "lighter";
        ctx.fillStyle = `rgba(255,255,255,${(p.spark ? 0.9 : 0.55) * k})`;
      }
      ctx.beginPath(); ctx.arc(this.X(p.x), this.Y(p.y), rad, 0, 7); ctx.fill();
      alive.push(p);
    }
    this.particles = alive;

    // Main engine: a glow plus a flickering flame, only on steps where the simulation fired it
    if (firing) {
      const dx = Math.sin(a), dy = -Math.cos(a), nx = x + dx * 0.33, ny = y + dy * 0.33;
      const gsz = 1.5 * s * flicker;
      ctx.globalCompositeOperation = "lighter";
      ctx.globalAlpha = 0.9;
      ctx.drawImage(GLOW, this.X(nx + dx * 0.25) - gsz / 2, this.Y(ny + dy * 0.25) - gsz / 2, gsz, gsz);
      ctx.globalAlpha = 1;
      const len = (0.6 + Math.random() * 0.35), half = 0.13;
      ctx.fillStyle = "rgba(255,255,255,0.95)";
      ctx.beginPath();
      ctx.moveTo(this.X(nx - dy * half), this.Y(ny + dx * half));
      ctx.lineTo(this.X(nx + dy * half), this.Y(ny - dx * half));
      ctx.lineTo(this.X(nx + dx * len), this.Y(ny + dy * len));
      ctx.closePath(); ctx.fill();
    } else if ((f.act[i] === 1 || f.act[i] === 3) && this.playing && i < this.n - 1) {
      const d = f.act[i] - 2, ox = -d * Math.cos(a), oy = -d * Math.sin(a);
      const px = x + ox * 0.45 - Math.sin(a) * 0.567, py2 = y + oy * 0.45 + Math.cos(a) * 0.467, gsz = 0.55 * s * flicker;
      ctx.globalCompositeOperation = "lighter";
      ctx.globalAlpha = 0.7;
      ctx.drawImage(GLOW, this.X(px) - gsz / 2, this.Y(py2) - gsz / 2, gsz, gsz);
      ctx.globalAlpha = 1;
    }
    ctx.globalCompositeOperation = "source-over";

    // The lander: legs, then body
    const lost = this.ended && this.flight.outcome === "crashed";
    ctx.fillStyle = lost ? "#6f6f6f" : "#fff";
    for (const leg of ["l0", "l1"]) {
      this.poly(P.leg_shape, this.lerp(leg + "x", i, frac), this.lerp(leg + "y", i, frac), this.lerp(leg + "a", i, frac));
    }
    this.poly(P.lander_shape, x, y, a);
  }

  poly(shape, x, y, a) {
    const ctx = this.ctx, c = Math.cos(a), sn = Math.sin(a);
    ctx.beginPath();
    shape.forEach(([vx, vy], k) => {
      const px = this.X(x + vx * c - vy * sn), py = this.Y(y + vx * sn + vy * c);
      k ? ctx.lineTo(px, py) : ctx.moveTo(px, py);
    });
    ctx.closePath();
    ctx.fill();
  }

  showOutcome() {
    const fl = this.flight;
    const text = {
      landed: ["Touchdown confirmed", "At rest on the pad."],
      crashed: ["Vehicle lost", fl.contact_speed != null ? `Hull hit the ground at ${fl.contact_speed.toFixed(1)} m/s.` : "Hull hit the ground."],
      flew_away: ["Vehicle lost", "Drifted out of range."],
      landed_off_pad: ["Touchdown off target", "Landed safely, but outside the pad."],
      never_settled: ["No touchdown", "Still airborne when the 20-second window closed."],
    }[fl.outcome];
    $(".outcome-title", this.el).textContent = text[0];
    $(".outcome-sub", this.el).textContent = `${PILOT[fl.pilot].name}. ${text[1]}`;
    countUp($('[data-o="time"]', this.el), fl.seconds, { decimals: 1, suffix: "<small>s</small>", duration: 900 });
    countUp($('[data-o="fuel"]', this.el), fl.fuel, { decimals: 0, duration: 900 });
    const sp = $('[data-o="speed"]', this.el);
    if (fl.contact_speed == null) sp.textContent = "n/a";
    else countUp(sp, fl.contact_speed, { decimals: 1, suffix: "<small>m/s</small>", duration: 900 });
    this.outcomeEl.hidden = false;
    this.el.classList.add("ended");
    requestAnimationFrame(() => requestAnimationFrame(() => this.outcomeEl.classList.add("show")));
  }
}

let panes = [];
function setPaneCount(n) {
  while (panes.length < n) { const p = new Pane(); stage.insertBefore(p.el, $("#countdown")); panes.push(p); }
  while (panes.length > n) panes.pop().el.remove();
  stage.dataset.panes = String(n);
}

const monitor = new FrameMonitor(() => { for (const p of panes) if (p.flight) p.layout(); });
window.__landfall = { monitor, state, sound, get panes() { return panes; } };   // for testing

let raf = null;
function tick(now) {
  monitor.frame(now);
  let finished = true;
  for (const p of panes) if (p.playing && !p.frame(now)) finished = false;
  const acts = panes.map(p => p.currentAct());
  sound.engine(acts.includes(2), acts.includes(1) || acts.includes(3));
  if (panes.every(p => p.done) && actions.hidden) showActions();
  if (finished) { raf = null; monitor.pause(); sound.engine(false, false); }
  else raf = requestAnimationFrame(tick);
}

function showActions() {   // centred under the result cards (on phones: their own strip below the flight)
  actions.hidden = false;
  if (!desktop.matches) { actions.style.top = ""; actions.style.left = ""; return; }
  const st = stage.getBoundingClientRect();
  const cards = panes.map(p => $(".outcome-card", p.el).getBoundingClientRect());
  actions.style.top = `${Math.round(Math.max(...cards.map(c => c.bottom)) - st.top + 22)}px`;
  actions.style.left = `${Math.round((Math.min(...cards.map(c => c.left)) + Math.max(...cards.map(c => c.right))) / 2 - st.left)}px`;
}

// ---- Countdown: T-3, T-2, T-1, LAUNCH (click, Space or Escape skips it) --------------------------
const cdEl = $("#countdown"), cdText = $("#cd-text");
let skipCountdown = null;
cdEl.addEventListener("click", () => skipCountdown?.());
document.addEventListener("keydown", e => {
  if (!cdEl.hidden && (e.key === " " || e.key === "Escape")) { e.preventDefault(); skipCountdown?.(); }
});
function cdShow(text, cls) {
  cdText.textContent = text;
  cdText.className = "cd-text";
  void cdText.offsetWidth;                 // restart the animation
  cdText.classList.add(cls);
}
async function countdown() {
  cdEl.hidden = false;
  let skipped = false;
  const sleep = ms => new Promise(resolve => {
    const t = setTimeout(resolve, ms);
    skipCountdown = () => { skipped = true; clearTimeout(t); resolve(); };
  });
  for (const k of [3, 2, 1]) {
    if (skipped) break;
    cdShow(`T${MINUS}${k}`, "tick");
    sound.beep(false);
    panes.forEach(p => p.setClock(-k));
    await sleep(800);
  }
  skipCountdown = null;
  panes.forEach(p => p.setClock(0));
  if (skipped) { cdEl.hidden = true; return; }
  cdShow("Launch", "go");
  sound.beep(true);
  setTimeout(() => { if (cdText.textContent === "Launch") cdEl.hidden = true; }, 600);
}

const SURPRISES = {   // each pushes past what the AI practised
  "Storm": { gravity: [9, 10.5], wind: [15, 22], turbulence: [1.2, 2.2], engine: [95, 105] },
  "Heavy planet": { gravity: [11.5, 13.5], wind: [0, 6], turbulence: [0, 0.6], engine: [95, 110] },
  "Failing engine": { gravity: [9.5, 10.5], wind: [0, 6], turbulence: [0, 0.6], engine: [70, 84] },
  "Everything at once": { gravity: [10.5, 12], wind: [12, 18], turbulence: [1, 1.8], engine: [80, 90] },
};
const randomLanding = () => Math.floor(Math.random() * 100000);
let busy = false;

async function launch({ newLanding = true, scroll = true, autoplay = true } = {}) {
  if (busy) return;
  busy = true;
  if (newLanding && !state.landingEdited) state.landing = randomLanding();
  state.landingEdited = false;
  landingEl.value = state.landing;
  const pilots = state.compare ? [state.pilot, state.pilot2] : [state.pilot];
  launchBtn.disabled = true;
  actions.hidden = true;
  const scrolled = new Promise(resolve => {   // bring the flight into view before it starts
    if (!scroll || window.scrollY < 10) return resolve();
    addEventListener("scrollend", resolve, { once: true });
    setTimeout(resolve, 900);
    scrollTo({ top: 0, behavior: reduceMotion() ? "auto" : "smooth" });
  });
  let data;
  try {
    const res = await fetch("api/fly", { method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ pilots, ...state.sliders, landing: state.landing }) });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    data = await res.json();
  } catch (err) {
    setPaneCount(1);
    const o = panes[0].outcomeEl;
    o.hidden = false;
    o.classList.add("show");
    $(".outcome-title", o).textContent = "No signal";
    $(".outcome-sub", o).textContent = "The flight computer didn't answer. If the site was asleep it takes about a minute to wake up; then press Launch again.";
    $(".outcome-stats", o).hidden = true;
    launchBtn.disabled = false;
    busy = false;
    return;
  }
  shareURL();
  for (const p of panes) p.playing = false;
  setPaneCount(data.flights.length);
  const meta = `Landing ${data.landing}${state.note ? `. ${state.note}` : ""}`;
  data.flights.forEach((fl, k) => panes[k].load(fl, meta));
  $(".outcome-stats", panes[0].el).hidden = false;
  state.note = "";
  await new Promise(r => requestAnimationFrame(r));   // let the new pane sizes settle, then measure again
  for (const p of panes) { p.layout(); p.draw(0, 0, 0, performance.now()); }
  await scrolled;
  if (!autoplay) {                                    // reduced motion: wait for the visitor to press Launch
    panes.forEach(p => { p.setClock(0); p.phaseEl.textContent = "Ready"; });
    launchBtn.disabled = false;
    busy = false;
    return;
  }
  await countdown();
  launchBtn.disabled = false;
  busy = false;
  const t0 = performance.now();
  panes.forEach(p => p.start(t0));
  monitor.pause();
  if (!raf) raf = requestAnimationFrame(tick);
}

launchBtn.addEventListener("click", () => launch());
$("#surprise").addEventListener("click", () => {
  const names = Object.keys(SURPRISES), name = names[Math.floor(Math.random() * names.length)];
  for (const [key, [lo, hi]] of Object.entries(SURPRISES[name])) {
    const step = SITE.sliders[key].step;
    state.sliders[key] = Math.round((lo + Math.random() * (hi - lo)) / step) * step;
  }
  state.note = `Surprise: ${name.toLowerCase()}`;
  updateCheck();
  launch();
});
actions.addEventListener("click", e => {
  const act = e.target.closest("[data-act]")?.dataset.act;
  if (act === "relaunch") launch({ newLanding: false });
  if (act === "new") launch({ newLanding: true });
});
landingEl.addEventListener("keydown", e => { if (e.key === "Enter") launch({ newLanding: false }); });

let resizeTimer;
addEventListener("resize", () => {
  clearTimeout(resizeTimer);
  resizeTimer = setTimeout(() => {
    for (const p of panes) if (p.flight) {
      p.layout();
      p.draw(p.done ? p.n - 1 : Math.min(p.lastStep, p.n - 1), 0, 0, performance.now());
    }
    if (!actions.hidden) showActions();
  }, 120);
});
const topbar = $("#topbar");
const onScroll = () => topbar.classList.toggle("solid", scrollY > 24);
addEventListener("scroll", onScroll, { passive: true });
onScroll();

// =====================================================================================
// Sections: big numbers, trade-off, charts, flight records, footer
// =====================================================================================
const R = SITE.results, L = R.landed;
const AG = Object.fromEntries(R.agents.map(a => [a.id, a.name]));
const SC = Object.fromEntries(R.scenarios.map(s => [s.id, s.name]));
const FINAL = "randomized_memory_tuned";
const pval = (agent, reference, scenario) => R.paired.find(t => t.agent === agent && t.reference === reference && t.scenario === scenario)?.p;
const fmtP = p => (p < 0.001 ? "p < 0.001" : `p = ${p.toFixed(3)}`);

const BIG = [   // value, decimals, suffix, label, comparison
  [Math.round(L.strong_wind[FINAL] * 100), 0, "%", "Strong wind", `safe landings. Baseline: ${pct(L.strong_wind.baseline)}.`],
  [Math.round(L.normal[FINAL] * 100), 0, "%", "Normal conditions", `safe landings, average score ${Math.round(R.mean_score.normal[FINAL])}, the best of every agent.`],
  [Math.round(L.harsh[FINAL] * 100), 0, "%", "Everything at once", `${(L.harsh[FINAL] * 100).toFixed(1)}% vs ${(L.harsh.baseline * 100).toFixed(1)}% for the baseline. Still mostly unsolved.`],
  [+R.seconds[FINAL].toFixed(1), 1, "<small>s</small>", "Per landing", `on ${Math.round(R.fuel[FINAL])} fuel. Baseline: ${R.seconds.baseline.toFixed(1)} s on ${Math.round(R.fuel.baseline)}.`],
];
$("#bignums").innerHTML = BIG.map(([v, d, suf, label, vs], k) => `<div class="bignum" data-k="${k}">
  <span class="bignum-value">${v.toFixed(d)}${suf}</span><span class="bignum-label">${label}</span><span class="bignum-vs">${vs}</span></div>`).join("");

$("#tradeoff").innerHTML = `
  <h3 class="h3">Where it is worse</h3>
  <div>
    <p>Heavier gravity and weaker engines than it ever practised with hurt the final AI more than the others. It learned to fly fast and lean, which wins in wind but leaves less spare thrust. The careful pilot keeps a bigger margin. This is the likely reason, not a proven one.</p>
    <table>
      <thead><tr><th scope="col">Landed safely</th><th scope="col">Final AI</th><th scope="col">Careful</th><th scope="col">Baseline</th></tr></thead>
      <tbody>${["heavy_gravity", "weak_engine"].map(sc => `<tr><td>${SC[sc]}</td><td>${pct(L[sc][FINAL])}</td><td>${pct(L[sc].randomized_2m)}</td><td>${pct(L[sc].baseline)}</td></tr>`).join("")}</tbody>
    </table>
  </div>`;

// Heatmap table: one grey ramp, light = lands more often (validated ramp #474747 to #ffffff)
const ramp = r => { const v = Math.round(0x47 + (0xff - 0x47) * r); return `rgb(${v},${v},${v})`; };
const heat = el("table", { class: "heat" });
heat.innerHTML = `<colgroup><col class="rowhead">${R.scenarios.map(() => "<col>").join("")}</colgroup>
  <thead><tr><th scope="col"><span class="visually-hidden">Agent</span></th>${R.scenarios.map(s =>
    `<th scope="col" class="${s.id === "strong_wind_extreme" ? "limit-start" : ""}">${s.name}</th>`).join("")}</tr></thead>
  <tbody>${R.agents.map(a => `<tr class="${a.id === FINAL ? "final" : ""}"><th scope="row">${a.name}</th>${R.scenarios.map(s => {
    const r = L[s.id][a.id];
    return `<td class="${s.id === "strong_wind_extreme" ? "limit-start" : ""}" style="background:${ramp(r)};color:${r > 0.5 ? "#000" : "#fff"}" title="${a.name}, ${s.name}: ${pct(r)} of ${R.landings_per_cell} landings">${pct(r)}</td>`;
  }).join("")}</tr>`).join("")}</tbody>`;
$("#heatmap").append(heat);

// Learning curves: small multiples, one agent each
const CW = 300, CH = 190, M = { l: 46, r: 10, t: 10, b: 26 }, YMIN = -650, YMAX = 330, XMAX = 2.1;
const cx = v => M.l + (v / XMAX) * (CW - M.l - M.r);
const cy = v => M.t + (1 - (Math.max(YMIN, v) - YMIN) / (YMAX - YMIN)) * (CH - M.t - M.b);
for (const a of R.agents) {
  const c = SITE.curves[a.id];
  const box = el("div", { class: `curve${a.id === FINAL ? " final" : ""}` }, `<h4>${a.name}</h4>`);
  const s = svg("svg", { viewBox: `0 0 ${CW} ${CH}`, role: "img", "aria-label": `${a.name}: final practice score ${Math.round(c.mean.at(-1))}` });
  const axis = svg("g", { class: "axis" });
  for (const v of [-600, -400, -200, 0, 200]) {
    axis.append(svg("line", { x1: M.l, x2: CW - M.r, y1: cy(v), y2: cy(v), stroke: v === 200 ? "rgba(255,255,255,.6)" : "rgba(255,255,255,.12)", "stroke-dasharray": v === 200 ? "3 3" : "" }));
    if (v % 400 === 0 || v === 200) {
      const t = svg("text", { x: M.l - 6, y: cy(v) + 4, "text-anchor": "end" }); t.textContent = v < 0 ? MINUS + -v : v; axis.append(t);
    }
  }
  for (const v of [0, 1, 2]) { const t = svg("text", { x: cx(v), y: CH - 6, "text-anchor": "middle" }); t.textContent = `${v}M`; axis.append(t); }
  s.append(axis);
  const band = c.high.map((v, k) => `${cx(c.steps[k])},${cy(v)}`).concat(c.low.map((v, k) => `${cx(c.steps[k])},${cy(v)}`).reverse());
  s.append(svg("polygon", { points: band.join(" "), fill: "rgba(255,255,255,.13)" }));
  s.append(svg("polyline", { points: c.mean.map((v, k) => `${cx(c.steps[k])},${cy(v)}`).join(" "), fill: "none", stroke: "#fff", "stroke-width": a.id === FINAL ? 2 : 1.4 }));
  const endX = cx(c.steps.at(-1)), endY = cy(c.mean.at(-1));
  s.append(svg("circle", { cx: endX, cy: endY, r: 3.5, fill: "#fff", stroke: "#000", "stroke-width": 2 }));
  const endLabel = svg("text", { x: endX - 6, y: endY - 9, "text-anchor": "end", class: "readout" });
  endLabel.textContent = Math.round(c.mean.at(-1));
  s.append(endLabel);
  const cross = svg("line", { y1: M.t, y2: CH - M.b, stroke: "rgba(255,255,255,.5)", visibility: "hidden" });
  const read = svg("text", { x: M.l + 6, y: M.t + 12, class: "readout", visibility: "hidden" });
  const hit = svg("rect", { x: M.l, y: M.t, width: CW - M.l - M.r, height: CH - M.t - M.b, fill: "transparent" });
  hit.addEventListener("pointermove", e => {
    const pt = s.createSVGPoint(); pt.x = e.clientX; pt.y = e.clientY;
    const p = pt.matrixTransform(s.getScreenCTM().inverse());
    const steps = ((p.x - M.l) / (CW - M.l - M.r)) * XMAX;
    let k = 0;
    for (let j = 0; j < c.steps.length; j++) if (Math.abs(c.steps[j] - steps) < Math.abs(c.steps[k] - steps)) k = j;
    cross.setAttribute("x1", cx(c.steps[k])); cross.setAttribute("x2", cx(c.steps[k]));
    read.textContent = `${c.steps[k].toFixed(2)}M steps: ${Math.round(c.mean[k])}`;
    cross.setAttribute("visibility", "visible"); read.setAttribute("visibility", "visible");
  });
  hit.addEventListener("pointerleave", () => { cross.setAttribute("visibility", "hidden"); read.setAttribute("visibility", "hidden"); });
  s.append(cross, read, hit);
  box.append(s);
  $("#curves").append(box);
}

// Paired bars with 95% intervals (the bars grow in when the chart scrolls into view)
function pairs(target, ref, agent, scenarios, notes, names) {
  const series = [[ref, "#6b6b6b"], [agent, "#ffffff"]];
  const wrap = el("div", { class: "pairs" });
  wrap.innerHTML = `<div class="pairs-legend">${series.map(([, col], k) => `<span><span class="swatch" style="background:${col}"></span>${names[k]}</span>`).join("")}</div>` +
    scenarios.map((sc, k) => `<div class="pair-row"><span class="pair-name">${SC[sc]}</span><div class="pair-bars">${series.map(([id, col]) => {
      const r = L[sc][id], [lo, hi] = R.ci95[sc][id];
      return `<div class="bar" title="${AG[id]}, ${SC[sc]}: ${pct(r)} (95% interval ${pct(lo)} to ${pct(hi)})">
        <span class="bar-fill" style="width:${r * 100}%;background:${col}"></span>
        <span class="bar-ci" style="left:${lo * 100}%;width:${(hi - lo) * 100}%"></span>
        <span class="bar-label" style="left:${hi * 100}%">${pct(r)}</span></div>`;
    }).join("")}</div><span class="pair-p">${notes[k]}</span></div>`).join("");
  $(target).append(wrap);
}
pairs("#fragile", "shaped_v2", "shaped_v2_2m", ["normal", "heavy_gravity", "weak_engine"],
  ["both near 100%", fmtP(pval("shaped_v2_2m", "shaped_v2", "heavy_gravity")), fmtP(pval("shaped_v2_2m", "shaped_v2", "weak_engine"))],
  ["1 million practice steps", "2 million practice steps"]);
const memSc = ["strong_wind", "strong_wind_extreme", "harsh", "harsh_extreme"];
pairs("#memory", "randomized_2m", "randomized_memory", memSc, memSc.map(sc => fmtP(pval("randomized_memory", "randomized_2m", sc))),
  ["No memory", "Memory of the last 4 moments"]);

// Flight records: the MP4s from Step 6, shown in black and white to match the site
const CLIPS = [
  ["side_by_side_heavy_gravity.mp4", "Same landing, gravity 12", "Left: trained in normal conditions only. Right: the final AI. One example; across all 300 heavy-gravity test landings the rates are 80% and 92%.", true],
  ["untrained_crash.mp4", "Before learning", "The untrained network on its first test landing."],
  ["baseline_landing.mp4", "Baseline", "The baseline in the normal game."],
  ["final_strong_wind_arrow.mp4", "Final AI in strong wind", "The arrow shows the gust pushing on the lander."],
];
const clips = $("#clips");
for (const [file, title, text, wide] of CLIPS) {
  const fig = el("figure", { class: `clip${wide ? " wide" : ""}` });
  fig.innerHTML = `<video muted loop playsinline preload="metadata" src="videos/${file}" aria-label="${title}"></video><figcaption><b>${title}</b>${text}</figcaption>`;
  clips.append(fig);
}
const videoIO = new IntersectionObserver(entries => {
  for (const e of entries) {
    if (reduceMotion()) continue;
    if (e.isIntersecting) e.target.play().catch(() => {}); else e.target.pause();
  }
}, { threshold: 0.35 });
clips.querySelectorAll("video").forEach(v => videoIO.observe(v));

// Footer links (only the ones that are filled in)
const links = Object.entries(LINKS).filter(([, url]) => url);
if (links.length) {
  const nav = $("#footer-links");
  nav.innerHTML = links.map(([name, url]) => `<a class="glass" href="${url}" target="_blank" rel="noopener">${name}</a>`).join("");
  nav.hidden = false;
}

// Sections fade and slide in; big numbers count up when they arrive.
revealOnScroll(".section > .h2, .section > .lead, .story li, .bignum, .tradeoff, .chart, .clip", node => {
  if (!node.classList.contains("bignum")) return;
  const [v, d, suf] = BIG[node.dataset.k];
  countUp($(".bignum-value", node), v, { decimals: d, suffix: suf, duration: 1600 });
});

// =====================================================================================
// Start: loading screen, then the final AI in the strong-wind test, a landing it has never seen
// =====================================================================================
renderPilots();
updateCheck();
landingEl.value = state.landing;
state.landingEdited = true;   // fly exactly this landing first
await intro;
loader.classList.add("out");
setTimeout(() => loader.remove(), 900);
await wait(reduceMotion() ? 0 : 350);
launch({ scroll: false, autoplay: !reduceMotion() });
