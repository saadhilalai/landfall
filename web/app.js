/* LANDFALL — front end.
   The server flies each landing and sends back a recording (50 snapshots per simulated
   second). This file draws that recording live on a canvas, runs the telemetry overlay,
   the mission-control panel (sliders with physics zones, landing check) and the charts. */
"use strict";

(async function main() {
  const $ = (sel, el = document) => el.querySelector(sel);
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
  const MINUS = "−";
  const signed = (v, d = 1) => (v < -0.05 ? MINUS : "") + Math.abs(v).toFixed(d);
  const pct = r => {
    const v = r * 100;
    return (v >= 99.5 && v < 100 ? v.toFixed(1) : Math.round(v)) + "%";
  };

  const SITE = await fetch("data/site.json").then(r => r.json());
  const P = SITE.physics;
  const W = P.world.w, H = P.world.h, FPS = 50;
  const PILOT = Object.fromEntries(SITE.pilots.map(p => [p.id, p]));
  const reduceMotion = matchMedia("(prefers-reduced-motion: reduce)").matches;

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
    const q = new URLSearchParams({ pilot: state.pilot, ...(state.compare ? { vs: state.pilot2 } : {}),
      ...Object.fromEntries(Object.entries(state.sliders).map(([k, v]) => [k, +v.toFixed(3)])), landing: state.landing });
    history.replaceState(null, "", `?${q}${location.hash}`);
  }

  // =====================================================================================
  // Physics: what the engines can do against gravity, wind and turbulence
  // (all constants measured in the simulator by export_site_data.py)
  // =====================================================================================
  function physics(s) {
    const e = s.engine / 100, g = s.gravity;
    const thrust = P.main_thrust_accel * e;                       // m/s² of push from the main engine
    const sideways = Math.sqrt(Math.max(thrust * thrust - g * g, 0)) + P.side_thrust_accel * e;
    return {
      twr: thrust / g,                                           // thrust to weight
      gravityMax: thrust,                                        // above this gravity, the engine can't hold it up
      engineMin: (g / P.main_thrust_accel) * 100,                // below this engine %, same
      windMax: (sideways * P.total_mass) / P.gust_peak,          // strongest wind the engines can push back against
      turbulenceMax: (P.side_spin_accel * e) / (P.gust_peak * P.spin_per_torque),
    };
  }

  function impossibleFrom(key, ph) {   // [from, to] of the red zone on this slider, or null
    const s = SITE.sliders[key];
    const range = {
      gravity: [ph.gravityMax, s.max],
      engine: [s.min, ph.engineMin],
      wind: [ph.windMax, s.max],
      turbulence: [ph.turbulenceMax, s.max],
    }[key];
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
  // Mission control panel
  // =====================================================================================
  const pilotsEl = $("#pilots"), aboutEl = $("#pilot-about"), compareEl = $("#compare"), pilot2El = $("#pilot2");
  const landingEl = $("#landing"), launchBtn = $("#launch");

  for (const p of SITE.pilots) {
    const b = el("button", { type: "button", class: "pilot", role: "radio", "data-id": p.id }, p.name);
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
  landingEl.addEventListener("input", () => { state.landing = Math.max(0, Math.min(999999, parseInt(landingEl.value || "0", 10))); state.landingEdited = true; });

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
    sliderEls[key] = { ctl, input, out: $("output", ctl), zones: $(".zones", ctl), glyph: $(".zone-glyph", ctl), name: $(".zone-name", ctl) };
  }

  const fmtSlider = (key, v) => {
    const s = SITE.sliders[key];
    const digits = key === "turbulence" ? 2 : key === "engine" ? 0 : 1;
    const text = key === "engine" && v % 1 ? v.toFixed(2) : v.toFixed(digits);
    return `${text}${s.unit ? `<small>${s.unit}</small>` : ""}`;
  };

  function updateSliders(ph) {
    for (const [key, s] of Object.entries(SITE.sliders)) {
      const { input, out, zones, glyph, name } = sliderEls[key];
      const v = state.sliders[key];
      input.value = v;
      out.innerHTML = fmtSlider(key, v);
      // the zone bar: trained (green, thick), beyond (amber, thin), impossible (red, dotted)
      const span = s.max - s.min, at = x => ((x - s.min) / span) * 100;
      const red = impossibleFrom(key, ph);
      const parts = [];
      const cuts = [s.min, s.max, ...s.trained, ...(red || [])].filter(x => x >= s.min && x <= s.max).sort((a, b) => a - b);
      for (let i = 0; i < cuts.length - 1; i++) {
        const a = cuts[i], b = cuts[i + 1];
        if (b - a < 1e-9) continue;
        const zone = zoneOf(key, (a + b) / 2, ph);
        parts.push(`<span class="z-${zone}" style="left:${at(a)}%;width:${at(b) - at(a)}%"></span>`);
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
      const rate = near.t.landed[id];
      const where = near.d < 0.15 ? `Measured: the Step 6 "${near.t.name}" test, ${near.t.landings} unseen landings.`
        : near.d < 0.5 ? `Closest Step 6 test: "${near.t.name}" (${near.t.landings} landings). Your settings are near it, not identical.`
        : `Untested here. The closest Step 6 test is "${near.t.name}", so treat this as a rough guide.`;
      return `<div class="expect"><span class="expect-rate${near.d >= 0.5 ? " untested" : ""}">${pct(rate)}</span>
        <span class="expect-who">${name}, expected landing rate</span><span class="expect-note">${where}</span></div>`;
    }).join("");
  }

  // =====================================================================================
  // The stage: one pane per pilot, each a canvas + telemetry + result screen
  // =====================================================================================
  const stage = $("#stage"), actions = $("#stage-actions");
  const GAUGE_R = 40, GAUGE_LEN = 2 * Math.PI * GAUGE_R * 0.75;   // 270-degree gauges

  function gaugeHTML(key, label, unit) {
    return `<div class="gauge" data-g="${key}">
      <svg viewBox="0 0 92 92" aria-hidden="true">
        <circle class="gauge-track" cx="46" cy="46" r="${GAUGE_R}" stroke-dasharray="${GAUGE_LEN} 999" transform="rotate(135 46 46)"/>
        <circle class="gauge-arc" cx="46" cy="46" r="${GAUGE_R}" stroke-dasharray="${GAUGE_LEN} 999" stroke-dashoffset="${GAUGE_LEN}" transform="rotate(135 46 46)"/>
      </svg><div><b>0</b><span>${label}<br>${unit}</span></div></div>`;
  }

  class Pane {
    constructor() {
      this.el = el("div", { class: "pane" });
      this.el.innerHTML = `
        <canvas aria-hidden="true"></canvas>
        <div class="pane-head"><span><span class="pane-pilot"></span><span class="pane-live">Live simulation</span></span><span class="pane-meta"></span></div>
        <div class="telemetry" aria-hidden="true">
          <div class="gauges">${gaugeHTML("speed", "Speed", "m/s")}${gaugeHTML("alt", "Altitude", "m")}</div>
          <dl class="readouts left">
            <div class="compact-only"><dt>Altitude</dt><dd data-t="alt2"></dd></div>
            <div><dt>Vertical speed</dt><dd data-t="vs"></dd></div>
            <div><dt>Horizontal speed</dt><dd data-t="hs"></dd></div>
          </dl>
          <div class="clock"><span class="tplus" data-t="clock">T+ 00:00.00</span><span class="phase" data-t="phase">Ready</span></div>
          <dl class="readouts right">
            <div><dt>Tilt</dt><dd data-t="tilt"></dd></div>
            <div><dt>Fuel used</dt><dd data-t="fuel"></dd></div>
            <div><dt>Wind</dt><dd data-t="wind"></dd></div>
          </dl>
        </div>
        <div class="outcome" hidden role="status">
          <h2 class="outcome-title"></h2><p class="outcome-sub"></p>
          <dl class="outcome-stats">
            <div><dt>Flight time</dt><dd data-o="time"></dd></div>
            <div><dt>Fuel used</dt><dd data-o="fuel"></dd></div>
            <div><dt>Landing speed</dt><dd data-o="speed"></dd></div>
          </dl>
        </div>`;
      this.canvas = $("canvas", this.el);
      this.ctx = this.canvas.getContext("2d");
      this.t = {};
      for (const node of this.el.querySelectorAll("[data-t]")) this.t[node.dataset.t] = node;
      this.gauges = {};
      for (const g of this.el.querySelectorAll("[data-g]")) this.gauges[g.dataset.g] = { arc: $(".gauge-arc", g), value: $("b", g) };
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
      this.touched = false;
      this.done = false;
      this.crashSparks = false;
      this.outcomeEl.hidden = true;
      this.outcomeEl.classList.remove("show");
      this.el.classList.remove("ended");
      $(".pane-pilot", this.el).textContent = PILOT[flight.pilot].name;
      $(".pane-meta", this.el).textContent = meta;
      this.layout();
      this.setTelemetry(0, 0);
      this.draw(0, 0, 0);
    }

    layout() {
      const r = this.canvas.getBoundingClientRect();
      this.dpr = Math.min(window.devicePixelRatio || 1, 2);
      this.w = r.width; this.h = r.height;
      this.canvas.width = Math.round(r.width * this.dpr);
      this.canvas.height = Math.round(r.height * this.dpr);
      const tele = $(".telemetry", this.el).getBoundingClientRect().height;
      stage.style.setProperty("--tele-h", `${Math.round(tele)}px`);
      const top = r.width < 760 ? 92 : 118, bottom = tele + 18;
      // Show the world from just under the lowest ground to the top, where the lander starts.
      const floor = Math.max(0, Math.min(...this.flight.terrain.map(p => p[1])) - 0.5);
      this.s = Math.max(4, Math.min((r.width - 24) / W, (r.height - top - bottom) / (H - floor)));
      this.ox = (r.width - W * this.s) / 2;
      this.base = r.height - bottom + floor * this.s;      // screen y of the world's y = 0
      this.skyTop = top - 40;
      this.streaks = Array.from({ length: Math.round(r.width / 16) }, () => ({
        x: Math.random() * r.width, y: this.skyTop + Math.random() * Math.max(40, this.Y(this.flight.pad[2]) - 24 - this.skyTop),
        len: 18 + Math.random() * 46, k: 0.6 + Math.random() * 0.8,
      }));
    }

    X(x) { return this.ox + x * this.s; }
    Y(y) { return this.base - y * this.s; }

    start(t0) { this.t0 = t0; this.prev = t0; this.playing = true; }

    // Spawn engine exhaust and touchdown dust for every simulated step we pass.
    simulateSteps(upTo) {
      const f = this.f;
      for (let k = this.lastStep + 1; k <= upTo; k++) {
        const a = f.a[k], x = f.x[k], y = f.y[k], act = f.act[k];
        if (act === 2) {
          const dx = Math.sin(a), dy = -Math.cos(a);              // down along the body
          for (let i = 0; i < 3; i++) {
            const sp = 6 + Math.random() * 4, spread = (Math.random() - 0.5) * 2.4;
            this.particles.push({ x: x + dx * 0.3, y: y + dy * 0.3,
              vx: f.vx[k] + dx * sp - dy * spread, vy: f.vy[k] + dy * sp + dx * spread,
              life: 0, max: 0.22 + Math.random() * 0.25, r: 0.05 + Math.random() * 0.07 });
          }
        } else if (act === 1 || act === 3) {
          const d = act - 2, ox = -d * Math.cos(a), oy = -d * Math.sin(a);
          const px = x + ox * 0.4 - Math.sin(a) * 0.567, py = y + oy * 0.4 + Math.cos(a) * 0.467;
          this.particles.push({ x: px, y: py, vx: f.vx[k] + ox * 3.5, vy: f.vy[k] + oy * 3.5,
            life: 0, max: 0.16 + Math.random() * 0.1, r: 0.04 + Math.random() * 0.03 });
        }
        if (!this.touched && f.legs[k] > 0) {
          this.touched = true;
          for (const [lx, ly] of [[f.l0x[k], f.l0y[k]], [f.l1x[k], f.l1y[k]]]) {
            for (let i = 0; i < 10; i++) {
              const dir = Math.random() < 0.5 ? -1 : 1;
              this.particles.push({ x: lx, y: ly - 0.2, vx: dir * (0.8 + Math.random() * 2.6), vy: Math.random() * 0.6,
                life: 0, max: 0.5 + Math.random() * 0.5, r: 0.03 + Math.random() * 0.04, dust: true });
            }
          }
        }
      }
      this.lastStep = Math.max(this.lastStep, upTo);
    }

    explode() {
      const f = this.f, k = this.n - 1;
      for (let i = 0; i < 46; i++) {
        const ang = Math.random() * Math.PI, sp = 2 + Math.random() * 7;
        this.particles.push({ x: f.x[k], y: f.y[k], vx: Math.cos(ang) * sp * (Math.random() < 0.5 ? -1 : 1), vy: Math.sin(ang) * sp,
          life: 0, max: 0.7 + Math.random() * 0.9, r: 0.03 + Math.random() * 0.05, spark: true });
      }
    }

    frame(now) {
      const dt = Math.min(0.05, (now - this.prev) / 1000);
      this.prev = now;
      const t = (now - this.t0) / 1000 - 0.35;                  // a beat of stillness before ignition
      const pos = Math.max(0, Math.min(t * FPS, this.n - 1));
      const i = Math.floor(pos), frac = pos - i;
      this.simulateSteps(i);
      if (pos >= this.n - 1 && !this.crashSparks) {
        this.crashSparks = true;
        if (this.flight.outcome === "crashed") this.explode();
      }
      this.setTelemetry(i, frac);
      this.draw(i, frac, dt);
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

    setTelemetry(i, frac) {
      const f = this.f, T = this.t;
      const x = this.lerp("vx", i, frac), y = this.lerp("vy", i, frac);
      const alt = Math.max(0, this.lerp("y", i, frac) - this.flight.pad[2] - P.rest_height);
      const speed = Math.hypot(x, y);
      const sec = (i + frac) / FPS;
      T.clock.textContent = `T+ 00:${String(Math.floor(sec)).padStart(2, "0")}.${String(Math.floor((sec % 1) * 100)).padStart(2, "0")}`;
      T.vs.innerHTML = `${signed(y)}<small>m/s</small>`;
      T.hs.innerHTML = `${signed(x)}<small>m/s</small>`;
      T.alt2.innerHTML = `${alt.toFixed(1)}<small>m</small>`;
      let deg = (this.lerp("a", i, frac) * 180) / Math.PI;
      deg = ((deg + 180) % 360 + 360) % 360 - 180;
      T.tilt.innerHTML = `${signed(deg, 0)}<small>deg</small>`;
      T.fuel.textContent = Math.round(f.fuel[i]);
      const w = f.wind[i];
      T.wind.innerHTML = Math.abs(w) < 0.05 ? "0.0" :
        `<svg class="wind-arrow" viewBox="0 0 26 12" style="transform:scaleX(${w < 0 ? -1 : 1})"><path d="M1 6h20M16 1l6 5-6 5" fill="none" stroke="#fff" stroke-width="1.5"/></svg>${Math.abs(w).toFixed(1)}`;
      const act = f.act[i];
      T.phase.textContent = i >= this.n - 1 ? "Signal end" : f.legs[i] > 0 ? "Ground contact" :
        act === 2 ? "Main engine" : act ? "Steering thrusters" : "Coasting";
      this.setGauge("speed", speed, 12, speed.toFixed(1));
      this.setGauge("alt", alt, 11, alt.toFixed(1));
    }

    setGauge(key, value, max, text) {
      const g = this.gauges[key];
      g.arc.setAttribute("stroke-dashoffset", String(GAUGE_LEN * (1 - Math.min(value / max, 1))));
      g.value.textContent = text;
    }

    draw(i, frac, dt) {
      const ctx = this.ctx, f = this.f, s = this.s;
      ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
      ctx.fillStyle = "#000";
      ctx.fillRect(0, 0, this.w, this.h);

      // Wind, made visible: faint streaks moving with the gust pushing on the lander right now.
      const wind = f.wind[i];
      if (Math.abs(wind) > 0.05 && !reduceMotion) {
        ctx.strokeStyle = `rgba(255,255,255,${Math.min(0.32, 0.06 + (Math.abs(wind) / 25) * 0.3)})`;
        ctx.lineWidth = 1;
        ctx.beginPath();
        for (const st of this.streaks) {
          st.x += wind * 26 * st.k * dt;
          if (st.x > this.w + 60) st.x -= this.w + 120;
          if (st.x < -60) st.x += this.w + 120;
          ctx.moveTo(st.x, st.y);
          ctx.lineTo(st.x - Math.sign(wind) * st.len * Math.min(1, Math.abs(wind) / 10 + 0.3), st.y);
        }
        ctx.stroke();
      }

      // Ground: a thin white line, continued flat to the screen edges.
      const ter = this.flight.terrain;
      ctx.strokeStyle = "#fff";
      ctx.lineWidth = 1.25;
      ctx.beginPath();
      ctx.moveTo(0, this.Y(ter[0][1]));
      for (const [x, y] of ter) ctx.lineTo(this.X(x), this.Y(y));
      ctx.lineTo(this.w, this.Y(ter[ter.length - 1][1]));
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
      ctx.fillStyle = "rgba(255,255,255,0.32)";
      for (let k = 0; k <= i; k += 4) ctx.fillRect(this.X(f.x[k]) - 0.75, this.Y(f.y[k]) - 0.75, 1.5, 1.5);

      // Exhaust, dust and sparks
      const alive = [];
      for (const p of this.particles) {
        p.life += dt;
        if (p.life >= p.max) continue;
        if (p.spark) p.vy -= 9 * dt;
        const drag = p.dust ? 2.5 : 1.6;
        p.vx -= p.vx * drag * dt; p.vy -= p.vy * drag * dt;
        p.x += p.vx * dt; p.y += p.vy * dt;
        const k = 1 - p.life / p.max;
        ctx.fillStyle = p.dust ? `rgba(180,180,180,${0.7 * k})` : `rgba(255,255,255,${0.9 * k})`;
        ctx.beginPath(); ctx.arc(this.X(p.x), this.Y(p.y), Math.max(0.6, p.r * s * (p.spark ? 1 : 0.6 + k * 0.6)), 0, 7); ctx.fill();
        alive.push(p);
      }
      this.particles = alive;

      // Main engine flame
      const x = this.lerp("x", i, frac), y = this.lerp("y", i, frac), a = this.lerp("a", i, frac);
      if (f.act[i] === 2 && i < this.n - 1) {
        const dx = Math.sin(a), dy = -Math.cos(a), nx = x + dx * 0.33, ny = y + dy * 0.33;
        const len = 0.55 + Math.random() * 0.3, half = 0.13;
        ctx.save();
        ctx.shadowColor = "rgba(255,255,255,0.9)";
        ctx.shadowBlur = 14;
        ctx.fillStyle = "rgba(255,255,255,0.95)";
        ctx.beginPath();
        ctx.moveTo(this.X(nx - dy * half), this.Y(ny + dx * half));
        ctx.lineTo(this.X(nx + dy * half), this.Y(ny - dx * half));
        ctx.lineTo(this.X(nx + dx * len), this.Y(ny + dy * len));
        ctx.closePath(); ctx.fill();
        ctx.restore();
      }

      // The lander: legs, then body
      const lost = this.done && this.flight.outcome === "crashed";
      ctx.fillStyle = lost ? "#7a7a7a" : "#fff";
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
      $('[data-o="time"]', this.el).innerHTML = `${fl.seconds.toFixed(1)}<small>s</small>`;
      $('[data-o="fuel"]', this.el).textContent = Math.round(fl.fuel);
      $('[data-o="speed"]', this.el).innerHTML = fl.contact_speed == null ? "n/a" : `${fl.contact_speed.toFixed(1)}<small>m/s</small>`;
      this.outcomeEl.hidden = false;
      this.el.classList.add("ended");
      requestAnimationFrame(() => this.outcomeEl.classList.add("show"));
    }
  }

  let panes = [];
  function setPaneCount(n) {
    while (panes.length < n) { const p = new Pane(); stage.insertBefore(p.el, actions); panes.push(p); }
    while (panes.length > n) panes.pop().el.remove();
    stage.dataset.panes = String(n);
  }

  let raf = null;
  function tick(now) {
    let finished = true;
    for (const p of panes) if (!p.frame(now)) finished = false;
    if (panes.every(p => p.done) && actions.hidden) showActions();
    raf = finished ? null : requestAnimationFrame(tick);
  }

  const narrow = matchMedia("(max-width: 1080px)");
  function showActions() {   // just under the result stats (on phones: in their own strip below the flight)
    actions.hidden = false;
    if (narrow.matches) { actions.style.top = ""; return; }
    const top = stage.getBoundingClientRect().top;
    const below = Math.max(...panes.map(p => $(".outcome-stats", p.el).getBoundingClientRect().bottom));
    actions.style.top = `${Math.round(below - top + 30)}px`;
  }

  const SURPRISES = {   // each pushes past what the AI practised
    "Storm": { gravity: [9, 10.5], wind: [15, 22], turbulence: [1.2, 2.2], engine: [95, 105] },
    "Heavy planet": { gravity: [11.5, 13.5], wind: [0, 6], turbulence: [0, 0.6], engine: [95, 110] },
    "Failing engine": { gravity: [9.5, 10.5], wind: [0, 6], turbulence: [0, 0.6], engine: [70, 84] },
    "Everything at once": { gravity: [10.5, 12], wind: [12, 18], turbulence: [1, 1.8], engine: [80, 90] },
  };
  const randomLanding = () => Math.floor(Math.random() * 100000);

  async function launch({ newLanding = true, scroll = true } = {}) {
    if (newLanding && !state.landingEdited) state.landing = randomLanding();
    state.landingEdited = false;
    landingEl.value = state.landing;
    const pilots = state.compare ? [state.pilot, state.pilot2] : [state.pilot];
    launchBtn.disabled = true;
    actions.hidden = true;
    const scrolled = new Promise(resolve => {   // bring the flight into view before it starts
      if (!scroll || window.scrollY < 10) return resolve();
      window.addEventListener("scrollend", resolve, { once: true });
      setTimeout(resolve, 900);
      window.scrollTo({ top: 0, behavior: reduceMotion ? "auto" : "smooth" });
    });
    let data;
    try {
      const res = await fetch("api/fly", { method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ pilots, ...state.sliders, landing: state.landing }) });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      data = await res.json();
    } catch (err) {
      setPaneCount(1);
      panes[0].outcomeEl.hidden = false;
      panes[0].outcomeEl.classList.add("show");
      $(".outcome-title", panes[0].el).textContent = "No signal";
      $(".outcome-sub", panes[0].el).textContent = "The flight computer didn't answer. If the site was asleep it takes about a minute to wake up; then press Launch again.";
      launchBtn.disabled = false;
      return;
    }
    launchBtn.disabled = false;
    shareURL();
    setPaneCount(data.flights.length);
    const meta = `Landing ${data.landing}${state.note ? `. ${state.note}` : ""}`;
    data.flights.forEach((fl, k) => panes[k].load(fl, meta));
    state.note = "";
    await new Promise(r => requestAnimationFrame(r));   // let the new pane sizes settle, then measure again
    for (const p of panes) { p.layout(); p.draw(0, 0, 0); }
    await scrolled;
    if (reduceMotion) {   // no autoplay animation: show the end state straight away
      for (const p of panes) { p.lastStep = p.n - 1; p.draw(p.n - 1, 0, 0); p.setTelemetry(p.n - 1, 0); p.done = true; p.showOutcome(); }
      showActions();
      return;
    }
    const t0 = performance.now();
    panes.forEach(p => p.start(t0));
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
  document.addEventListener("keydown", e => {
    if (e.key === "Enter" && e.target === landingEl) launch({ newLanding: false });
  });

  let resizeTimer;
  window.addEventListener("resize", () => {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(() => {
      for (const p of panes) if (p.flight) {
        p.layout();
        const i = p.done ? p.n - 1 : Math.min(p.lastStep, p.n - 1);
        p.draw(i, 0, 0);
      }
      if (!actions.hidden) showActions();
    }, 120);
  });
  const topbar = $("#topbar");
  const onScroll = () => topbar.classList.toggle("solid", window.scrollY > 24);
  window.addEventListener("scroll", onScroll, { passive: true });
  onScroll();

  // =====================================================================================
  // Sections: big numbers, trade-off, charts, flight records
  // =====================================================================================
  const R = SITE.results, L = R.landed;
  const AG = Object.fromEntries(R.agents.map(a => [a.id, a.name]));
  const SC = Object.fromEntries(R.scenarios.map(s => [s.id, s.name]));
  const FINAL = "randomized_memory_tuned";
  const pval = (agent, reference, scenario) => R.paired.find(t => t.agent === agent && t.reference === reference && t.scenario === scenario)?.p;
  const fmtP = p => (p < 0.001 ? "p < 0.001" : `p = ${p.toFixed(3)}`);

  $("#bignums").innerHTML = [
    [pct(L.strong_wind[FINAL]), "Strong wind", `safe landings. Baseline: ${pct(L.strong_wind.baseline)}.`],
    [pct(L.normal[FINAL]), "Normal conditions", `safe landings, average score ${Math.round(R.mean_score.normal[FINAL])}, the best of every agent.`],
    [pct(L.harsh[FINAL]), "Everything at once", `${(L.harsh[FINAL] * 100).toFixed(1)}% vs ${(L.harsh.baseline * 100).toFixed(1)}% for the baseline. Still mostly unsolved.`],
    [`${R.seconds[FINAL].toFixed(1)}<small>s</small>`, "Per landing", `on ${Math.round(R.fuel[FINAL])} fuel. Baseline: ${R.seconds.baseline.toFixed(1)} s on ${Math.round(R.fuel.baseline)}.`],
  ].map(([v, label, vs]) => `<div class="bignum"><span class="bignum-value">${v}</span><span class="bignum-label">${label}</span><span class="bignum-vs">${vs}</span></div>`).join("");

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
  heat.innerHTML = `<colgroup><col class="rowhead">${R.scenarios.map(() => "<col>").join("")}</colgroup><thead><tr><th scope="col"><span class="visually-hidden">Agent</span></th>${R.scenarios.map(s =>
    `<th scope="col" class="${s.id === "strong_wind_extreme" ? "limit-start" : ""}">${s.name}</th>`).join("")}</tr></thead>
    <tbody>${R.agents.map(a => `<tr class="${a.id === FINAL ? "final" : ""}"><th scope="row">${a.name}</th>${R.scenarios.map(s => {
      const r = L[s.id][a.id];
      return `<td class="${s.id === "strong_wind_extreme" ? "limit-start" : ""}" style="background:${ramp(r)};color:${r > 0.5 ? "#000" : "#fff"}" title="${a.name}, ${s.name}: ${pct(r)} of ${R.landings_per_cell} landings">${pct(r)}</td>`;
    }).join("")}</tr>`).join("")}</tbody>`;
  $("#heatmap").append(heat);

  // Learning curves: small multiples, one agent each
  const CW = 300, CH = 190, M = { l: 46, r: 10, t: 10, b: 26 }, YMIN = -650, YMAX = 330, XMAX = 2.1;
  const cx = v => M.l + (v / XMAX) * (CW - M.l - M.r), cy = v => M.t + (1 - (Math.max(YMIN, v) - YMIN) / (YMAX - YMIN)) * (CH - M.t - M.b);
  for (const a of R.agents) {
    const c = SITE.curves[a.id];
    const box = el("div", { class: `curve${a.id === FINAL ? " final" : ""}` }, `<h4>${a.name}</h4>`);
    const s = svg("svg", { viewBox: `0 0 ${CW} ${CH}`, role: "img", "aria-label": `${a.name}: final practice score ${Math.round(c.mean.at(-1))}` });
    const axis = svg("g", { class: "axis" });
    for (const v of [-600, -400, -200, 0, 200]) {
      axis.append(svg("line", { x1: M.l, x2: CW - M.r, y1: cy(v), y2: cy(v), stroke: v === 200 ? "rgba(255,255,255,.6)" : "rgba(255,255,255,.12)", "stroke-dasharray": v === 200 ? "3 3" : "" }));
      if (v % 400 === 0 || v === 200) {   // label every other line, so the numbers stay readable when small
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
    const endLabel = svg("text", { x: endX - 6, y: endY - 9, "text-anchor": "end", class: "readout" }); endLabel.textContent = Math.round(c.mean.at(-1)); s.append(endLabel);
    // hover: crosshair + readout
    const cross = svg("line", { y1: M.t, y2: CH - M.b, stroke: "rgba(255,255,255,.5)", visibility: "hidden" });
    const read = svg("text", { x: M.l + 6, y: M.t + 12, class: "readout", visibility: "hidden" });
    const hit = svg("rect", { x: M.l, y: M.t, width: CW - M.l - M.r, height: CH - M.t - M.b, fill: "transparent" });
    hit.addEventListener("pointermove", e => {
      const pt = s.createSVGPoint(); pt.x = e.clientX; pt.y = e.clientY;
      const p = pt.matrixTransform(s.getScreenCTM().inverse());
      const steps = ((p.x - M.l) / (CW - M.l - M.r)) * XMAX;
      let k = 0; for (let j = 0; j < c.steps.length; j++) if (Math.abs(c.steps[j] - steps) < Math.abs(c.steps[k] - steps)) k = j;
      cross.setAttribute("x1", cx(c.steps[k])); cross.setAttribute("x2", cx(c.steps[k]));
      read.textContent = `${c.steps[k].toFixed(2)}M steps: ${Math.round(c.mean[k])}`;
      cross.setAttribute("visibility", "visible"); read.setAttribute("visibility", "visible");
    });
    hit.addEventListener("pointerleave", () => { cross.setAttribute("visibility", "hidden"); read.setAttribute("visibility", "hidden"); });
    s.append(cross, read, hit);
    box.append(s);
    $("#curves").append(box);
  }

  // Paired bars with 95% intervals
  function pairs(target, ref, agent, scenarios, notes, names) {
    const series = [[ref, "#6b6b6b"], [agent, "#ffffff"]];
    const wrap = el("div", { class: "pairs" });
    wrap.innerHTML = `<div class="pairs-legend">${series.map(([id, col], k) => `<span><span class="swatch" style="background:${col}"></span>${names[k]}</span>`).join("")}</div>` +
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
  const io = new IntersectionObserver(entries => {
    for (const e of entries) {
      if (reduceMotion) continue;
      if (e.isIntersecting) e.target.play().catch(() => {}); else e.target.pause();
    }
  }, { threshold: 0.35 });
  clips.querySelectorAll("video").forEach(v => io.observe(v));

  // =====================================================================================
  // First flight: the final AI in the strong-wind test, a landing it has never seen
  // =====================================================================================
  renderPilots();
  updateCheck();
  landingEl.value = state.landing;
  state.landingEdited = true;   // fly exactly this landing first
  launch({ scroll: false });
})();
