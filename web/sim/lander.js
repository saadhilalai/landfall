/* lander.js — the LunarLander game, rewritten in JavaScript so landings fly in the browser.

   A line-by-line port of Gymnasium's LunarLander-v3 (gymnasium/envs/box2d/lunar_lander.py),
   with the project's own additions from Python:
     - PhysicsConditions (domain_randomization.py): gravity, wind, turbulence, engine power
     - LandingStats (landing_stats.py): how the landing ended, judged by the physics engine
     - record_flight (formerly app.py): the flight recording the page draws

   The physics engine is planck.js, a JavaScript rewrite of Box2D (the engine the Python game
   uses), with the same settings. The random draws are bit-for-bit NumPy's (rng.js), so a
   landing number gives exactly the same terrain, starting push and gust pattern as in Python.
   The flight itself can drift apart from Python's after a while: the two engines round
   numbers slightly differently (64-bit vs 32-bit), and a landing is chaotic enough for tiny
   differences to grow. check_browser_sim.py measures that the landing RATES still match. */

import { World, Vec2, Edge, Polygon, Box, RevoluteJoint } from "../vendor/planck-1.5.0.js";
import { Generator } from "./rng.js";

// ---- The game's constants (lunar_lander.py) -----------------------------------------------------
const FPS = 50;
const SCALE = 30.0;
const MAIN_ENGINE_POWER = 13.0;
const SIDE_ENGINE_POWER = 0.6;
const INITIAL_RANDOM = 1000.0;
const LANDER_POLY = [[-14, 17], [-17, 0], [-17, -10], [17, -10], [17, 0], [14, 17]];
const LEG_AWAY = 20, LEG_DOWN = 18, LEG_W = 2, LEG_H = 8, LEG_SPRING_TORQUE = 40;
const SIDE_ENGINE_HEIGHT = 14, SIDE_ENGINE_AWAY = 12, MAIN_ENGINE_Y_LOCATION = 4;
const VIEWPORT_W = 600, VIEWPORT_H = 400;
const W = VIEWPORT_W / SCALE, H = VIEWPORT_H / SCALE;
const CHUNKS = 11;
const MAX_STEPS = 1000;   // the time limit Gymnasium adds to LunarLander-v3 (20 simulated seconds)

// ---- The project's additions --------------------------------------------------------------------
const PAD_HALF_WIDTH = 0.2;                    // landing_stats.py
export const FUEL_COST = [0, 0.1, 1, 0.1];     // nothing, left, main, right (landing_stats.py)

const windCurve = i => Math.tanh(Math.sin(0.02 * i) + Math.sin(Math.PI * 0.01 * i));

export class LunarLander {
  /** conditions: gravity (m/s², pointing down), wind, turbulence, engine (multiplier, 1 = normal) */
  constructor({ gravity = 10, wind = 0, turbulence = 0, engine = 1 } = {}) {
    this.gravity = -gravity;
    this.windPower = wind;
    this.turbulencePower = turbulence;
    this.enableWind = wind > 0 || turbulence > 0;
    this.mainPower = MAIN_ENGINE_POWER * engine;
    this.sidePower = SIDE_ENGINE_POWER * engine;
  }

  reset(seed) {
    this.rng = new Generator(seed);
    this.world = new World({ gravity: new Vec2(0, this.gravity) });
    this.world.on("begin-contact", c => this._contact(c, true));
    this.world.on("end-contact", c => this._contact(c, false));
    this.gameOver = false;
    this.prevShaping = null;
    this.steps = 0;

    // Terrain: 11 random heights, flattened under the pad, then smoothed
    const height = Array.from({ length: CHUNKS + 1 }, () => this.rng.uniform(0, H / 2));
    const chunkX = Array.from({ length: CHUNKS }, (_, i) => (W / (CHUNKS - 1)) * i);
    this.helipadX1 = chunkX[Math.floor(CHUNKS / 2) - 1];
    this.helipadX2 = chunkX[Math.floor(CHUNKS / 2) + 1];
    this.helipadY = H / 4;
    for (let k = -2; k <= 2; k++) height[Math.floor(CHUNKS / 2) + k] = this.helipadY;
    const at = i => height[(i + height.length) % height.length];        // Python's height[-1] = the last one
    const smoothY = Array.from({ length: CHUNKS }, (_, i) => 0.33 * (at(i - 1) + at(i) + at(i + 1)));

    this.moon = this.world.createBody();
    this.moon.createFixture(new Edge(new Vec2(0, 0), new Vec2(W, 0)));
    this.terrain = [];
    for (let i = 0; i < CHUNKS - 1; i++) {
      const p1 = new Vec2(chunkX[i], smoothY[i]), p2 = new Vec2(chunkX[i + 1], smoothY[i + 1]);
      this.moon.createFixture(new Edge(p1, p2), { density: 0, friction: 0.1 });
      this.terrain.push([chunkX[i], smoothY[i]]);
    }
    this.terrain.push([chunkX[CHUNKS - 1], smoothY[CHUNKS - 1]]);

    // The lander, and its random first push
    const x0 = VIEWPORT_W / SCALE / 2, y0 = VIEWPORT_H / SCALE;
    this.lander = this.world.createBody({ type: "dynamic", position: new Vec2(x0, y0), angle: 0 });
    this.lander.createFixture(new Polygon(LANDER_POLY.map(([x, y]) => new Vec2(x / SCALE, y / SCALE))),
      { density: 5.0, friction: 0.1, filterCategoryBits: 0x0010, filterMaskBits: 0x001, restitution: 0 });
    this.lander.applyForceToCenter(new Vec2(this.rng.uniform(-INITIAL_RANDOM, INITIAL_RANDOM),
                                            this.rng.uniform(-INITIAL_RANDOM, INITIAL_RANDOM)), true);
    if (this.enableWind) {
      this.windIdx = this.rng.integers(-9999, 9999);
      this.torqueIdx = this.rng.integers(-9999, 9999);
    }

    // Two spring-loaded legs
    this.legs = [];
    for (const i of [-1, 1]) {
      const leg = this.world.createBody({ type: "dynamic", position: new Vec2(x0 - (i * LEG_AWAY) / SCALE, y0),
                                          angle: i * 0.05 });
      leg.createFixture(new Box(LEG_W / SCALE, LEG_H / SCALE),
        { density: 1.0, restitution: 0, filterCategoryBits: 0x0020, filterMaskBits: 0x001 });
      leg.groundContact = false;
      this.world.createJoint(new RevoluteJoint({
        localAnchorA: new Vec2(0, 0), localAnchorB: new Vec2((i * LEG_AWAY) / SCALE, LEG_DOWN / SCALE),
        enableMotor: true, enableLimit: true, maxMotorTorque: LEG_SPRING_TORQUE, motorSpeed: 0.3 * i,
        lowerAngle: i === -1 ? 0.9 - 0.5 : -0.9, upperAngle: i === -1 ? 0.9 : -0.9 + 0.5,
      }, this.lander, leg));
      this.legs.push(leg);
    }
    return this._step(0).obs;   // the game's reset ends with one "do nothing" step
  }

  _contact(contact, begin) {
    const a = contact.getFixtureA().getBody(), b = contact.getFixtureB().getBody();
    if (begin && (a === this.lander || b === this.lander)) this.gameOver = true;
    for (const leg of this.legs || []) if (leg === a || leg === b) leg.groundContact = begin;
  }

  step(action) {
    const result = this._step(action);
    this.steps += 1;
    result.truncated = this.steps >= MAX_STEPS;
    return result;
  }

  _step(action) {
    const lander = this.lander;
    // Wind and turbulence, only while no leg touches the ground
    if (this.enableWind && !(this.legs[0].groundContact || this.legs[1].groundContact)) {
      lander.applyForceToCenter(new Vec2(windCurve(this.windIdx) * this.windPower, 0), true);
      this.windIdx += 1;
      lander.applyTorque(windCurve(this.torqueIdx) * this.turbulencePower, true);
      this.torqueIdx += 1;
    }

    const angle = lander.getAngle();
    const tip = [Math.sin(angle), Math.cos(angle)];
    const side = [-tip[1], tip[0]];
    const dispersion = [this.rng.uniform(-1, 1) / SCALE, this.rng.uniform(-1, 1) / SCALE];
    const pos = lander.getPosition();

    let mPower = 0;
    if (action === 2) {             // main engine
      mPower = 1;
      const ox = tip[0] * (MAIN_ENGINE_Y_LOCATION / SCALE + 2 * dispersion[0]) + side[0] * dispersion[1];
      const oy = -tip[1] * (MAIN_ENGINE_Y_LOCATION / SCALE + 2 * dispersion[0]) - side[1] * dispersion[1];
      lander.applyLinearImpulse(new Vec2(-ox * this.mainPower * mPower, -oy * this.mainPower * mPower),
                                new Vec2(pos.x + ox, pos.y + oy), true);
    }
    let sPower = 0;
    if (action === 1 || action === 3) {   // side thrusters: 1 = left, 3 = right
      const direction = action - 2;
      sPower = 1;
      const ox = tip[0] * dispersion[0] + side[0] * (3 * dispersion[1] + (direction * SIDE_ENGINE_AWAY) / SCALE);
      const oy = -tip[1] * dispersion[0] - side[1] * (3 * dispersion[1] + (direction * SIDE_ENGINE_AWAY) / SCALE);
      lander.applyLinearImpulse(new Vec2(-ox * this.sidePower * sPower, -oy * this.sidePower * sPower),
        new Vec2(pos.x + ox - (tip[0] * 17) / SCALE, pos.y + oy + (tip[1] * SIDE_ENGINE_HEIGHT) / SCALE), true);
    }

    this.world.step(1.0 / FPS, 6 * 30, 2 * 30);

    const p = lander.getPosition(), v = lander.getLinearVelocity();
    const state = [
      (p.x - VIEWPORT_W / SCALE / 2) / (VIEWPORT_W / SCALE / 2),
      (p.y - (this.helipadY + LEG_DOWN / SCALE)) / (VIEWPORT_H / SCALE / 2),
      (v.x * (VIEWPORT_W / SCALE / 2)) / FPS,
      (v.y * (VIEWPORT_H / SCALE / 2)) / FPS,
      lander.getAngle(),
      (20.0 * lander.getAngularVelocity()) / FPS,
      this.legs[0].groundContact ? 1 : 0,
      this.legs[1].groundContact ? 1 : 0,
    ];

    // The official score
    const shaping = -100 * Math.hypot(state[0], state[1]) - 100 * Math.hypot(state[2], state[3])
                    - 100 * Math.abs(state[4]) + 10 * state[6] + 10 * state[7];
    let reward = this.prevShaping === null ? 0 : shaping - this.prevShaping;
    this.prevShaping = shaping;
    reward -= mPower * 0.3 + sPower * 0.03;
    let terminated = false;
    if (this.gameOver || Math.abs(state[0]) >= 1) { terminated = true; reward = -100; }
    if (!lander.isAwake()) { terminated = true; reward = 100; }
    return { obs: Float32Array.from(state), reward, terminated, truncated: false };
  }

  /** How many legs (0-2) are physically touching the ground: asks the physics engine, not the
      game's leg flags, which switch off when a leg slides from one ground segment to the next. */
  legsTouching() {
    let n = 0;
    for (const leg of this.legs) {
      for (let ce = leg.getContactList(); ce; ce = ce.next) {
        if (ce.contact.isTouching()) { n += 1; break; }
      }
    }
    return n;
  }

  /** The sideways wind force applied on the last step (0 when landed or with wind off). */
  currentWind() {
    if (!this.enableWind || this.legs[0].groundContact || this.legs[1].groundContact) return 0;
    return windCurve(this.windIdx - 1) * this.windPower;
  }

  outcome(obs) {
    if (this.gameOver) return "crashed";              // the body (not the legs) hit the ground
    if (Math.abs(obs[0]) >= 1) return "flew_away";     // off the side of the screen
    if (!this.lander.isAwake()) return Math.abs(obs[0]) <= PAD_HALF_WIDTH ? "landed" : "landed_off_pad";
    return "never_settled";                           // still hovering when time ran out
  }
}

// ---- One recorded flight, in the format the page draws -------------------------------------------
const r3 = v => Math.round(v * 1000) / 1000;
const r2 = v => Math.round(v * 100) / 100;
const r1 = v => Math.round(v * 10) / 10;

/** Fly one landing. pilot: a Pilot (policy.js); sliders: {gravity, wind, turbulence, engine (%)}. */
export function recordFlight(pilotId, pilot, sliders, landing) {
  const game = new LunarLander({ gravity: sliders.gravity, wind: sliders.wind,
                                 turbulence: sliders.turbulence, engine: sliders.engine / 100 });
  let obs = game.reset(landing);
  pilot.reset(obs);
  const picker = new Generator(1_000_000 + landing);    // only the untrained pilot uses it
  const random = () => picker.random();

  const keys = ["x", "y", "a", "vx", "vy", "l0x", "l0y", "l0a", "l1x", "l1y", "l1a"];
  const rec = Object.fromEntries([...keys, "act", "wind", "fuel", "legs"].map(k => [k, []]));
  let fuel = 0, contactSpeed = null, score = 0, last;

  const snapshot = action => {
    const b = game.lander, [l0, l1] = game.legs;
    const bp = b.getPosition(), bv = b.getLinearVelocity(), p0 = l0.getPosition(), p1 = l1.getPosition();
    const values = [bp.x, bp.y, b.getAngle(), bv.x, bv.y, p0.x, p0.y, l0.getAngle(), p1.x, p1.y, l1.getAngle()];
    keys.forEach((k, i) => rec[k].push(r3(values[i])));
    rec.act.push(action);
    rec.wind.push(r2(game.currentWind()));
    rec.fuel.push(r1(fuel));
    rec.legs.push(game.legsTouching());
  };

  snapshot(0);
  for (;;) {
    const v = game.lander.getLinearVelocity();
    const before = Math.hypot(v.x, v.y);
    const action = pilot.act(random);
    last = game.step(action);
    pilot.observe(last.obs);
    fuel += FUEL_COST[action];
    score += last.reward;
    if (contactSpeed === null && (game.legsTouching() || game.gameOver)) contactSpeed = before;
    snapshot(action);
    if (last.terminated || last.truncated) break;
  }

  return {
    pilot: pilotId, outcome: game.outcome(last.obs), seconds: (rec.x.length - 1) / 50,
    fuel: r1(fuel), contact_speed: contactSpeed === null ? null : r2(contactSpeed), score: r1(score),
    terrain: game.terrain.map(([x, y]) => [r3(x), r3(y)]),
    pad: [r3(game.helipadX1), r3(game.helipadX2), r3(game.helipadY)], frames: rec,
  };
}
