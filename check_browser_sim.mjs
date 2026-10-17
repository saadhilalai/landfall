// check_browser_sim.mjs — flies landings with the browser simulator (web/sim) in Node.
// Reads a list of jobs as JSON on stdin: [{pilot, sliders: {gravity, wind, turbulence, engine}, landing}, ...]
// and prints one summary per job. Used by check_browser_sim.py.
import { readFileSync } from "node:fs";
import { Pilot } from "./web/sim/policy.js";
import { recordFlight } from "./web/sim/lander.js";

const specs = JSON.parse(readFileSync(new URL("./web/data/pilots.json", import.meta.url), "utf8")).pilots;
const pilots = Object.fromEntries(Object.entries(specs).map(([id, s]) => [id, new Pilot(s)]));
const jobs = JSON.parse(readFileSync(0, "utf8"));
const t0 = performance.now();
const out = jobs.map(({ pilot, sliders, landing }) => {
  const f = recordFlight(pilot, pilots[pilot], sliders, landing);
  return { outcome: f.outcome, seconds: f.seconds, fuel: f.fuel, score: f.score, x: f.frames.x, y: f.frames.y };
});
console.log(JSON.stringify({ ms_per_flight: (performance.now() - t0) / jobs.length, flights: out }));
