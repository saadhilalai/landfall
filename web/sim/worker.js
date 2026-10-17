/* worker.js — the flight computer. Flies landings in a background thread, so the page stays
   smooth while it works. Gets {id, pilots, sliders, landing}; sends back the flight recordings. */

import { loadPilots } from "./policy.js";
import { recordFlight } from "./lander.js";

const pilots = loadPilots(new URL("../data/pilots.json", import.meta.url));

onmessage = async ({ data: { id, pilots: ids, sliders, landing } }) => {
  try {
    const loaded = await pilots;
    const flights = ids.map(p => recordFlight(p, loaded[p], sliders, landing));
    postMessage({ id, result: { landing, sliders, flights } });
  } catch (err) {
    postMessage({ id, error: String(err) });
  }
};
