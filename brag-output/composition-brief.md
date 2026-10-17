# Hyperframes Composition Brief: LANDFALL

## Objective
Create a short launch-style brag video for LANDFALL, the live rocket-landing AI website (an S4D project by Saad Hilal).

## Output
- Composition directory: `brag-output/composition/`
- Rendered video: `brag-output/brag.mp4`
- Format: landscape — 1920x1080
- Duration: 23 seconds

## Source Material
- Project root: `rocket-lander/` (this repository)
- Primary files read: `web/index.html`, `web/style.css`, `web/app.js`, `web/data/site.json`, `PROGRESS.md`, `runs/final/results.json`
- Real flight recordings (from the project's own simulator via `POST /api/fly`):
  - untrained pilot, normal conditions, landing 10001: crashed after 1.94 s, impact 12.37 m/s
  - Final AI, strong-wind test conditions (gravity 10, wind 16.25, turbulence 1.625), official test landing 10000: landed, 3.78 s, 88.9 fuel, 1.55 m/s
- Learning curve: Final AI (`randomized_memory_tuned`) mean training score from `site.json`
- Product name: LANDFALL
- Tagline / strongest claim: 73% safe landings in strong wind vs 33% for the baseline, tested once on 300 unseen landings per condition
- Key UI moments to recreate: the webcast telemetry bar, the wind slider with its trained / beyond-training zones, the glass result cards (VEHICLE LOST, TOUCHDOWN CONFIRMED), the S4D loading screen
- Copy that must appear verbatim:
  - VEHICLE LOST
  - TOUCHDOWN CONFIRMED
  - Beyond training
  - AN S4D PROJECT
  - LANDFALL
  - Designed & built by Saad Hilal

## Creative Direction
- Tone preset: cinematic
- Creative direction: mission-control launch trailer; quiet, confident, real telemetry instead of claims
- Interpretation: few big words, long enough holds, drama from real motion
- Angle: From flight 1 to the storm — first flight crashes, it practises 2 million steps, then lands a real test landing in a storm beyond its training, then the real numbers.
- Hook: an untrained lander tumbles and crashes under a webcast HUD; VEHICLE LOST
- Outro / punchline: LANDFALL slams in; "A rocket lander that taught itself."
- Avoid:
  - Generic SaaS language
  - Abstract filler visuals
  - Unrelated visual redesign (keep the site's black / white / D-DIN / glass look)
  - Claiming "2 million flights" (it is 2 million practice steps)

## Visual Identity
- Background: #000000 with a slow starfield
- Text: #ffffff; secondary #a3a3a3; tertiary #6b6b6b
- Accent: none decorative; zone colours only inside the slider UI: #29c7ac, #f5b83d, #ff5f57
- Display font: D-DIN Exp Bold (local woff2 from `web/fonts/`)
- Body font: D-DIN (local woff2)
- Visual references from the project: liquid-glass panels with a thin bright edge, thin white terrain line, white lander, white flame with glow, rolling flight-computer digits, wide uppercase tracked headings

## Storyboard
Use the storyboard in `brag-output/brag-plan.md` as the creative contract.

Scene summary:
1. First flight — 3.8s — untrained crash replay, VEHICLE LOST card
2. It taught itself — 3.8s — real learning curve draws, 0 → 2,000,000 practice steps
3. The storm — 6.8s — wind slider beyond training, LAUNCH on 8.74s, real storm landing, TOUCHDOWN CONFIRMED card near 13.11s
4. The proof — 4.1s — 73% vs baseline 33%, tested once on 300 unseen landings
5. LANDFALL — 4.5s — S4D logo, LANDFALL slam on 18.56s, tagline, credit

## Audio
- Audio role: cinematic support
- Audio arc: low bed → crash stop → build → launch lift → landing → logo ring-out → fade
- Music: happy-beats-business-moves-vol-12-by-ende-dot-app.mp3
- Music treatment: ~0.32 bed, fade out over the last ~1.2 s
- Music cue guidance: bundled preset `happy-beats-business-moves-vol-12-by-ende-dot-app.music-cues.json`; strong cues 8.74 / 13.11 / 18.56 s; beat grid for sequential reveals
- Audio-reactive treatment: subtle; bass/RMS lifts starfield brightness and HUD glass glow
- Audio-coupled moments:
  - crash impact — hard hit + flash
  - countdown — light tick, LAUNCH accent
  - touchdown — soft thud, success accent on the card
  - LANDFALL slam — deep bell
- SFX selection guidance: sparse and motion-matched; prefer low high-frequency-risk files from `sfx-analysis.md`
- Exact SFX choice: chosen during composition to match the implemented animation
- Audio files: copied into `brag-output/composition/assets/`

## Hyperframes Instructions
Built with the Hyperframes domain skills (core, animation, creative, keyframes, cli). Flights are drawn on a canvas as a pure function of timeline time (seek-safe): positions come from the recordings at 50 snapshots per simulated second; particles, shake and wind streaks use seeded deterministic maths. `hyperframes check` is the gate before render.
