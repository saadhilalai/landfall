# Brag Plan: LANDFALL

## What is this app?
LANDFALL is a live website where a self-taught AI lands a rocket in gravity, wind, turbulence and engine power you choose, with real reinforcement-learning results behind it (an S4D project by Saad Hilal).

## The angle
**From flight 1 to the storm.** The video is a 23-second mission-control trailer of one AI's whole arc: its very first flight (an untrained network that tumbles and crashes), the 2 million practice steps where it taught itself, then a real test landing in a strong-wind storm it never practised for, ending on the real numbers. Every flight in the video is a frame-accurate replay of a recording from the project's own simulator (`/api/fly`), drawn in the site's own mission-control style. Nothing is mocked: the crash is untrained landing 10001, the storm is official test landing 10000, and the learning curve is the Final AI's actual training curve.

## Hook (first 2-3 seconds)
Black, stars, a webcast HUD. Small label "FLIGHT 1". The untrained lander tumbles out of the sky, the telemetry spins (tilt past 120°), it hits the ground at 12.4 m/s: screen shake, white flash, glass card "VEHICLE LOST". The viewer has watched an AI fail before they know what it is.

## Key moments (the middle)
- **The learning curve draws itself**: the Final AI's real score climbs from about −600 to 280 while a flight-computer counter rolls "0 → 2,000,000 practice steps"; the line punches through the dashed "SOLVED 200" mark.
- **The storm**: mission-control slider shows WIND 16.3 in the amber "BEYOND TRAINING" zone, LAUNCH, and the Final AI flies official test landing 10000 through gusts (wind streaks, flame, exhaust, rolling telemetry), touches down, "TOUCHDOWN CONFIRMED · 3.8 s · 89 fuel · 1.6 m/s".
- **The proof**: "73%" counts up huge, "safe landings in strong wind", against "Baseline: 33%". "Tested once. 300 unseen landings."

## Outro / punchline
S4D logo, "AN S4D PROJECT", then LANDFALL slams in wide-tracked on the strongest beat. Tagline: "A rocket lander that taught itself." Small: "Designed & built by Saad Hilal". Music fades under the logo.

## User flow worth showing
1. Set the conditions: the wind slider sits beyond the AI's training range (amber zone, "Beyond training").
2. Launch: the flight plays live with webcast telemetry (T+ clock, gauges, rolling digits, wind arrow).
3. Result: the glass "TOUCHDOWN CONFIRMED" card with flight time, fuel used and landing speed.

## Tone
- Preset: cinematic
- Creative direction: mission-control launch trailer; quiet, confident, real telemetry instead of claims
- Interpretation: few big words, generous holds, dramatic reveals carried by real motion (the crash, the curve, the landing) rather than hype copy; restrained sound with three beat-locked payoffs.

## Format: landscape — 1920x1080
## Duration: 23 seconds

## Visual identity (from the project)
- Background: #000000 (pure black, with a slow starfield)
- Accent: none decorative; physics-zone colours only where the UI uses them: trained #29c7ac, beyond training #f5b83d, impossible #ff5f57
- Text: #ffffff, secondary #a3a3a3, tertiary #6b6b6b
- Display font: D-DIN Exp Bold (wide, uppercase, tracked)
- Body font: D-DIN
- Strongest visual element: the live landing with the glass webcast telemetry bar and the glass result card

## Share copy (draft)
Its first flight crashed. Two million practice steps later it lands in storms it never trained for: 73% safe landings in strong wind, vs 33% for the baseline. LANDFALL, an S4D project.

## Audio direction
- Role: cinematic support
- Music: happy-beats-business-moves-vol-12-by-ende-dot-app.mp3 (steady, clean, 110 BPM)
- Music treatment: starts at 0 at a low bed (~0.32), slightly lower under the crash, full under the storm and numbers, fades out over the last ~1.2 s under the logo
- Music cue guidance: preset `assets/music/cues/happy-beats-business-moves-vol-12-by-ende-dot-app.music-cues.json` (109.96 BPM). Strong cues to target: 8.74s (LAUNCH of the storm flight), 13.11s (TOUCHDOWN CONFIRMED card), 18.56s (LANDFALL wordmark slam). Beat-grid windows: 3.82-7.09s for the learning-curve labels (every other beat), 14.20-16.93s for the 73% / 33% reveal (every other beat).
- Audio-reactive treatment: subtle; music bass/RMS gently lifts the starfield brightness and the glass HUD edge glow. No visualizer graphics.
- SFX posture: sparse, motion-matched, professional restraint
- Audio-coupled moments: crash impact + flash; countdown ticks before LAUNCH; touchdown thud + success accent on the card; counter rolling (light ticks only, not per digit); logo slam
- Restraint rule: no stacked hits, no SFX on every reveal, nothing louder than the crash and the logo

## Storyboard

### Scene 1 — First flight — 3.8s (0.0–3.8)
Starfield, terrain line, landing pad, webcast telemetry bar. Top-left label "FLIGHT 1" with "Untrained network" under it. The untrained lander (real recording, landing 10001) falls and tumbles; telemetry digits roll (tilt swings past 100°). Impact at ~2.4s: shake + flash + sparks, then glass card "VEHICLE LOST" — "Hull hit the ground at 12.4 m/s." holds to the end.
Sequential/interaction: yes — the flight itself (simulated live replay), then the result card.
Audio intent: tension, then a hard stop.
Audio-coupled idea: crash impact on the hit; card lands with a soft drop.
Music: low bed
Transition mood: hard cut → Scene 2

### Scene 2 — It taught itself — 3.8s (3.8–7.6)
Big line "It taught itself." (holds ≥1.2s). A thin white learning curve (the Final AI's real mean training score) draws left to right across the frame with its best-to-worst band; dashed "SOLVED 200" line; a rolling counter "0 → 2,000,000 practice steps"; the end dot lands at "280".
Sequential/interaction: yes — curve draws, counter rolls, end value lands.
Audio intent: momentum building.
Audio-coupled idea: light counter ticks; a small accent when the line crosses 200.
Transition mood: clean wipe → Scene 3

### Scene 3 — The storm — 6.8s (7.6–14.4)
Mission-control moment: a glass slider card "WIND 16.3" with the zone line (teal 0–15 "trained", amber beyond) and the label "Beyond training"; short countdown T−1 → "LAUNCH" landing on 8.74s. The Final AI flies official test landing 10000 through gusts: wind streaks, wind arrow, flame and exhaust only when its engine fires, rolling telemetry. Touchdown dust and small shake; glass card "TOUCHDOWN CONFIRMED" with 3.8 s · 89 fuel · 1.6 m/s (count up) at ~13.0s, held ~1.4s.
Sequential/interaction: yes — set condition, launch, flight, result.
Audio intent: lift on launch, controlled flight, satisfying landing.
Audio-coupled idea: countdown tick, launch accent on 8.74, touchdown thud, success accent on the card (13.11).
Transition mood: soft crossfade → Scene 4

### Scene 4 — The proof — 4.1s (14.4–18.5)
Huge "73%" counts up (left-anchored), label "Safe landings in strong wind". Right column "Baseline 33%". Bottom line: "Tested once. 300 unseen landings per condition." All copy held together for ≥1.8s.
Sequential/interaction: yes — 73% counts, baseline arrives on the next-but-one beat, footnote last.
Audio intent: confident.
Audio-coupled idea: one soft accent when 73% settles.
Transition mood: dramatic → Scene 5

### Scene 5 — LANDFALL — 4.5s (18.5–23.0)
S4D logo and "AN S4D PROJECT", then "LANDFALL" slams in wide-tracked on 18.56s. Tagline "A rocket lander that taught itself." Small credit "Designed & built by Saad Hilal". Music fades out; final hold.
Sequential/interaction: logo → title → tagline → credit.
Audio intent: arrival, then quiet.
Audio-coupled idea: deep bell on the LANDFALL slam, then let the music fade.

**Music mood for this video:** cinematic
**Audio summary:** a low steady bed that gives way to a hard crash, builds through the learning curve, lifts on the storm launch, lands with a thud and a success accent, and rings out under the LANDFALL logo.
