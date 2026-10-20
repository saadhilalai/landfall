# Rocket Lander: Progress Notes

_Last updated 2026-10-16. Nothing is running. All models, results and logs are saved and committed._

| Step | Status |
|---|---|
| 1. Setup | ✅ Done |
| 2. Baseline agent | ✅ Done (redone with 3 seeds) |
| 3. Custom reward shaping | ✅ Done (v1, then a tuning round to v2) |
| 4. Domain randomization | ✅ Done. Final agent design: randomized + memory |
| 5. Optuna auto-tuning | ✅ Done. **Final agent: `randomized_memory_tuned`** |
| 6. Final results, charts, videos | ✅ Done (evaluated once, plan committed first) |
| 7. Live demo website (LANDFALL) | ✅ Live at https://s4dai-landfall.static.hf.space (free static Hugging Face Space) |
| 8. README, packaging, resume bullets | ✅ Done (README, demo GIF, clean-install check, resume bullets) |

---

## How to pick up again

```bash
cd ~/rocket-lander
.venv/bin/tensorboard --logdir runs/tensorboard     # graphs, in a separate terminal → http://localhost:6006
```

On a fresh clone (no `.venv`):
```bash
brew install python@3.11 swig ffmpeg
python3.11 -m venv .venv && .venv/bin/pip install -r requirements.txt
```

Useful commands:
```bash
.venv/bin/python app.py                                     # the LANDFALL website → http://127.0.0.1:7860
.venv/bin/python make_space.py --push s4dAi/landfall       # upload the site to Hugging Face (after `hf auth login`)
.venv/bin/python export_browser_pilots.py                  # re-export the networks the site flies (after retraining)
.venv/bin/python check_browser_sim.py                      # browser physics vs Python on 1,500 test landings (~2 min)
.venv/bin/python train.py --agent shaped_v2                 # train one agent type (3 seeds, ~5 min)
.venv/bin/python compare.py --agents baseline shaped_v2     # compare agents on 1,000 landings per seed
.venv/bin/python compare.py --agents baseline shaped_v2 randomized --reference baseline shaped_v2 \
    --scenarios normal heavy_gravity weak_engine strong_wind harsh --episodes 500   # stress test (~3 min)
```

## Files

| File | What it does |
|---|---|
| `config.py` | **Every** shared setting: PPO settings, seeds, training budget, randomization ranges, test scenarios |
| `envs.py` | Builds each agent's world. `AGENTS` lists the only differences between agent types |
| `landing_stats.py` | Judges a landing by asking the physics engine directly (works around the leg-sensor bug) |
| `reward_shaping.py` | The custom reward, with versions v1 and v2 |
| `domain_randomization.py` | Changes gravity, wind, turbulence and engine power (fixed values or random ranges) |
| `train.py` | Trains an agent type, once per seed. Logs to TensorBoard and writes `runs/<agent>/seed<N>/run_info.json` |
| `evaluate.py` | Flies a model through a fixed list of landings |
| `tune.py` | Step 5: Optuna search (`search`), its summary and charts (`report`), and the tuned-vs-current check (`confirm`) |
| `final_eval.py` | Step 6: the ONE-TIME final evaluation on the test landings (refuses to run twice) |
| `make_charts.py` | Draws `charts/*.png` from saved results (safe to re-run) |
| `make_videos.py` | Renders `videos/*.mp4` at 1920×1280, replaying recorded test landings |
| `compare.py` | Compares agents on identical landings, with a paired significance test (`--seeds`, `--scenarios`, `--reference`) |
| `app.py` | Step 7: the website's server. Flies each landing with the real simulator and sends back the flight recording |
| `web/` | Step 7: the website itself (`index.html`, `style.css`, `app.js`, D-DIN fonts, `data/site.json`) |
| `export_site_data.py` | Writes `web/data/site.json`: pilots, slider ranges, measured physics, Step 6 results, learning curves |
| `make_space.py` | Packs the site into `build/hf_space/` (a free Docker Space) and optionally uploads it |
| `render.py` | Sharp redraw of the game at any size (used by `make_videos.py`) and the wind readout |
| `screenshots/` | Step 7 screenshots of the site at desktop (1440×900) and phone (390×844) size |
| `check_setup.py` | Step 1 simulator check |
| `models/<agent>_seed<N>.zip` | 12 trained models (~150 KB each) |
| `runs/` | Per-run info, learning-curve data (`evaluations.npz`), episode logs, TensorBoard, training logs, comparison JSONs |

---

## Rules that keep the comparison fair

- **The same PPO settings for every agent** (RL Baselines3 Zoo's tuned LunarLander settings, in `config.py`): 16 simulators in parallel, 1 CPU thread. Runs are **bit-for-bit reproducible** (verified by retraining from scratch).
- **3 training seeds per agent: 42, 142, 242.** The same seed gives every agent type the same starting network and the same practice worlds, so comparisons are paired.
- **Training budget rule:** train 1M steps, then a 100-landing validation test. If the score is below 200, keep training in 250k chunks, up to 2M.
- **Separate sets of landings:**
  - validation, seeds 5000–5099: used only for the "is it solved?" check
  - analysis, seeds 20000+: used for comparisons while building
  - **test, seeds 10000–10099: still untouched, reserved for Step 6**
- **Grading is always on the official game score** plus the **"landed safely" rate**. "Landed safely" means the physics engine says the lander came to a complete rest, between the flags.

---

## Step 1: Setup
- Homebrew Python 3.11.17, with `.venv` in this folder. All packages installed and Box2D built without problems. Training runs on the CPU (Apple M2).
- Fastest setup: one process per simulator, PyTorch on 1 thread, about 13,500 steps/second. **1M training steps take about 1.5 minutes.**

## Step 2: Baseline (official reward)
- Every seed passed 200 at the 1M check. Validation scores: 260.5 / 262.1 / 235.9, so **252.8 ± 12.0** on average.
- Learning curve (seed 42): −148 at 33k steps → 152 at 360k → 255 at 688k → 270 at 1M.

### The leg-sensor bug (tell this story accurately in the README)
1. After Step 2, **7 of the 100 validation flights looked like hovering**: the episode ended with only one leg reading "touching". I first reported them as "balancing on one leg, firing engines until time ran out". **That was wrong.**
2. Digging in showed that **6 of the 7 had actually come to rest** in under 320 steps. The episode had ended because the physics engine put the lander to sleep. Only the leg flag disagreed.
3. **The cause is a bug in LunarLander.** The ground is made of separate flat segments. When a leg slides from one segment onto the next, the game switches its "touching" flag OFF, even though the leg is still touching the new segment. I confirmed this by asking Box2D directly: every "missing" leg was physically on the ground, right next to a segment joint (x = 10 or 12).
4. **Only 1 of the 7 (landing 5018) really never settled.** It rested on both legs, but the faulty sensor told the agent one leg was up, so it kept firing a side thruster to "fix" it until time ran out. A stricter check found one more (landing 5020), fidgeting on two legs, which the old rule had counted as a landing.
5. **The fix:** judge outcomes, and compute leg contact for the reward, **from the physics engine directly**. The agent's own inputs are unchanged, so every agent still faces the same game.
6. **The real "never settles" rate for the baseline is about 0.4%** (11 of 3,000 analysis landings), not 7%.

## Step 3: Custom reward (`reward_shaping.py`)

These rules are added **on top of** the official reward. Every agent is still graded on the official score.

| Rule | When | v1 | v2 |
|---|---|---|---|
| Tilt | every step while flying | −1.0 × lean (radians) | same |
| Hover | in the air, below height 0.2, descending slower than 0.05 | −0.1 / step | same |
| Not settling | on one leg, or firing engines with both legs down | −0.1 / step | same |
| Hard landing | each touchdown | −200 × (speed − **0.15**), capped at −30 | safe speed **0.20** |
| Settle bonus | came to rest on the pad | +100 × fraction of time left | same |
| Fuel bonus | came to rest on the pad | **+0.25** per fuel unit under 200 | **+0.5** |

Two design fixes came out of testing the rules **before** training:
- **Cap on the hard-landing penalty.** A random pilot lost −259 per crash, which made the ground scarier than flying off-screen (−100). The AI could have learned to avoid landing at all.
- **Tilt only counts while flying.** A lander parked crooked on a slope piled up −176, enough to make crashing on purpose look attractive.

Results: 1,000 analysis landings × 3 seeds each, official score (`runs/compare_3seeds.json`)

| | Baseline | Shaped v1 | **Shaped v2** |
|---|---|---|---|
| Official score (± spread across seeds) | 257.1 ± 14.4 | 254.0 ± 3.9 | **267.2 ± 0.8** |
| Landed safely | 99.1% | 99.6% | **99.7%** |
| Crashed (of 3,000) | 9 | 1 | **0** |
| Never settled (of 3,000) | 11 | 3 | **1** |
| Fuel per landing (seeds 42 / 142 / 242) | 165.9 (135 / 133 / 230) | 181.2 (174 / 196 / 174) ❌ | **136.5** (135 / 139 / 135) |
| Average landing time | 6.5 s | 6.9 s | **5.6 s** |
| Failures vs baseline on the same landings | n/a | 26 vs 13, p = 0.053 | **26 vs 8, p = 0.003** |
| Baseline's 11 never-settled landings | n/a | landed 11/11 | landed 11/11 |

**The v1 fuel finding.** v1 fixed hovering but **burned more fuel**, which is the opposite of what its fuel bonus was meant to do. I had set the safe touchdown speed at the baseline's median, so 59% of the baseline's touchdowns counted as "too fast". v1 overcorrected: it touched down at **half the speed** (0.080 vs 0.155), and its final approach took **92 steps instead of 53**. It burned fuel at the same rate per second, so the longer approach meant more fuel. In short, reward rules compete with each other, and you find out how by measuring behavior, not points.

**v2 (one tuning round):** safe speed 0.20, fuel bonus doubled. Fuel came back down to about 136 on every seed, landings got faster, and v2 scored best. To be honest about the fuel number: v2 matches the baseline's two typical seeds (133–135). The average advantage comes from avoiding the baseline's 230-fuel outlier seed. So v2 is more **consistent**, not more efficient than the best baseline.

### Decisions made
1. 3 seeds per agent, with the baseline retrained on 3 seeds.
2. A stricter "landed" rule: the lander must actually come to rest.
3. One tuning round from v1 to v2. **v1 is kept** (`models/shaped_v1_*`, `runs/shaped_v1/`) for the README's iteration story. The refactored code reproduces v1's rewards exactly (0 differences over 8,716 steps).
4. **The 5-seed rule:** add 2 seeds per agent if the v2 vs baseline result is borderline (p ≥ 0.01). It came out at p = 0.003, so we **stayed at 3 seeds**.
5. Step 4 uses the **v2** reward.

(`runs/compare_v1.json` is the original v1 comparison, from when the agent was still called "shaped". `runs/compare_3seeds.json` reproduces the same numbers.)

---

## Step 4: Domain randomization

The randomized agents use the v2 reward, and every practice episode re-rolls the physics. Every test value lies outside the training range.

| | Gravity | Wind | Turbulence | Engine | Engine ÷ gravity |
|---|---|---|---|---|---|
| Normal game | −10 | off | off | 1.0× | 1.80 |
| **Training range** (re-rolled every episode) | −11 to −8 | 0–15 | 0–1.5 | 0.85–1.15× | ≥ 1.39 |
| heavy_gravity test | **−12** | off | off | 1.0× | 1.50 |
| weak_engine test | −10 | off | off | **0.8×** | 1.44 |
| strong_wind_original | −10 | **20** | **2.0** | 1.0× | 1.80 |
| harsh_original | **−12** | **20** | **2.0** | **0.8×** | 1.20 |
| strong_wind_extreme (softened once) | −10 | **17.5** | **1.75** | 1.0× | 1.80 |
| harsh_extreme (softened once) | **−11.5** | **17.5** | **1.75** | **0.825×** | 1.29 |
| **strong_wind** (main test, softened twice) | −10 | **16.25** | **1.625** | 1.0× | 1.80 |
| **harsh** (main test, softened twice) | **−11.25** | **16.25** | **1.625** | **0.8375×** | 1.34 |

### Round 1: randomized agent with the normal budget (stops once the score passes 200, about 1M steps)

Stress test: 500 analysis landings × 3 seeds (`runs/stress_test_3seeds.json`). Wind tests used the **original** values.

| Landed safely (official score) | Baseline | Shaped v2 | Randomized |
|---|---|---|---|
| normal | 99.1% (258) | **99.8% (268)** | 92.9% (224) |
| strong_wind_original | 18.3% (19) | 15.5% (17) | **20.7% (47)** |
| heavy_gravity | 97.1% (248) | **98.8% (261)** | 94.5% (220) |
| weak_engine | 92.9% (234) | **96.1% (253)** | 93.5% (217) |
| harsh_original | 0.2% (−231) | 0.5% (−193) | 0.0% (−152) |

**Round 1 headline: randomization did NOT help.** The randomized agent was worse in the normal game, about equal to the baseline elsewhere, and nobody could land in harsh. Suspected causes: under-training, no way to sense the conditions, and test scenarios that might not be landable at all.

### Round 2, part 1: landability check (decided BEFORE looking at results; rule in `config.py`)

The rule: a scenario counts as landable if a specialist (trained in it, with memory, for 2M steps) lands at least 50% of 500 times. If not, every changed value moves halfway back toward the edge of the training range, and the check runs once more.

| Specialist landed safely | Original | Softened once (now `*_extreme`) |
|---|---|---|
| strong_wind | 10.6% | 25.4% |
| harsh | 0.0% | 1.6% |

**Both fail, even after softening.** The rule allows only one softening round, so the next step is **your decision** (see below).

The landability check turned out to be flawed: specialists **underestimate** what's landable. The memory generalist below lands **40.6%** in strong_wind_extreme, which beats the specialist trained only there (25.4%). Learning to land from scratch in a gale is harder than learning to land first and then handling wind, which is the argument for curriculum learning.

### Round 2, part 2: memory, with the SAME 2M budget for every agent (`runs/stress_test_2m.json`)

These three agents differ in only one thing at a time:

| Agent | Randomized physics | Memory (last 4 moments) |
|---|---|---|
| `shaped_v2_2m` | no | no |
| `randomized_2m` | yes | no |
| `randomized_memory` | yes | yes |

Stress test: 500 landings × 3 seeds per cell.

| Landed safely (official score) | shaped_v2_2m | randomized_2m | randomized_memory |
|---|---|---|---|
| normal | **99.7% (277)** | 99.3% (257) | **99.7% (259)** |
| heavy_gravity | 85.5% (236) | **99.3% (251)** | 97.7% (249) |
| weak_engine | 69.7% (194) | **96.1% (242)** | 94.1% (239) |
| strong_wind_extreme | 27.3% (81) | 33.8% (104) | **40.6% (130)** |
| harsh_extreme | 1.4% (−112) | 1.3% (−108) | **2.7% (−80)** |
| **strong_wind** (main) | 34.2% (106) | 37.2% (121) | **46.8% (152)** |
| **harsh** (main) | 3.7% (−85) | 2.0% (−75) | **6.2% (−46)** |
| Fuel per landing (normal game) | **109** | 172 | 171 |

What the paired tests on the same landings show:
- **Randomization works once it gets enough practice.** randomized_2m vs shaped_v2_2m: heavy gravity 99.3% vs 85.5%, weak engine 96.1% vs 69.7%, strong wind 33.8% vs 27.3% (all p ≤ 0.0001). Normal game: no significant difference (p = 0.30).
- **Memory helps against wind, the thing it was meant to sense.** randomized_memory vs randomized_2m: strong wind **40.6% vs 33.8% (p < 0.0001)** and harsh 2.7% vs 1.3% (p = 0.002). It costs a little under steady changes: heavy gravity 97.7% vs 99.3% (p = 0.0001) and weak engine 94.1% vs 96.1% (p = 0.001). **Memory is a trade-off, not a free win.**
- **Surprise: extra practice in one world made an agent MORE fragile** (`runs/practice_effect.json`). shaped_v2 at 1M vs 2M, same landings: the normal game is unchanged (99.8% → 99.7%, p = 0.73), but heavy gravity drops from 98.8% to **85.5%** and weak engine from 96.1% to **69.7%** (both p < 0.0001). At 2M it lands faster (4.7 s vs 5.6 s) on less fuel (109 vs 136), likely by braking at the last moment with exactly enough thrust for normal gravity and no safety margin. That margin disappears with a weaker engine or stronger gravity. **This is the textbook case for domain randomization.**
- **Extra practice fixed the randomized agent** (1M → 2M, same landings): normal 92.9% → 99.3%, heavy gravity 94.5% → 99.3%, weak engine 93.5% → 96.1% (all p < 0.001). Round 1's main problem was under-training.
- **Strong wind is still the limit.** In the main strong_wind test the memory agent lands 46.8%, vs 37.2% without memory (p < 0.0001). The main harsh test is still mostly unlandable: 6.2% at best. (Main wind tests: `runs/stress_test_2m_main_wind.json`.)

---

## Decisions made after Step 4 (2026-10-15 evening)

1. **Wind tests: option C.** The main `strong_wind` and `harsh` tests apply the halfway rule a second time. The once-softened versions stay as `*_extreme` rows that show where the AI breaks. The first designs remain as `*_original`.
2. **Final agent: `randomized_memory`.** It's best in wind, where the memory gain is proven.
3. **Step 5 honesty rule:** tuning may only use the training conditions and the validation landings, never the stress scenarios or the final test landings.

---

## Step 5: Optuna auto-tuning (done)

**Honesty rules:**
- Every trial trained in the **training** conditions and was scored **only** on the 100 validation landings, flown in the training conditions.
- The stress scenarios and the final test landings (10000–10099) were never used.
- The confirmation rule was set before seeing any results.

**The search** (`tune.py search`, `runs/optuna/`):
- 28 trials in about 48 minutes, 3 at a time: 18 completed, 10 pruned early.
- Each trial trained a randomized_memory agent with seed 42 for 2M steps.
- Trial 0 used the current settings and scored **233.8** (88% landed). The best trial, number 23, scored **264.7** (95% landed).
- Most important knobs (`importance.png`): learning passes per batch `n_epochs` (0.36), round length `n_steps` (0.24), and the policy-update limit `clip_range` (0.14).

**Winning settings** (in `config.TUNED_PPO_KWARGS`):

| Setting | Before → after |
|---|---|
| n_epochs | 4 → 9 |
| n_steps | 1024 → 512 |
| batch_size | 64 → 128 |
| clip_range | 0.2 → 0.1 |
| ent_coef | 0.01 → 0.0004 |
| gamma | 0.999 → 0.995 |
| gae_lambda | 0.98 → 0.9 |
| learning_rate | 3e-4 → 4.75e-4 |

The network size didn't change.

**Confirmation** (`tune.py confirm`, `runs/optuna/confirm.json`): retrained with 3 seeds at 2M steps, scored on validation landings in the training conditions.

| | Score (± spread across seeds) | Landed | Seed 42 | Seed 142 | Seed 242 |
|---|---|---|---|---|---|
| randomized_memory (current settings) | 227.0 ± 7.3 | 84.0% | 233.8 | 216.9 | 230.2 |
| **randomized_memory_tuned** | **261.0 ± 6.6** | **93.3%** | 256.7 | 270.3 | 256.1 |

- **The tuned settings win on every seed**, so by the pre-set rule they are **adopted**.
- Seeds 142 and 242 were never used in the search, and they improve too (+53 and +26). So this isn't just luck on seed 42.
- A small "winner's curse" shows up: trial 23 scored 264.7, but the same settings retrained on seed 42 score 256.7. The best-of-28 was a little lucky, as expected.
- Side effect in the normal game (validation during training): **274–280 points with about 90–100 fuel**, compared with about 252–259 points and 161–184 fuel for the current settings.
- On purpose, the tuned agent has **not** been run on the stress scenarios. Its first contact with them will be the honest Step 6 evaluation.

## Step 6: Final evaluation, charts, videos (done)

**Run exactly once.** `final_eval.py` (the agents, seeds, scenarios, metrics, paired tests and video rules) was committed in `f0b3a58` (originally `6ca403d`, see below) **before** it ran, and it refuses to run a second time. Before that, it was dry-run on throwaway landings (seeds 25000+) in a single process, so the test landings stayed untouched.
- 8 agents × 3 seeds × 7 scenarios × 100 test landings (10000–10099) = **16,800 landings**, which took 1.7 minutes.
- Outputs: `runs/final/episodes.csv` (every landing), `results.json` and `results.md`.

**Landed safely** (300 test landings per cell):

| Agent | Normal | Heavy gravity | Weak engine | Strong wind | Harsh | Strong wind (extreme) | Harsh (extreme) |
|---|---|---|---|---|---|---|---|
| Baseline | 99.7% | 96.0% | 91.0% | 33.3% | 2.7% | 24.0% | 2.0% |
| Custom reward v1 | 100% | 99.3% | 95.7% | 37.3% | 1.7% | 29.0% | 1.3% |
| Custom reward v2, 1M | 99.7% | 97.0% | 89.0% | 32.3% | 2.3% | 29.3% | 1.3% |
| Randomized, 1M | 94.7% | 95.3% | 93.0% | 38.3% | 1.7% | 33.0% | 0.0% |
| Custom reward v2, 2M | 100% | 80.0% | 69.3% | 35.3% | 1.7% | 29.7% | 0.3% |
| Randomized, 2M | 99.7% | **99.0%** | **93.7%** | 40.0% | 2.3% | 33.7% | 1.0% |
| Randomized + memory | 100% | 95.7% | 87.7% | 45.0% | 5.3% | 38.7% | 3.7% |
| **Final (tuned)** | **100%** | 91.7% | 79.7% | **73.0%** | **13.7%** | **69.0%** | **7.0%** |

**What the paired tests on the same landings show:**
- **The final agent vs the baseline:**
  - Strong wind: **73.0% vs 33.3%** (p < 0.001).
  - Harsh: 13.7% vs 2.7% (p < 0.001).
  - Strong wind (extreme): 69.0% vs 24.0% (p < 0.001).
  - Best normal-game score: 282 vs 259.
  - **But it's worse** under heavy gravity (91.7% vs 96.0%, p = 0.03) and with a weak engine (79.7% vs 91.0%, p < 0.001).
- **More practice in one world made the agent more fragile (replicated on the test set).** Reward v2 at 2M vs 1M steps: heavy gravity 80.0% vs 97.0%, weak engine 69.3% vs 89.0% (both p < 0.001).
- **Randomization works at an equal budget.** Randomized vs not, both at 2M steps: heavy gravity 99.0% vs 80.0%, weak engine 93.7% vs 69.3% (p < 0.001). In wind the gap isn't significant (40.0% vs 35.3%, p = 0.28).
- **Memory, at an equal budget:**
  - Harsh: 5.3% vs 2.3% (p = 0.049). Harsh (extreme): 3.7% vs 1.0% (p = 0.008).
  - Strong wind: +5 points, but **not significant on 300 landings** (p = 0.21). On 1,500 analysis landings earlier it was significant.
  - Costs: heavy gravity 95.7% vs 99.0% (p = 0.002), weak engine 87.7% vs 93.7% (p < 0.001).
- **Tuning helped a lot in wind but cost robustness to steady changes.** Tuned vs untuned memory agent:
  - Strong wind: 73.0% vs 45.0% (p < 0.001). Harsh: 13.7% vs 5.3% (p < 0.001).
  - Heavy gravity: 91.7% vs 95.7% (p = 0.02). Weak engine: 79.7% vs 87.7% (p < 0.001).
- **The likely mechanism is efficiency vs safety margin** (consistent with the data, not proven).
  - The leanest flyers crash most when the engine is weaker than anything seen in training:

    | Agent | Normal-game landing | Normal-game fuel | Weak engine, crashed |
    |---|---|---|---|
    | Final (tuned) | 4.0 s | 96 | 19.7% |
    | Reward v2, 2M | 4.7 s | 110 | 27.0% |
    | Randomized, 2M | 6.6 s | 172 | 5.7% |

  - Tuning optimised the official score, which rewards saving fuel, so it found a fast, lean flyer. Spending less time in the air probably also helps in wind.

**Charts** (`charts/`, one fixed colour per agent, palette checked for colourblind safety):
1. Learning curves: small multiples, mean of 3 seeds with a band.
2. Landing success heatmap: every agent × every scenario.
3. More practice made it fragile: reward v2, 1M vs 2M steps.
4. Memory vs no memory in wind, with honest p-values.

**Videos** (`videos/`, 1920×1280, 50 fps (real time), H.264):
- The simulator only draws at 600×400, so `make_videos.py` redraws the same physics objects as sharp vector shapes at 3.2×.
- Every clip replays a recorded test landing and is **checked to match the evaluation's outcome and score exactly**.
- Landings were picked by the fixed rule: the seed-42 model, and the first matching test landing in seed order.
- The clips:
  - `untrained_crash.mp4`
  - `baseline_landing.mp4`
  - `final_strong_wind.mp4` (+ `_arrow` version with a live wind indicator)
  - `side_by_side_heavy_gravity.mp4` (3840×1280; landing 10001: no randomization crashed, final landed), plus each half on its own
  - `broll_1.mp4` to `broll_5.mp4`
- The side-by-side is an **illustrative example**: across all heavy-gravity test landings the rates are 80.0% vs 91.7%.

## Step 7: The live demo website, "LANDFALL" (built)

**First version (replaced):** a Gradio app that encoded each landing as an MP4. It worked, but the user judged the
design too generic. **Redesign brief:** inspired by launch webcasts (style only, no names or logos), black and white,
thin lines, sharp edges, wide uppercase D-DIN headings, huge numbers, mission-control feel, works on phones.
Mission name chosen by the user: **LANDFALL**.

**How it works:** `app.py` (FastAPI) flies the landing with the real simulator and the chosen pilot (~26 ms) and
returns the recording: position, legs, speed, tilt, engine state, wind and fuel, 50 per simulated second. The page
(`web/`, plain HTML/CSS/JS, no framework) draws it on a full-screen canvas at 60 fps with a telemetry overlay.
No video encoding, so it is instant (a two-pilot request is ~70 ms, ~11 KB gzipped).

- **Faithful:** flights use exactly the Step 6 test setup. Checked: 45/45 replays (3 pilots × 5 scenarios × 3 test
  landings) match `runs/final/episodes.csv` (outcome, score, steps, fuel).
- **Pilots:** Final AI (`randomized_memory_tuned`), Careful (`randomized_2m`), Baseline, Untrained (seed-42 models).
- **Telemetry:** T+ clock, speed and altitude gauges, vertical/horizontal speed, tilt, fuel used, wind direction and
  strength, engine phase. Wind is also drawn as streaks moving with the gust.
- **Result screen:** TOUCHDOWN CONFIRMED / VEHICLE LOST (plus off-target and no-touchdown), with flight time, fuel
  used and landing speed (speed just before first ground contact). Compare mode: split screen, same landing.
- **Slider zones:** green = the range the AI trained in, amber = beyond training, red dotted = physically impossible.
  The red zones come from physics **measured in the simulator** (`export_site_data.py`): main engine 17.24 m/s² at
  100% (thrust-to-weight 1.72 at gravity 10), side thruster 2.48 m/s², side-thruster spin 4.97 rad/s² vs 1.10 per
  unit of turbulence. Red = thrust < weight, or gusts/turbulence stronger than the engines can counter.
  Sanity check (analysis seeds, not the test set): below thrust = weight both trained pilots landed 0/30; at
  thrust/weight 1.03 the careful pilot landed 3/30.
- **Landing check:** live thrust-to-weight, verdict, and the expected landing rate taken from the **nearest Step 6
  test condition** (labelled "measured", "closest test" or "untested" by distance; 0% when physics rules it out).
- **Colours:** only the three zone colours (teal #29c7ac, amber #f5b83d, red #ff5f57), checked with the dataviz
  colourblind validator; each zone also has its own line style and a text label. Charts are greyscale.
- **Sections:** mission story (3 parts), big results, the "where it is worse" trade-off, all four charts redrawn in
  the site style (HTML/SVG from `web/data/site.json`), and four flight-record clips (shown in black and white).
- **Shareable links:** the address bar always holds the current flight (`?pilot=…&vs=…&wind=…&landing=…`).
- **Tested** in Chrome (desktop 1440×900, phone 390×844 via device emulation): flights, compare, Surprise me,
  impossible conditions, crash screen, reduced-motion mode. Screenshots in `screenshots/`.
- **Hosting:** first built as a Docker Space; replaced by a static site (see "Deployed" below).

### Polish round (2026-10-16): liquid glass, extras, S4D branding

- **Branding:** S4D logo (cropped from the user's file into `web/brand/`) in the header, loading screen
  ("AN S4D PROJECT", then LANDFALL), favicon and footer ("Designed & built by Saad Hilal · S4D").
  Footer link slots: fill `LINKS` at the top of `web/app.js` (GitHub, LinkedIn); empty ones stay hidden.
- **Liquid glass** on the telemetry, mission-control panel, pilot picker, buttons and result cards: real
  background blur, a thin bright edge and a shine that follows the mouse. Mission control now floats over the
  live landing on desktop.
- **Extras:** T−3 / T−2 / T−1 / LAUNCH countdown (click, Space or Esc skips); telemetry digits roll and tick
  12.5 times a second; engine glow, flame and exhaust only on steps where the simulation fired that engine,
  with exhaust turning into dust when it hits the ground and engine light on the ground near touchdown; small
  shake on touchdown, bigger shake and flash on a crash; slow starfield behind everything (stars hidden behind
  the ground); sections fade and slide in; big numbers and result stats count up; optional synthesized sound
  (engine rumble, thruster hiss, countdown beeps, touchdown thud, crash), OFF by default.
- **Performance:** a "lite" mode (no real blur, fewer particles, one star layer, lower canvas resolution) is picked
  automatically on phones and low-core/low-memory devices, and switched on mid-flight if frames get slow.
  Measured with Chrome on this Mac (headless, GPU): desktop full effects 60 fps (median 16.7 ms, 90th percentile
  16.8 ms); phone size with the CPU slowed 4× in lite mode 60 fps. `?fx=full` / `?fx=lite` force a mode.
- **Reduced motion:** no autoplay on load, no shake or flash, static stars, no zoom on the countdown, no
  count-ups or rolling digits; flights still play when the visitor presses Launch.
- **Checked:** screenshots at desktop and phone size and two 60 fps screen recordings of a full launch
  (`screenshots/landfall_launch_desktop.mp4`, `landfall_launch_phone.mp4`). Bugs found and fixed: compare mode
  sized its telemetry for the whole pane instead of the visible part; the sound button waited for audio to
  start before updating; the fading loading screen swallowed clicks; panel content showed through the sticky
  Launch row.

### Deployed (2026-10-16): flights now run in the browser

- **Why:** Hugging Face now requires a paid PRO plan for Docker and Gradio Spaces ("402 Payment Required"); only
  static Spaces are free. So the site no longer needs a server: every landing is flown in the visitor's browser.
- **How:** `web/sim/` is LunarLander-v3 rewritten in JavaScript (`lander.js`) on planck.js 1.5.0 (a JavaScript
  Box2D, MIT, vendored in `web/vendor/`), NumPy's random generator rewritten bit-for-bit (`rng.js`: SeedSequence +
  PCG64, checked equal to NumPy on 10 seeds), and the pilots' networks (`policy.js`, weights exported as float32
  by `export_browser_pilots.py` into `web/data/pilots.json`, 116 KB). It runs in a background thread
  (`sim/worker.js`); two pilots take ~22 ms in Chrome. The untrained pilot's random button presses use their
  own seeded stream, so its flights differ from the Python ones (it crashes either way).
- **Checked against Python** (`check_browser_sim.py`, `runs/browser_check.json`): 3 trained pilots × 5 test
  scenarios × the 100 official test landings = 1,500 flights. Same terrain on every landing; **same outcome on
  99.2%**; landing rates differ by 0.6 points on average (at most 3); 82.1% of flights identical within 1 cm from
  start to finish (the rest drift apart after a median 4.6 s: planck.js uses 64-bit numbers, Box2D 32-bit).
  The first flight on the site (Final AI, strong wind, landing 10000) is identical: 3.78 s, 88.9 fuel, 1.55 m/s.
- **Live:** https://s4dai-landfall.static.hf.space (Space page: https://huggingface.co/spaces/s4dAi/landfall).
  Tested in headless Chrome on the live site: first flight, Surprise me, compare mode, all four pilots, the four
  video clips, the footer, phone size with the CPU slowed 4×, and inside the Space page's frame.
- `app.py` still runs the site locally and keeps the Python flight at `/api/fly` as the reference.

## Step 8: README and packaging (done, 2026-10-17)

- **README.md:** live demo and video links at the top, demo GIF (`docs/landfall_demo.gif`, recorded from the live
  site), results at a glance, how PPO and reward shaping work in plain words, the v1 → v2 story, the leg-sensor
  bug, domain randomization (including the failed first round and the landability rule), honest tuning, the
  one-shot test with the full results table (landed, score, fuel), the four charts, the surprises (more practice
  made it fragile, efficiency vs margin, memory trade-off, specialists underestimate), the website, videos, how to
  run, file layout, credits.
- **requirements.txt:** only what the code imports (removed unused `pandas` and `moviepy`), with comments.
- **Clean-install check:** cloned the repo into an empty folder, made a fresh Python 3.11 venv from
  `requirements.txt`, and ran `check_setup.py`, `evaluate.py` (Final AI, 20 test landings: all landed), a short
  training run, `make_charts.py` (byte-identical charts), the Python flight, `export_browser_pilots.py`
  (identical output) and `make_space.py`. All worked.
- **Website fix found while recording the GIF:** at 1280-wide laptop screens the T+ clock wrapped onto two lines.
  The page now switches to the compact telemetry bar whenever the full one doesn't fit. Checked at 1024, 1280,
  1366, 1440, 1536 and 1920 wide, and in compare mode.

### Commit IDs changed (2026-10-17)

Before the first push, every commit's author was switched to the user's GitHub identity: the name "Saad Hilal"
with GitHub's private no-reply email (`259462928+saadhilalai@users.noreply.github.com`), so a personal email
never appears on GitHub. Dates, messages and files are unchanged, but every commit ID changed. Files written
earlier (like `runs/final/results.json`) still name the original IDs.

On 2026-10-19 the history was rewritten twice more: once to remove commit message trailers and tidy the
README, and once for privacy (see "Privacy check" at the end). Each rewrite changed every ID again.
"Current" is the ID today:

| Original | Current | Commit |
|---|---|---|
| `4fcf3a6` | `804d935` | Setup, baseline, reward shaping v1/v2, domain randomization |
| `fdab33e` | `841c06c` | Step 4 round 2 |
| `bc27617` | `584d565` | Step 5 |
| `6ca403d` | `f0b3a58` | Pre-register the one-time final evaluation |
| `42527bb` | `a2eed8e` | Step 6 |
| `734ce03` | `1698ab1` | Step 7 |
| `3a5f4eb` | `ff6c4a3` | Step 7 polish |
| `cf4c242` | `36e595b` | Launch video |
| `86f87b2` | `5605a98` | Deploy (browser simulator) |
| `3892e7b` | `b2645c0` | Step 8 |
| `b2ec844` | `7ab6f30` | make_space fix |

## Done before the first push (2026-10-17)

- Commit author on every commit: "Saad Hilal" with GitHub's no-reply email (see the ID table above).
- `LICENSE`: MIT, © 2026 Saad Hilal. Third-party files keep their own licenses (listed in the README).
- Trailer music checked: "Happy Beats & Business Moves Vol. 12" by Sascha Ende (ende.app), CC BY 4.0: free for
  public and commercial use with credit. Kept, and credited in the README, `brag-output/CREDITS.md` and
  `brag-output/share-copy.txt`. Sound effects: Kenney, CC0.
- Site footer: GitHub link (https://github.com/saadhilalai). LinkedIn left out for now. Redeployed and checked live.
- README: "How this was built" section.

## Still open (needs the user)

1. ~~Create the empty GitHub repository and send the link, then push.~~ Done 2026-10-17: https://github.com/saadhilalai/landfall (public; `main` pushed, Hard Mode brought over without its training logs, GitHub login via `gh auth login`).

---

# Hard Mode (branch `hard-mode`, started overnight 2026-10-17)

> **On `main`:** the Hard Mode code, every trained brain (`models_hard/`), the chart, the videos and the one-time
> hidden test results (`runs_hard/final_test/`, `runs_hard/report.md`) were brought over on 2026-10-17. The 64 MB of
> training logs (validation checks, practice logs, TensorBoard) stay on the local `hard-mode` branch, so paths below
> like `runs_hard/attempt…/` or `runs_hard/level…/` only exist there.

**Goal:** keep training and make the AI better at harder challenges with **curriculum learning**: master one level
before moving to the next, carrying the same brain forward. Nothing on `main`, the final results, the README or the
live website changes; this branch is for the user to review.

**Sharing the Mac:** the walking-robot project trains at the same time and comes first. Hard Mode uses at most 2 CPU
cores (one training process per core, PyTorch on 1 thread, the 16 practice landers stepped one after another) and
runs at the lowest priority (`nice -n 19`).

## Morning summary (read this first)

**How far it got.** Levels 1 (fuel tank) and 2 (moving pad) were mastered by all 3 seeds. Level 3 (rocks and
walls) was mastered by 2 of 3 seeds, but only after teaching it in stages (short walls first). Level 4 (the
two-pad mission) was not mastered by anyone. The best mission brains complete 76–79% of missions without
obstacles; that is "a fair try fell short", so by the rules the curriculum stops there and the best brains are
saved. Seed 42 stopped one stage earlier, at the 3.5 m walls (89.0%, one point short).

**Hidden test, run once** (300 landings per cell that no brain had ever flown, seeds 40000–40299, random weather and
engine power on every landing; `runs_hard/final_test/results.json`, `runs_hard/report.md`). Landed safely:

| Level | The Final AI as it was (seeds 42 / 142 / 242) | After the curriculum (seeds 42 / 142 / 242) |
|---|---|---|
| 1 · Fuel tank (110) | 79.0% / 90.3% / 74.0% (avg **81.1%**) | 88.3% / 90.3%¹ / 91.0% (avg **89.9%**) |
| 2 · Pad in a random spot | 76.7% / 86.7% / 68.7% (avg **77.3%**) | 87.0% / 87.7% / 86.3% (avg **87.0%**) |
| 3 · Rocks and walls | 69.0% / 72.7% / 58.3% (avg **66.7%**) | – / 82.7% / 85.7% (avg of 2: **84.2%**)² |
| 4a · Mission, no obstacles | 1.3% / 21.3% / 29.7% | – / 79.3% / 76.0% (not mastered)³ |
| 4 · Mission with obstacles | 0.0% / 3.3% / 5.3% | – / 28.7% / 22.0% (never practised missions with obstacles) |

¹ Seed 142's Final AI already passed Level 1 on validation, so its Level 1 brain is the Final AI itself.
² Seed 42 stopped at the 3.5 m-wall stage. ³ In overtime (outside the rules, see below): 87.3% / 79.7%.

**The biggest wins.**
- Running out of fuel: the Final AI ran its 110 tank dry on 12–40% of Level 1 landings; the Level 1 brains on
  12–15%, the Level 3 brains on 3–5%.
- **No forgetting; the opposite.** Each new level made the brain *better* at the old ones. On Level 1: Level 1
  brains 89.9% on average, Level 2 brains 94.4%, Level 3 brains 96.3%. On Level 2: Level 2 brains 87.0%, Level 3
  brains 92.7%. Learning to fly around walls and to far pads made it a better pilot overall.
- Level 3: from 66.7% to 84.2% (the two seeds that got there). On the mission without obstacles: from ~17% to 76–79%.

**Honest caveats.**
- "Mastered" meant 90% on 200 validation landings. On the hidden test the mastered brains came out 1–8 points lower
  (86–91% on Levels 1–2, 83–86% on Level 3): picking the first checkpoint that crosses 90% on a noisy 200-landing
  check favours lucky checkpoints. A bigger validation set (or requiring two checks in a row) would fix that.
- Several fixes were made during the night, all on validation landings, with the hidden test untouched until the
  end: a lower learning rate for fine-tuning; obstacle sensors rebuilt as cones (the laser beams had a blind spot);
  practice with more walls; a mission reward loophole (sitting still) closed; take-off made learnable. Each one is
  explained in the log below, and every abandoned attempt is kept in `runs_hard/attempt*` and `models_hard/attempt*`.
- The mission brains were never trained with obstacles (they stopped at stage 4a), which is why they hit an
  obstacle on most Level 4 missions.

**What's here (branch `hard-mode`, folder `~/rocket-lander-hard-mode`).** Code: `hard_mode/`. Brains:
`models_hard/` (best + last per level and seed). Logs and every validation check: `runs_hard/`. Chart:
`charts/hard_mode_curriculum.png`. Videos: `videos/hard_mode/` (Levels 1–2 seed 42, Level 3 seed 142, and the
mission's best attempt, seed 142). Nothing on `main`, the README or the live site changed.

**If you want to go further,** the mission is the open problem: the overtime runs show it levels off around 80–87%
without obstacles. Next ideas: a bigger network for the mission, requiring 2 validation passes in a row, then
mission stages with obstacles (low walls first), the same way Level 3 was cracked.

## The levels (each keeps everything from the levels before it)

Every level keeps Step 4's random weather and engine power (`config.TRAIN_CONDITIONS`): gravity 8–11, wind 0–15,
turbulence 0–1.5, engine 85–115%, re-rolled every landing.

| Level | What's new | Time limit |
|---|---|---|
| 1 | **Fuel tank of 110 units.** Main engine 1 unit per step, side thruster 0.1. At empty, the engines stop. | 20 s |
| 2 | **The pad moves**: one of 7 spots across the ground, a new one every landing | 20 s |
| 3 | **Obstacles**: 1–2 rocks beside the pad, and a tall wall between the start and a far-away pad. Touching one = crash | 20 s |
| 4 | **Mission**: land on pad A and come to rest, take off, fly to pad B (at least 6 m away), land again. Tank 275 | 40 s |

**Why a 110 tank** (`hard_mode/calibrate.py`, 500 calibration landings, seeds 8000–8499): with an endless tank the
Final AI burns 101 fuel per safe landing on average (median 100, 90th percentile 120). With 110 it would have run
dry on 23% of its landings, so it has to learn to save fuel. But it stays fair: even in the hardest training
weather (gravity 11, engine 85%), a perfect "fall, then full brake" landing needs about 98.

## How the AI senses the new things (plain English)

The Final AI saw 8 numbers, 50 times a second: where it is, how fast it's moving, its tilt and spin, and whether
each leg touches the ground. In Hard Mode it sees **19** (and, like the Final AI, remembers the last 4 moments):

- **Fuel gauge (1 number).** How much of the tank is used up, from 0 (full) to 1 (empty), like the needle on a
  car's fuel gauge read backwards. It has to learn on its own that a needle near 1 means "stop wasting thrust".
- **Pad position (2 numbers changed, 1 added).** Its position is now measured *from the pad*, like a beacon on the
  pad saying "you are 3 m to my left and 8 m up". It still flies toward "zero", which is exactly what the Final AI
  learned. Because the pad can now be near the edge of the screen, it is also told where the pad is on the screen
  (0 = middle, −1 or +1 = an edge), so it can tell how close it is to flying off the side.
- **Obstacle sensors (8 numbers).** Eight proximity sensors, like the parking sensors on a car but reaching 6 m.
  Each one watches a slice of directions, centred on left, down-left (two angles), straight down, down-right (two
  angles), right and up, and reports how close the nearest rock or wall is anywhere in its slice (0 = nothing
  within 6 m, 1 = touching). The slices always point the same way relative to the ground, whichever way the
  lander leans. They ignore the ground itself, which the AI already handles. (The first version used 8 thin laser
  beams instead; see the log below for why that failed.)
- **Mission phase (1 number).** 0 while heading for the first pad, 1 after landing on it. When it flips, the
  "beacon" numbers switch to the second pad.

**Every new number reads 0 when there's no news** (full tank, pad in the middle, nothing nearby, first pad).
This came from the first training try: the first design had readings stuck at 1 when there was nothing to report
(e.g. "nothing within 6 m" = 1 on all 8 sensors × 4 moments). To the network, 32 inputs stuck at 1 act like one
extra "bias" that gets pushed 32 times harder than anything else, and after 500,000 practice steps the carried-over
brain had dropped from 83.5% to 59% on validation. Stopped, re-encoded, restarted from the Final AI.

**Carrying the brain over** (`hard_mode/brain.py`): the Final AI's first layer gets new connections for the 11 new
numbers, all starting at zero. So on its first Hard Mode flight it ignores them and flies exactly like the Final AI.
Checked: on 20 landings with an endless tank, Hard Mode Level 1 matches the original game exactly (5,114 steps,
zero difference) and the widened brain presses the same button as the Final AI at every step.

## Rules (decided before any Hard Mode training)

- **Move on at 90%:** a level is mastered when the brain lands safely on **90%+ of 200 validation landings**
  (seeds 7000–7199) in that level, with random weather. Checked every 500,000 practice steps (and once before any
  practice). The best brain so far is kept.
- **Keep training from the previous brain** (never start over). Level 1 starts from the Final AI.
- **Fair try:** up to 12 million practice steps per level (6× the 2 million budget of the main project). If a level
  isn't mastered by then, the curriculum stops there, and the reason and the best brain are recorded.
- **Hidden test:** 300 landings per level (seeds 40000–40299), never flown during training or validation, run
  **once** at the end with `hard_mode/final_test.py` (committed now, before training; it refuses to run twice).
  It also checks every brain on the earlier levels, to see whether it forgot them.
- **Seeds:** seed 42 first; 142 and 242 if there's time, each its own full curriculum from its own Final AI.
- **Same settings as the Final AI:** the tuned PPO settings, the v2 reward rules (`hard_mode/worlds.py`, measured
  to the target pad), the same "landed safely" rule (at rest on the pad, judged by the physics engine).
- A short video for every mastered level.

## Hard Mode log (overnight)

**Level 1 (fuel tank 110).**
- Seed 42: started at 83.5% on validation (the Final AI as is), **mastered at 91.5% after 1.5 million steps**
  (6 minutes). It learned to save fuel: 88 per safe landing instead of 98, and running the tank dry fell from
  25% to 8% of landings.
- Seed 142: its Final AI already landed 93.0% of the validation landings with the 110 tank, so it counted as
  mastered before any practice and went straight to Level 2.

**Level 2, first attempt (moving pad), learning rate 4.75e-4 as pre-registered.** Both seeds stalled and
swung up and down: seed 42 went 88% → 75.5% → 82.5% → 85% → 86.5% → 80% over 2.5 million steps; seed 142 stayed
between 81.5% and 87.5% for 3.5 million. Two checks before changing anything:
- *Is the level fair?* On 350 calibration landings (seeds 8000–8349), with an endless tank, the Level 1 brain
  stays under 100 fuel on 90% of its landings even for the farthest pads (6 m from the middle). So 110 is enough;
  the failures are crashes while flying sideways in wind, not impossible fuel budgets.
- *Why the swinging?* TensorBoard: on every update 10–16% of the samples hit PPO's limit on how far the policy
  may move (clip fraction 0.10–0.16 with clip range 0.1). The brain was being shoved around by every batch.

**Decision:** from Level 2 on, fine-tune with a quarter of the learning rate (1.2e-4); all other settings unchanged.
The tuned rate was chosen for learning from scratch, where big steps help; polishing a good brain needs small
ones. Level 2 restarted for both seeds from their Level 1 brains (the curriculum rule: always continue from the
previous level's brain). The first attempt is kept in `runs_hard/attempt1_level2_lr4.75e-4/` and
`models_hard/attempt1_level2_lr4.75e-4/`. This decision used validation landings only; the hidden test is untouched.

**Level 2, second attempt (learning rate 1.2e-4).** Both seeds mastered it:
- Seed 142: 86% at the start → **91.0% after 2 million steps** (6 minutes).
- Seed 42: 88% at the start, 83–89% for a while → **91.0% after 5 million steps** (15 minutes).
Validation still wobbles by a few points from check to check (200 landings: ±2 points is noise), so a pass at
91% may be a little lucky; the hidden test will tell.

**Level 3 (obstacles), plain.** Both seeds started at ~78% (the Level 2 brains), and seed 142 stayed flat at
75–78% for 3.5 million steps. Where it fails (seed 142's brain, 300 calibration landings): 36 of 41 obstacle hits
are **walls, in mid-air**; only 5 are rocks near the ground. (Rocks aren't unfair: on safe landings the legs stay
within 2.4 m of the pad centre, and every rock starts at least 2.6 m out.) A wall stands about 3 m before a
far-away pad, 2.5–5 m tall, so the AI must learn a new move: stay high, clear the wall, then drop almost straight
down. The Final AI's habit is a fast diagonal descent, which runs straight into it.

**Decision: an experiment on the two cores.**
- Seed 42 keeps learning plain Level 3 (as planned).
- Seed 142 switches to a **staged curriculum inside Level 3**: the same rocks, but the walls grow in steps, with
  the same 90% rule at every step: 3a walls 0.5–2 m → 3b walls 1.5–3.5 m → the real Level 3 (2.5–5 m) → Level 4.
  Its plain attempt is kept in `runs_hard/attempt_plain_level3_seed142/`.
That's curriculum learning applied once more, inside the hardest level: teach "stay above the wall" with walls
low enough that the current habit almost works, then raise them.

**Level 3 kept failing for a reason the AI couldn't fix: a sensor blind spot.** Both experiments stayed flat
(plain seed 42: 77–80%; staged seed 142 on 2 m walls: 84–89%), and in 14,391 practice landings of the plain run
the obstacle-hit rate stayed at 15% from start to finish: no learning at all. Tracing the sensor readings in two
wall crashes showed why. The sensors were 8 thin laser beams, 30° apart, and a wall is only 0.5–0.9 m wide. As the
lander comes down toward a wall from above, the wall's top sits in the gap between two beams: in one crash, none
of the 8 beams touched the wall until the lander was directly over it. The AI was being punished for hitting
something it could not see.

**Fix: cone-shaped sensors** (`env.py`, `_rays`). Same 8 numbers, same meaning (closeness 0–1, same 8
directions), but each sensor now covers a whole slice of directions and reports the nearest point of any obstacle
in it (obstacle outlines are sampled every 10 cm). In the same two crashes, the wall now shows up from about 4 m
away and its reading rises steadily as the lander closes in. Levels 1–2 never had obstacles, so their sensor
readings were always 0 and their brains are unaffected. All three Level 3 attempts with the laser beams are kept
in `runs_hard/attempt_laser_sensors_level3/` and `models_hard/attempt_laser_sensors_level3/`. Both experiments
restart from their Level 2 brains: seed 42 on plain Level 3, seed 142 on the staged walls (3a → 3b → 3).

**Plain vs staged Level 3, with the cone sensors.**
- Seed 42, plain (walls 2.5–5 m from the start): 77.5% → 77–81% over 4.5 million steps, and in 22,545 practice
  landings the obstacle-hit rate only went 15.7% → 14.7%. Stopped there (kept in
  `runs_hard/attempt_plain_level3_cone_sensors/`): it was barely learning, while the staged run beside it was.
- Seed 142, staged: stage 3a (walls 0.5–2 m) 87.5% → **mastered at 90.5% after 3.5 million steps**, with
  practice obstacle hits falling 10.4% → 5.8%. On to 3b (walls 1.5–3.5 m), where hits started falling too.
- So seed 42 now takes the staged route too, from its Level 2 brain. Lesson: with the walls at full height, a
  brain that flies diagonally almost never stumbles on the "stay high" move by chance, so there's little to learn
  from. With low walls, staying a bit higher works sometimes, gets rewarded, and the habit grows with the walls.

**Stage 3b (walls 1.5–3.5 m): walls are rare, so practise with more of them.** Seed 142 sat at 83.5–88% for
3 million steps; seed 42 started at 86.5%. As wall hits fell (13 → 9 of 200), fuel crashes rose (ran dry 10% →
20%): it was clearing walls by hovering high. Two measurements on 300 calibration landings (the 3a brain, endless
tank):
- **Walls are rare:** only 16% of landings have one. A wall only stands in front of the 2 far pad spots (of 7),
  and only when it fits clear of the pad. Without a wall the brain lands 95.6%; with a full-height wall, 25%.
  Reaching 90% overall needs roughly 60% on the wall landings.
- **The tank is fair:** every successful wall landing used at most 101 fuel. So running dry means crossing the
  wall inefficiently, which is learnable.
**Decision:** in *practice* only, half the landings use a far pad, so walls show up in 36% of practice landings
instead of 14% (`levels.py`, `practice_far_pads`). Validation and the hidden test keep the real, even spread of
pad spots (checked: identical layouts). Both seeds restart stage 3b from their 3a brains; the first 3b attempts are
kept in `runs_hard/attempt_stage3b_no_wall_practice_boost/`.

**Stage 3b with more practice walls.**
- Seed 142: 86.5% → **mastered at 90.0% after 5.5 million steps** (17 minutes). Practice landings showed the
  learning: obstacle hits fell from 23.7% to 18.5% while practice walls were 3× as common as in the real level.
- Seed 42: 86.5% at the start, 85–89% throughout, **best 89.0%: not mastered within the 12 million-step fair
  try.** By the rule, seed 42's curriculum stops here; its best brain is `models_hard/level3b_seed42_best.zip`.
  Why: it was close (1 point; one validation check moves ±2–3 points by luck), but 24 checks in a row stayed
  under 90%.

**The real Level 3 (walls 2.5–5 m), seed 142.** Thanks to the stages it started at **87.5%** (the Level 2 brain
started this level at 77.5%), then hovered around 82–87% for 6 million steps. Where the 3b brain fails, on 500
calibration landings:

| Landings | Share | Landed | Main failure |
|---|---|---|---|
| Pad near the middle (no wall) | 74% | 89.4% | crashes (7%), often after running dry |
| Far pad, no wall | 11% | 87.0% | crashes |
| Far pad with a wall | 16% | 50.0% | hitting the wall (36%) |

With a wall, the weather decides a lot: with a weak engine and heavy gravity (thrust/weight 1.2–1.6) it lands
15% of them; with a strong engine (2.0+) 81%. 90% overall needs roughly 95% on the near pads and 70% with walls.

**Second method, side by side (the free core).** Seed 142, same starting brain (3b), Level 3, with bigger
practice batches (2,048 steps × 16 landers per round instead of 512, batches of 512 instead of 128) and a
learning rate that starts at 2.4e-4 and shrinks to 0 over the 12 million steps. The idea: smaller, steadier steps
as it gets close, to stop the ±3-point wobble and squeeze out the last points. Tagged `bigbatch` (its own
models, logs and TensorBoard runs). Whichever run masters Level 3 first goes on to Level 4.

**Level 3 mastered (seed 142): 90.5% after 8.5 million steps** (25 minutes), with the original settings. The
big-batch second method was stopped after 1.5 million steps (84–87.5%) since it was no longer needed; it's kept in
`runs_hard/attempt_level3_bigbatch_seed142_stopped/`.

**Third seed.** With a core free, seed 242 runs the same recipe from the start: Level 1 at the tuned learning rate
(**mastered: 92.0% after 1 million steps**), then 2 → 3a → 3b → 3 → 4 at the fine-tuning rate, with the cone
sensors and the extra practice walls.

**Level 4 (mission), first try: the AI found a loophole.** It started at 0% (128 of 200 missions hit an obstacle,
70 never took off from pad A). Within 1.5 million steps it had "fixed" the obstacle hits by never taking off:
176 of 200 missions ended sitting on pad A. Landing on A pays +100, and sitting out the 5-second takeoff window
cost nothing, while taking off risked a −100 crash. So doing nothing was the safest choice: my reward design's
fault, not the AI's. Kept in `runs_hard/attempt1_level4_free_exit/`.
**Fixes:** (1) sitting out the takeoff window now scores −100, like a crash: abandoning the mission is a failure,
not a free exit. (2) Learn the mission in stages, like Level 3: **4a** is the mission without rocks and walls
("land, take off, fly 6+ m, land again"), then the real Level 4. Seed 142 restarts from its Level 3 brain.

**Mission stage 4a (no obstacles): teaching it to take off.** Even with the −100 for giving up, 186–197 of 200
validation missions still ended sitting on pad A. Looking inside the brain explained it: on pad A, with the target
switched to B, it fires the side thruster toward B 99% of the time and the main engine about 0.1%. Side thrusters
can't lift it, and at 0.1% it essentially never tries the main engine long enough to discover lift-off. Two things
in the game's own score also pushed against trying:
- **Leg points:** the game gives +10 per leg on the ground, so lifting off cost −20 right away.
- **Speed points:** the game's score charges for speed (that's what makes landings gentle), so each puff of the
  main engine on the pad cost about −2 points at once: more than the small "try the main engine" nudge (+0.5).
  In 3,070 practice missions, 96% stayed stuck (`runs_hard/attempt3_level4a_speed_penalty_on_pad/`).
**Fixes (mission only):** after pad A the leg points no longer count; while waiting on pad A the distance/speed
score is paused and restarts from zero once airborne; and a small nudge for trying the main engine there (+0.5 per
step, at most 10 points). Checked by hand: sitting still now nets 0 after pad A (+100 for landing, −100 for never
leaving), while 30 steps of main engine nets about +123. Tries without these fixes are kept in
`runs_hard/attempt2_level4a_no_takeoff_help/` and `attempt3_level4a_speed_penalty_on_pad/`.

**Stage 4a, more exploration.** With the scoring fixed, practice missions were still 95% stuck after 1.5 million
steps: the brain was so sure of itself on pad A that it almost never sampled the main engine, and one puff alone
doesn't lift it, so there was almost nothing to learn from (`runs_hard/attempt4_level4a_low_exploration/`).
PPO has a knob for this, the *entropy bonus* (points for keeping its options open). The Final AI's tuned value is
tiny (0.0004); for the mission it's raised to 0.01, the starting value from Step 2, and the main-engine nudge to
+1 per step (still at most 20 steps). Result: the AI discovered take-off. Validation missions completed went from
0–2.5% to **20.5% after 500,000 steps**. In practice, once airborne it usually finished the mission (about 20%
completed vs 2–4% crashed): flying to pad B and landing again already worked. **The bottleneck was only the
decision to lift off.** But it hovered at 11–20% for the rest of the 12 million steps: not mastered
(`runs_hard/attempt5_level4a_exploration_only/`, best brain kept there).

**Final attempt at the mission: practise the take-off more often.** Same trick as the extra practice walls in
Level 3. In practice only, half the missions now start parked on pad A (just above it, no push, target already
switched to B, 80–130 fuel already used), so the "take off now" moment comes up about 10× as often. Validation and
the hidden test are unchanged. It continues from the best 4a brain (20.5%), with the exploration bonus 0.01, and
if it masters 4a, the real Level 4 follows with the same settings. This is the last attempt at Level 4: if it falls
short, the curriculum stops there.

**Seed 242 (third seed):** Level 1 mastered (92.0%, 1 million steps), Level 2 (91.0%, 2.5 million), 3a (91.0%,
3 million), 3b (90.0%, 4 million), and **Level 3 mastered: 90.5% after 10.5 million steps**. So two of three
seeds mastered the real Level 3, both through the staged walls. Seed 242 then goes straight to the same mission
recipe as seed 142 (stage 4a with the exploration bonus and parked practice starts, then Level 4).

**Videos so far** (`videos/hard_mode/`): Level 1 and 2 (seed 42), Level 3 (seed 142, since seed 42 didn't master
it): validation landing 7016, over a wall and onto the pad with 7 of 110 fuel left. (Housekeeping: the Level 3
render was meant to pause one training run to stay within 2 cores, but the pause command missed, so for about a
minute 3 processes ran. Fixed for later renders.)

**Final mission attempt (parked practice starts), seed 142: best 75.5%, not mastered.** Validation missions
completed, every 500,000 steps: 20.5% → 36% → 20.5% → 27.5% → 10% → … down to 0–10% between 4 and 8 million
steps (it went through a bad patch while re-learning) … → 16% → 35.5% → 50% → 41.5% → 39.5% → 41% → 48.5% →
**75.5% at 12 million**, the very last check. By the rule, **seed 142's curriculum stops at the mission's first
stage (4a)**, with its best brain saved (`models_hard/level4a_seed142_best.zip`). Why: discovering take-off and
then making it reliable took most of the budget; the last 3.5 million steps show it was getting there.

**Overtime (outside the rules, for information).** Because the curve was still climbing steeply, seed 142 keeps
training from the 75.5% brain for up to another 12 million steps (tagged `overtime`, kept apart in
`runs_hard/level4a_overtime/` and `models_hard/level4a_overtime_*`). If it masters 4a in overtime, the real Level 4
follows, also in overtime. None of this changes the official result above; it just shows what more practice buys.

**Hidden test plan, extended before it runs:** besides the pre-registered rows, it also tests each seed's best
mission-stage (4a) brain on 4a and on the real Level 4, and any overtime brain, labelled as overtime
(`hard_mode/final_test.py`).

**Seed 242, mission stage 4a (same recipe): best 68.5%, not mastered.** Validation missions completed: 14.5% →
2.5% → … 15% → 36% → 43% → 61% → **68.5% at 4 million steps** → then it slid back down: 38%, 15–33% … 3.5% at 12
million. **So seed 242's curriculum also stops at mission stage 4a**, best brain saved
(`models_hard/level4a_seed242_best.zip`).

**What both seeds show:** the exploration bonus that let the AI discover take-off then kept shaking the skill
loose. Both seeds climbed to 68–76% and fell back (seed 142's first overtime try, still with the high bonus, went
75.5% → 73% → 62.5% → 37% → 45% in 2 million steps; kept in `runs_hard/overtime_try1_high_exploration/`).

**Overtime, second try (outside the rules, for information):** both seeds restart from their best mission brains
(75.5% and 68.5%) with the exploration bonus back at the Final AI's normal 0.0004 (take-off is already
discovered) and a learning rate that shrinks from 1.2e-4 to 0 over 12 million steps, so the skill can settle
instead of drifting. If either masters 4a in overtime, the real Level 4 follows, also in overtime.

**Is the mission brain only lucky in practice?** In practice the AI picks buttons partly by chance; in validation it
always presses its top button. If the take-off depended on lucky picks, validation would underrate it. Checked on
100 calibration missions: seed 142's best mission brain completes 71% (top button) vs 72% (picked by chance), seed
242's 75% vs 80%. So no: the best brains genuinely complete about 70–75% of missions, and what keeps going wrong
is the training itself, which climbs to that level and then gets knocked back down.

**Overtime round 1 (normal exploration, shrinking learning rate):**
- Seed 142 (from 75.5%): 67% → 38% → … wobbling 40–67% for 6 million steps … then, as the learning rate shrank,
  75% → 80% → 58–72% → 78% → 83% → **84.0%** → 83%. Best 84.0%, and the steadiest mission run so far.
- Seed 242 (from 68.5%): 39.5–78.5%, mostly 63–74%. Best 78.5%.
Neither reached 90%. **Overtime round 2** continues each from its best overtime brain (84.0% and 78.5%) with the
same settling recipe for another 12 million steps (tag `overtime2`), then the hidden test runs once.

**Overtime round 2:** no gain. Seed 142 wobbled 53–84% (best stayed at its starting 84.0%), seed 242 47–78.5%
(best stayed at 78.5%). Its "best" brains are the round-1 brains, so they're filed in
`models_hard/overtime_round2_no_gain/` and not tested twice. **Training ends here** (05:46).

## Pause for moving the Mac (2026-10-17, 12:59)

The user asked to safely pause everything for ~45 minutes (power and internet off, lid closed).
- **Nothing from this project was running**, so nothing needed a checkpoint or a freeze. Hard Mode training ended
  at 05:46 (last runs: overtime round 2), the one-time hidden test ran at 05:47–05:50, and every result, brain,
  log, chart and video is committed on this branch (`hard-mode`). `main` is committed too (9f32e14).
- Still open: the local LANDFALL web server (`python app.py` in `~/rocket-lander`, port 7860) was left as it was.
  It's idle and just sleeps with the Mac; if it's gone after the move, restart it with `.venv/bin/python app.py`.
- Not touched: the walking-robot project's training (`~/robot-walker`, `train.py --run a`), which belongs to a
  different session. Closing the lid puts the Mac to sleep, which pauses it, and it continues on wake.
- **To resume:** there's nothing to unfreeze for this project. Next steps are waiting on the user: the GitHub repo
  link (to push `main`) and which parts of Hard Mode, if any, to bring onto `main`.

## Parked (2026-10-17, 19:40)

**Where it stopped:** at a clean point, with nothing half-done and no training running.
- All 8 steps are done. The site is live at https://s4dai-landfall.static.hf.space, and the code is on GitHub at
  https://github.com/saadhilalai/landfall. GitHub's "About" box has the live demo link, the description ("Self-teaching
  rocket-landing AI with a live demo, built by Saad Hilal · S4D") and the topics reinforcement-learning, pytorch, ppo,
  domain-randomization and curriculum-learning.
- Hard Mode is finished and on `main`, without its training logs. The full logs stay on the local `hard-mode` branch
  (folder `~/rocket-lander-hard-mode`), which isn't pushed.
- Still running: only the idle local web server (`python app.py`, port 7860). If it's gone, restart it with
  `.venv/bin/python app.py`.
- This note is committed locally only, so `main` is one commit ahead of GitHub until the next push.

**What's next (nothing is waiting; all optional):**
1. ~~Push this note to GitHub.~~ Done.
2. ~~A LinkedIn link for the site footer.~~ Done 2026-10-17: https://www.linkedin.com/in/saad-hilal-24b989444/
   next to GitHub; live site redeployed and checked on desktop and phone.
3. Hard Mode (skipped for now, by choice): the mission level. Ideas: mission stages with obstacles (low walls first, like
   Level 3), a bigger network, and requiring two validation passes in a row.

## Launch film on the website (2026-10-19)

The /brag trailer now opens the LANDFALL site, above the live landing (like the robot project's site).
- `make_film.py` cuts `brag-output/brag.mp4` for the web into `web/media/`: full HD for computers (2.1 MB) and
  1280x720 for phones (1.2 MB), each with its first frame as the poster (78 KB / 46 KB). Screens up to 700 px wide get
  the phone cut.
- It plays muted and looping once the loading screen is gone. Below it there's a thin bar: a progress line, a T+ clock,
  pause/play, Unmute (the first unmute restarts the film so the music plays from the top) and "Fly it live", which
  scrolls to the landing. It pauses whenever it's scrolled out of view. With reduced motion or data saver it waits on
  its poster for Play.
- The film is as wide as the screen allows with its control bar still in view: full width on phones and tablets,
  about 84% on a 16:10 laptop, where its edges fade into the black page.
- The live landing now waits on the pad and flies when it scrolls into view. The automatic first flight no longer
  writes a share link into the address bar, so a reload opens on the film. Shared flight links open on the flight.
- Footer and Space card: music credit, "Happy Beats & Business Moves Vol. 12" by Sascha Ende (ende.app), CC BY 4.0.
- Checked in headless Chrome at 1470x830 (desktop) and 390x844 (phone): playback, unmute, pause, scroll pause,
  "Fly it live", reduced motion, no sideways scrolling.

## Privacy check (2026-10-19)

Before more people see the project, every file in every commit on both branches, the Hugging Face Space and
its deploy history were checked for passwords, keys and tokens, personal emails, phone numbers, addresses,
locations, Mac file paths, the computer's name, and hidden data in images and videos. Nothing secret turned
up. Three things were cleaned:

- **A Mac folder path inside nine Hard Mode brains** (the overtime and big-batch runs, trained with
  `--lr-decay`). The shrinking learning rate was a Python `lambda`, and a saved brain stores a lambda's code
  together with the full path of the file it came from. Each brain was saved again with only that path changed
  to `hard_mode/train.py`: the weights are byte for byte the same, and 20 validation landings per brain came
  out identical. `hard_mode/train.py` now uses SB3's `LinearSchedule`, which a saved brain stores by name only.
- **The computer's name in TensorBoard log file names** (`events.out.tfevents.<time>.<computer>.<process>.<n>`)
  is now `localhost`. TensorBoard reads them the same way. New training runs will name their log files after
  the computer again, so rename them the same way before committing.
- **A Mac folder path in `brag-output/composition-brief.md`** is now relative.

The history was rewritten so no earlier version contains them either, and commit times are now stored in UTC
(the same moments, without a time zone). Commit IDs changed again (table above). The GitHub repository was
deleted and created again under the same name, because GitHub keeps serving replaced commits by their old IDs;
the new repository holds only the clean history.
