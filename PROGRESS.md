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

**Run exactly once.** `final_eval.py` (the agents, seeds, scenarios, metrics, paired tests and video rules) was committed in `6ca403d` **before** it ran, and it refuses to run a second time. Before that, it was dry-run on throwaway landings (seeds 25000+) in a single process, so the test landings stayed untouched.
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

## Still open (needs the user)

1. GitHub and LinkedIn links for the site footer (`LINKS` at the top of `web/app.js`).
2. Create the GitHub repository and push (the repo has no remote yet).
3. Before the repo is public, decide:
   - `brag-output/brag.mp4` contains the /brag plugin's bundled music (ende.app); its licence isn't verified.
   - Commits use a personal email; GitHub can show a private "noreply" address instead.
   - A licence for the code (none yet; MIT is the usual choice for a portfolio project).
