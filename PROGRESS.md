# Rocket Lander: Progress Notes

_Last updated 2026-10-15 (evening). Nothing is running. All models, results and logs are saved and committed._

| Step | Status |
|---|---|
| 1. Setup | ✅ Done |
| 2. Baseline agent | ✅ Done (redone with 3 seeds) |
| 3. Custom reward shaping | ✅ Done (v1, then a tuning round to v2) |
| 4. Domain randomization | ✅ Done. Final agent design: randomized + memory |
| 5. Optuna auto-tuning | ✅ Done. **Final agent: `randomized_memory_tuned`** |
| 6. Final results, charts, videos | Not started |
| 7. Gradio demo | Not started |
| 8. README, packaging, resume bullets | Not started (`requirements.txt` and `.gitignore` exist) |

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
| `compare.py` | Compares agents on identical landings, with a paired significance test (`--seeds`, `--scenarios`, `--reference`) |
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

## What to do next, in order

1. **Step 6:** final evaluation on the untouched **test** landings (10000–10099):
   - Normal and harsh conditions, plus the other scenarios, with the extreme rows labelled as limits.
   - Learning-curve and success-rate charts.
   - Videos: an untrained crash, each agent landing, and the final agent (`randomized_memory_tuned`) in heavy wind.
2. **Step 7:** Gradio demo with gravity, wind and engine sliders and a Launch button, using the final agent.
3. **Step 8:** README (including the leg-sensor bug story, the v1 → v2 iteration, the "more practice made it more fragile" finding, and the honest tuning), packaging, and 3 resume bullets with the real numbers.
