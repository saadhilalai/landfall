# Rocket Lander: Progress Notes

_Parked on 2026-10-15. Nothing is running. All models, results and logs are saved and committed._

| Step | Status |
|---|---|
| 1. Setup | ✅ Done |
| 2. Baseline agent | ✅ Done (redone with 3 seeds) |
| 3. Custom reward shaping | ✅ Done (v1, then a tuning round to v2) |
| 4. Domain randomization | ⚠️ Trained and stress-tested, **not yet reviewed together**. The result is negative (see below), so we need to choose a fix |
| 5. Optuna auto-tuning | Not started |
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
    --scenarios normal strong_wind heavy_gravity weak_engine harsh --episodes 500   # stress test (~3 min)
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
| `compare.py` | Compares agents on identical landings, with a paired significance test |
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

## Step 4: Domain randomization (done, not yet reviewed)

The randomized agent uses the v2 reward, and every practice episode re-rolls the physics. Each harsh value is outside the training range, but the harsh world is still physically landable on paper: the engine is 1.2× stronger than gravity.

| | Gravity | Wind | Turbulence | Engine | Engine ÷ gravity |
|---|---|---|---|---|---|
| Normal game | −10 | off | off | 1.0× | 1.80 |
| **Training range** (re-rolled every episode) | −11 to −8 | 0–15 | 0–1.5 | 0.85–1.15× | ≥ 1.39 |
| strong_wind test | −10 | **20** | **2.0** | 1.0× | 1.80 |
| heavy_gravity test | **−12** | off | off | 1.0× | 1.50 |
| weak_engine test | −10 | off | off | **0.8×** | 1.44 |
| **harsh** test (all at once) | **−12** | **20** | **2.0** | **0.8×** | 1.20 |

**Training:** every seed passed 200 on the normal game. Seed 242 needed one extension, to 1.28M steps. Validation scores: 213.7 / 231.9 / 216.0, with 100% / 93% / 84% landed.

**Stress test:** 500 analysis landings × 3 seeds per agent and scenario (`runs/stress_test_3seeds.json`)

| Landed safely (official score) | Baseline | Shaped v2 | Randomized |
|---|---|---|---|
| normal | 99.1% (258) | **99.8% (268)** | 92.9% (224) |
| strong_wind | 18.3% (19) | 15.5% (17) | **20.7% (47)** |
| heavy_gravity | 97.1% (248) | **98.8% (261)** | 94.5% (220) |
| weak_engine | 92.9% (234) | **96.1% (253)** | 93.5% (217) |
| harsh | 0.2% (−231) | 0.5% (−193) | 0.0% (−152) |

**The headline, stated honestly: in its current form, domain randomization did NOT make the agent more robust.**
- In the **normal** game it's clearly worse: 92.9% vs 99.8% landed, and about 80% more fuel (247 vs 136).
- Under **heavy gravity** and a **weak engine**, shaped v2 (no randomization) is the most robust. The randomized agent is roughly at baseline level.
- In **strong wind** every agent fails, landing only 15–21% of the time. The randomized agent is a little better: 20.7% vs 15.5% for v2 (p = 0.0001), but not significantly better than the baseline (p = 0.07).
- In **harsh** conditions everyone lands about 0%. As it stands, this test can't tell the agents apart.

**Likely reasons (guesses, none tested yet):**
1. **Under-training.** The budget rule stops training as soon as the normal-game score reaches 200. The randomized agent has a much harder job but got the same ~1M steps.
2. **It can't tell what conditions it's in.** It only sees the current moment (8 numbers), so it can't work out the wind or engine strength from how it has been moving. The usual fix is a short memory: stack the last few observations, or use an LSTM policy.
3. **"harsh" and "strong_wind" might not be landable at all.** That hasn't been checked yet.
4. **Wind is drawn evenly from 0 to 15**, so strong wind was rare during training.

---

## What to do next, in order

1. **Review the Step 4 results together** (the section above).
2. **Choose a fix for Step 4.** My recommendation, each step taking about 5–10 minutes of compute:
   - **a. Check whether the harsh tests are even landable.** Train one specialist agent directly in `harsh` (and in `strong_wind`). If even a specialist can't land, soften the scenario. **Settle the new scenario values before looking at any agent's results**, so they aren't cherry-picked.
   - **b. Give the randomized agent the full 2M-step budget**, instead of stopping at the first score of 200. This changes the budget rule, so decide whether every agent gets 2M for fairness.
   - **c. Optionally, give it memory** by stacking the last 4 observations, so it can sense wind and engine strength from its recent motion.
   - Then re-run the stress test.
3. **Step 5:** Optuna search for the final agent's settings (about 1 hour).
4. **Step 6:** final evaluation on the untouched **test** landings (10000–10099), in normal and harsh conditions. Learning-curve and success-rate charts. Videos: an untrained crash, each agent landing, the final agent in heavy wind.
5. **Step 7:** Gradio demo with gravity, wind and engine sliders and a Launch button.
6. **Step 8:** README (including the leg-sensor bug story and the v1 → v2 iteration), packaging, and 3 resume bullets with the real numbers.
