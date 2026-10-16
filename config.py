"""
config.py — every setting used to train and test the agents, in one place.

All three agents (baseline, reward-shaped, randomized) use EXACTLY these
settings. The only thing that changes between them is the world they
practise in. That way, any difference in results comes from our idea,
not from luck or different knobs.
"""

ENV_ID = "LunarLander-v3"

# ---- Random seeds (fixed, so every run can be reproduced) -------------------
# Each agent is trained 3 times with different seeds, to show results aren't luck.
# A seed sets the neural network's starting weights and the practice worlds. The same
# seed gives every agent type the same starting brain, so comparisons are paired.
# Spaced 100 apart so the 16 parallel simulators of different runs never share a seed.
TRAIN_SEEDS = (42, 142, 242)

CURVE_EVAL_SEED = 2_024   # the quick check-ups drawn on the learning curve
VALIDATION_SEED = 5_000   # 100 landings (seeds 5000-5099) used to decide "is it solved yet?"
ANALYSIS_SEED = 20_000    # 1,000 fresh landings (seeds 20000-20999) for comparing agents while building
TEST_SEED = 10_000        # 100 landings (seeds 10000-10099) kept untouched for the final report

# ---- Training budget --------------------------------------------------------
SOLVED_SCORE = 200              # official "solved" mark for LunarLander
FIRST_CHECK_STEPS = 1_000_000   # train this long, then run the validation test
EXTRA_CHUNK_STEPS = 250_000     # if not solved, keep training in chunks of this size...
MAX_STEPS = 2_000_000           # ...up to this many steps in total
N_VALIDATION_EPISODES = 100
N_ANALYSIS_EPISODES = 1_000

# ---- Parallel simulators ----------------------------------------------------
N_ENVS = 16   # 16 landers practise at the same time

# ---- PPO settings -----------------------------------------------------------
# Starting point: the published, tuned LunarLander settings from RL Baselines3 Zoo.
# Step 5 (Optuna) will search for better ones for the final agent.
PPO_KWARGS = dict(
    policy="MlpPolicy",
    n_steps=1024,         # each lander plays 1024 steps before every learning update
    batch_size=64,        # learn from 64 moments at a time
    n_epochs=4,           # go over each batch of experience 4 times
    gamma=0.999,          # how much future rewards matter (0.999 = very far-sighted)
    gae_lambda=0.98,      # smoothing for "was this action better than expected?"
    ent_coef=0.01,        # small bonus for trying new things (exploration)
    learning_rate=3e-4,   # how big each brain update is
    clip_range=0.2,       # PPO's safety limit: never change the policy too much at once
    policy_kwargs=dict(net_arch=dict(pi=[64, 64], vf=[64, 64])),  # actor + critic, 2 layers of 64
)

# ---- Domain randomization (Step 4) --------------------------------------------
# The randomized agent re-rolls these at the start of every practice episode.
# Normal game: gravity -10, no wind, engine 1.0x. The engine is 1.8x stronger than
# gravity normally; the weakest combination below (gravity -11, engine 0.85x) is 1.39x.
TRAIN_CONDITIONS = dict(
    gravity=(-11.0, -8.0),         # m/s^2-ish, negative = down
    wind_power=(0.0, 15.0),        # sideways gusts (the game's own default when wind is on: 15)
    turbulence_power=(0.0, 1.5),   # random twisting (the game's default: 1.5)
    engine_power=(0.85, 1.15),     # multiplier on the game's engine strength
)

# Test scenarios. Every changed value lies OUTSIDE the training ranges above,
# but stays landable: in "harsh" the engine is still 1.2x stronger than gravity.
TEST_SCENARIOS = dict(
    normal=dict(),
    strong_wind=dict(wind_power=20.0, turbulence_power=2.0),
    heavy_gravity=dict(gravity=-12.0),
    weak_engine=dict(engine_power=0.8),
    harsh=dict(gravity=-12.0, wind_power=20.0, turbulence_power=2.0, engine_power=0.8),
)

# ---- Learning-curve check-ups -----------------------------------------------
CURVE_EVAL_EVERY = 32_768   # steps between check-ups (= every 2 learning updates)
CURVE_EVAL_EPISODES = 10    # landings per check-up
