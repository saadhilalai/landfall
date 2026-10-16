"""
tune.py — Step 5: Optuna searches for better PPO training settings for the final agent.

HONESTY RULES (so the Step 6 results stay trustworthy):
  * Every trial trains in the TRAINING conditions only (the randomized ranges in config.py).
  * Every trial is scored ONLY on the 100 validation landings (seeds 5000-5099), flown in
    those same training conditions. The search never sees the stress-test scenarios
    or the final test landings.

How the search works:
  * One trial = train a randomized_memory agent (seed 42, the same 2M steps as the real
    agent) with settings Optuna picks, then score it: the average official score.
  * Trial 0 is the current settings, so the search has to beat them.
  * Every 500k steps a trial reports its score so far. From 1M steps on, a trial doing
    worse than the median of earlier trials at the same point is stopped early ("pruned").
  * Optuna's TPE sampler remembers which settings scored well and proposes new ones near them.
  * Several trials run at once, in separate processes that share one study file.

Usage:
    python tune.py search --minutes 45 --workers 3   # run the search (~1 hour incl. the last trials)
    python tune.py report                             # table + charts of the finished search
    python tune.py confirm                            # tuned vs current agent, 3 seeds, validation landings
"""

import argparse
import json
import multiprocessing as mp
import warnings
from functools import partial
from pathlib import Path

import numpy as np
import optuna
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend

import config

warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)

OUT_DIR = Path("runs") / "optuna"
JOURNAL = OUT_DIR / "study_journal.log"
STUDY_NAME = "randomized_memory_ppo"
AGENT = "randomized_memory"
TRIAL_SEED = 42
CHECKPOINTS = (500_000, 1_000_000, 1_500_000, 2_000_000)
PRUNE_FROM = 1_000_000


def to_ppo_kwargs(params):
    """Optuna's flat parameter dict -> keyword arguments for PPO."""
    p = dict(params)
    width = p.pop("net_width")
    return dict(p, policy_kwargs=dict(net_arch=dict(pi=[width, width], vf=[width, width])))


def current_params():
    """The settings every agent so far was trained with, as Optuna parameters."""
    k = config.PPO_KWARGS
    keys = ("learning_rate", "n_steps", "batch_size", "n_epochs", "gamma", "gae_lambda", "ent_coef", "clip_range")
    return dict({key: k[key] for key in keys}, net_width=k["policy_kwargs"]["net_arch"]["pi"][0])


def suggest_params(trial):
    """The search space: a range around the current settings for each knob."""
    return dict(
        learning_rate=trial.suggest_float("learning_rate", 1e-4, 1e-3, log=True),
        n_steps=trial.suggest_categorical("n_steps", [512, 1024, 2048]),
        batch_size=trial.suggest_categorical("batch_size", [64, 128, 256]),
        n_epochs=trial.suggest_int("n_epochs", 3, 10),
        gamma=trial.suggest_categorical("gamma", [0.99, 0.995, 0.999]),
        gae_lambda=trial.suggest_categorical("gae_lambda", [0.9, 0.95, 0.98]),
        ent_coef=trial.suggest_float("ent_coef", 1e-4, 0.02, log=True),
        clip_range=trial.suggest_categorical("clip_range", [0.1, 0.2, 0.3]),
        net_width=trial.suggest_categorical("net_width", [64, 128, 256]),
    )


def validation_score(model, n_episodes):
    """Average official score on the validation landings, in the TRAINING conditions."""
    from evaluate import run_episodes, summarize
    s = summarize(run_episodes(model, n_episodes, config.VALIDATION_SEED, conditions=config.TRAIN_CONDITIONS))
    return s["mean_score"], s["success_rate"]


def objective(trial, checkpoints, n_episodes):
    import torch
    from stable_baselines3 import PPO
    from envs import make_vec_env

    torch.set_num_threads(1)
    params = suggest_params(trial)
    env = make_vec_env(AGENT)
    model = PPO(env=env, seed=TRIAL_SEED, device="cpu", verbose=0,
                **{**config.PPO_KWARGS, **to_ppo_kwargs(params)})
    try:
        for i, checkpoint in enumerate(checkpoints):
            model.learn(checkpoint - model.num_timesteps, reset_num_timesteps=(i == 0))
            score, landed = validation_score(model, n_episodes)
            trial.set_user_attr(f"landed_at_{checkpoint}", landed)
            trial.report(score, step=checkpoint)
            if trial.should_prune():
                raise optuna.TrialPruned()
    finally:
        env.close()
    return score


def load_study(journal, worker_id=0):
    return optuna.create_study(
        study_name=STUDY_NAME,
        storage=JournalStorage(JournalFileBackend(str(journal))),
        direction="maximize",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=worker_id, n_startup_trials=8, multivariate=True),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=PRUNE_FROM - 1),
    )


def worker(worker_id, journal, minutes, checkpoints, n_episodes, max_trials):
    study = load_study(journal, worker_id)
    study.optimize(partial(objective, checkpoints=checkpoints, n_episodes=n_episodes),
                   timeout=minutes * 60, n_trials=max_trials)


def search(args):
    journal = Path(args.journal)
    if journal.exists():
        raise SystemExit(f"{journal} already exists. Move or delete it to start a fresh search.")
    journal.parent.mkdir(parents=True, exist_ok=True)
    checkpoints = (20_000, 40_000) if args.smoke else CHECKPOINTS
    n_episodes = 5 if args.smoke else config.N_VALIDATION_EPISODES
    max_trials = 2 if args.smoke else None

    study = load_study(journal)
    study.enqueue_trial(current_params())   # trial 0: the settings to beat

    ctx = mp.get_context("spawn")
    procs = [ctx.Process(target=worker, args=(i, journal, args.minutes, checkpoints, n_episodes, max_trials))
             for i in range(args.workers)]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
    report(args)


def report(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from optuna.visualization.matplotlib import (plot_intermediate_values, plot_optimization_history,
                                                 plot_param_importances)

    study = load_study(Path(args.journal))
    trials = sorted(study.trials, key=lambda t: t.number)
    states = [t.state.name for t in trials]
    print(f"{len(trials)} trials: {states.count('COMPLETE')} complete, {states.count('PRUNED')} pruned, "
          f"{states.count('FAIL')} failed\n")
    print(f"{'#':>3} {'state':9} {'score':>7} {'landed':>7}  settings")
    for t in sorted(trials, key=lambda t: -(t.value if t.value is not None else -1e9)):
        last = max(t.intermediate_values) if t.intermediate_values else None
        score = t.value if t.value is not None else (t.intermediate_values[last] if last else float("nan"))
        landed = t.user_attrs.get(f"landed_at_{last}", float("nan")) if last else float("nan")
        print(f"{t.number:>3} {t.state.name:9} {score:7.1f} {landed:7.0%}  {t.params}")

    best = study.best_trial
    out = Path(args.journal).parent
    (out / "best_params.json").write_text(json.dumps(
        dict(trial=best.number, validation_score=best.value, params=best.params,
             current_settings_score=next((t.value for t in trials if t.number == 0), None)), indent=2))
    print(f"\nBest: trial {best.number}, validation score {best.value:.1f} -> {out / 'best_params.json'}")

    if len([t for t in trials if t.state.name == "COMPLETE"]) >= 2:
        for name, plot in (("history", plot_optimization_history), ("importance", plot_param_importances),
                           ("checkpoints", plot_intermediate_values)):
            ax = plot(study)
            ax.figure.set_size_inches(9, 5)
            ax.figure.tight_layout()
            ax.figure.savefig(out / f"{name}.png", dpi=120)
            plt.close(ax.figure)
        print(f"Charts saved in {out}/")


def confirm(args):
    """Tuned vs current settings: 3 seeds each, validation landings in training conditions.
    Decision rule (set before seeing the results): adopt the tuned settings only if their
    average validation score across the 3 seeds is higher."""
    import torch
    from stable_baselines3 import PPO

    torch.set_num_threads(1)
    results = {}
    for agent in ("randomized_memory", "randomized_memory_tuned"):
        per_seed = {}
        for seed in config.TRAIN_SEEDS:
            model = PPO.load(f"models/{agent}_seed{seed}.zip", device="cpu")
            score, landed = validation_score(model, config.N_VALIDATION_EPISODES)
            per_seed[seed] = dict(score=score, landed=landed)
        scores = [v["score"] for v in per_seed.values()]
        landed = [v["landed"] for v in per_seed.values()]
        results[agent] = dict(per_seed=per_seed, mean_score=float(np.mean(scores)), std_score=float(np.std(scores)),
                              mean_landed=float(np.mean(landed)))
        print(f"{agent:24s} score {np.mean(scores):6.1f} ± {np.std(scores):4.1f} | landed {np.mean(landed):6.1%} | "
              + ", ".join(f"seed {s}: {v['score']:.1f}" for s, v in per_seed.items()))
    adopt = results["randomized_memory_tuned"]["mean_score"] > results["randomized_memory"]["mean_score"]
    results["adopt_tuned"] = adopt
    print(f"\nDecision: {'ADOPT the tuned settings' if adopt else 'KEEP the current settings'}")
    (OUT_DIR / "confirm.json").write_text(json.dumps(results, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["search", "report", "confirm"])
    parser.add_argument("--minutes", type=float, default=45)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--journal", default=str(JOURNAL))
    parser.add_argument("--smoke", action="store_true", help="tiny run to check the plumbing")
    args = parser.parse_args()
    {"search": search, "report": report, "confirm": confirm}[args.command](args)


if __name__ == "__main__":
    main()
