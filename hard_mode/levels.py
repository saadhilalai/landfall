"""
hard_mode/levels.py — the four Hard Mode levels. Each level keeps everything from the levels before it,
and every level keeps Step 4's random weather and engine power (config.TRAIN_CONDITIONS).
"""

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class Level:
    number: int
    name: str
    fuel_tank: float      # fuel units in a full tank (main engine 1 per step, side thruster 0.1)
    random_pad: bool      # pad in one of 7 spots instead of always in the middle
    obstacles: bool       # rocks next to the pad, and a wall between the start and a far pad
    mission: bool         # land on pad A, take off, land on pad B
    max_steps: int        # time limit (50 steps = 1 simulated second)
    wall_height: tuple = (2.5, 5.0)   # walls stand this many metres above the ground
    practice_far_pads: float = 0.0    # PRACTICE ONLY: share of landings with a far pad (so a wall is likely).
                                      # Validation and the hidden test always use the real, even spread.
    practice_parked_start: float = 0.0  # PRACTICE ONLY (missions): share starting parked on pad A, target B


# The Final AI burns ~100 fuel per landing in the random training weather (calibrate.py): with a 110 tank it
# would have run dry on 23% of its landings. The hardest training weather needs ~98 even for a perfect burn.
TANK = 110.0

L1 = Level(1, "Limited fuel tank", TANK, random_pad=False, obstacles=False, mission=False, max_steps=1000)
L2 = replace(L1, number=2, name="Landing pad in a random spot", random_pad=True)
L3 = replace(L2, number=3, name="Obstacles: rocks and walls", obstacles=True)
L4 = replace(L3, number=4, name="Mission: pad A, then pad B", mission=True, fuel_tank=2.5 * TANK, max_steps=2000)

# Level 3 in stages (added overnight, see PROGRESS.md): the same rocks, but the walls grow in two steps before
# the real Level 3. Same rule at every stage: 90% on the validation landings to move up.
L3A = replace(L3, name="Obstacles, walls up to 2 m", wall_height=(0.5, 2.0))
L3B = replace(L3, name="Obstacles, walls up to 3.5 m", wall_height=(1.5, 3.5))
# Walls only stand in front of the 2 far pad spots (of 7), so only ~16% of landings have one. In practice, half
# the landings now use a far pad, so the AI meets walls ~3x as often. Evaluation is unchanged.
L3B = replace(L3B, practice_far_pads=0.5)
L3 = replace(L3, practice_far_pads=0.5)
# Level 4 in stages, if needed: the mission without rocks and walls first ("land, take off, fly, land"),
# then the real Level 4 with them.
L4A = replace(L4, name="Mission without obstacles", obstacles=False)
# The take-off decision happens once per mission, so it was learned slowly (best 20.5%). In practice, half the
# missions now start parked on pad A, so that moment comes up far more often. Evaluation is unchanged.
L4A = replace(L4A, practice_parked_start=0.5)
L4 = replace(L4, practice_parked_start=0.5)
BY_KEY = {"1": L1, "2": L2, "3a": L3A, "3b": L3B, "3": L3, "4a": L4A, "4": L4}
LEVELS = {lv.number: lv for lv in (L1, L2, L3, L4)}
PREVIOUS = {"2": "1", "3": "2", "4": "3", "3a": "2", "3b": "3a", "4a": "3"}   # default starting brain for each level

# Landings kept apart, like the main project (config.py):
VALIDATION_SEEDS = range(7000, 7200)    # 200 landings: the "90% to move on" check, used during training
HIDDEN_TEST_SEEDS = range(40000, 40300) # 300 landings: untouched until the one-time final test
PASS_RATE = 0.90
