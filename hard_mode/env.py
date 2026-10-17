"""
hard_mode/env.py — Hard Mode: LunarLander with a fuel tank, a pad that moves, obstacles and two-pad missions.

This is the game's own code (gymnasium/envs/box2d/lunar_lander.py, reset + step) with four additions,
switched on level by level (see levels.py). Everything else is the original game, so an agent that knows
the original game starts out knowing most of this one.

  1. FUEL TANK   Each landing starts with a full tank. The main engine burns 1 unit per step, a side
                 thruster 0.1 (the same 10:1 ratio as before). At empty, the engines stop working.
  2. RANDOM PAD  The pad (and the flat ground around it) sits in one of 7 spots across the ground.
  3. OBSTACLES   Rocks next to the pad, and a tall wall between the start and a far-away pad.
                 Touching an obstacle with the body or a leg counts as a crash.
  4. MISSION     Two pads, A and B: land on A and come to rest, take off, fly to B, land again.

What the AI senses (19 numbers per moment; the memory agents see the last 4 moments, so 76 inputs):
   0-1  where it is, measured FROM THE TARGET PAD (left/right, up/down), like a beacon on the pad
   2-7  its speed, tilt, spin and the two leg sensors, exactly as in the original game
   8    FUEL GAUGE: how much of the tank is used up (0 = full tank, 1 = empty)
   9    where the pad is on the screen (0 = middle, -1/+1 = the edges), so it can tell how close it is
        to flying off the side now that the pad can move
  10-17 OBSTACLE SENSORS: 8 proximity sensors, like a car's parking sensors. Each watches a slice of directions
        (centred on left, down-left, down, down-right, right and up) and reads how close the nearest rock or
        wall is in that slice (0 = nothing within 6 m, 1 = touching). Slices, not thin beams: the first
        version used 8 laser beams, and a thin wall could hide in the gap between two beams.
  18    MISSION PHASE: 0 = heading for the first pad, 1 = heading for the second

Every new number is exactly 0 when it carries no news (full tank, pad in the middle, nothing nearby, first
pad). That matters: a reading stuck at a constant non-zero value acts like an extra "bias" that training
pushes on dozens of times over, and it knocked the carried-over brain off balance in the first try.

In the original game the pad is always in the middle, so numbers 0-7 mean exactly what they meant
there. That's what lets the Final AI's brain carry over (see brain.py).
"""

import math

import Box2D
import gymnasium as gym
import gymnasium.envs.box2d.lunar_lander as ll
import numpy as np
from Box2D.b2 import contactListener, edgeShape, fixtureDef, polygonShape, revoluteJointDef
from gymnasium import spaces

W = ll.VIEWPORT_W / ll.SCALE          # 20 world units across
H = ll.VIEWPORT_H / ll.SCALE          # 13.3 world units high
CHUNKS = 11                           # the ground is 10 straight segments between 11 points
CHUNK_X = [W / (CHUNKS - 1) * i for i in range(CHUNKS)]
PAD_HALF = 2.0                        # the pad spans one ground point on each side of its centre (world units)
PAD_SPOTS = range(2, 9)               # pad centre at ground point 2..8 -> 7 possible spots (5 = the middle)
START_X, START_Y = W / 2, H           # where the lander starts (top middle), as in the original game

OBS_SIZE = 19
N_RAYS = 8
RAY_ANGLES = (-90, -60, -30, 0, 30, 60, 90, 180)   # degrees from straight down; + points right
RAY_LENGTH = 6.0
SENSOR_DIRECTIONS = np.array(RAY_ANGLES, dtype=float)
OBSTACLE_BIT = 0x0040
FUEL_COST = {0: 0.0, 1: 0.1, 2: 1.0, 3: 0.1}
TAKEOFF_WINDOW = 250                  # mission: steps (5 s) allowed on pad A before it must be airborne


class _Contacts(contactListener):
    """The original game's contact rules, plus: an obstacle touching the body or a leg is a crash."""

    def __init__(self, env):
        contactListener.__init__(self)
        self.env = env

    def _bodies(self, contact):
        return contact.fixtureA.body, contact.fixtureB.body

    def BeginContact(self, contact):
        g = self.env
        a, b = self._bodies(contact)
        hit_obstacle = g.obstacle_body is not None and g.obstacle_body in (a, b)
        if g.lander in (a, b):
            g.game_over = True
            g.hit_obstacle |= hit_obstacle
        for leg in g.legs:
            if leg in (a, b):
                if hit_obstacle:
                    g.game_over = True
                    g.hit_obstacle = True
                else:
                    leg.ground_contact = True

    def EndContact(self, contact):
        # Kept exactly as in the original game, including its bug: when a leg slides from one ground
        # segment onto the next, its flag switches off (see landing_stats.py). The AI learned with it.
        a, b = self._bodies(contact)
        for leg in self.env.legs:
            if leg in (a, b):
                leg.ground_contact = False


class HardLander(ll.LunarLander):
    """LunarLander with the Hard Mode additions. `level` is a Level from levels.py."""

    def __init__(self, level, render_mode=None, practice=False):
        super().__init__(render_mode=render_mode)
        self.level = level
        self.practice = practice          # practice landings may see walls more often (levels.py)
        big = np.full(OBS_SIZE, np.inf, dtype=np.float32)
        self.observation_space = spaces.Box(-big, big, dtype=np.float32)
        self.obstacle_body = None
        self.hit_obstacle = False

    # ---- Layout: pads and obstacles ---------------------------------------------------------------
    def _ground_y(self, x):
        """Height of the ground at x (the ground is straight lines between CHUNK_X points)."""
        i = min(max(int(x // (W / (CHUNKS - 1))), 0), CHUNKS - 2)
        x0, x1 = CHUNK_X[i], CHUNK_X[i + 1]
        return self.smooth_y[i] + (self.smooth_y[i + 1] - self.smooth_y[i]) * (x - x0) / (x1 - x0)

    def _draw_layout(self):
        """Random pad spots and obstacles. Drawn AFTER the original game's random draws, so with the pad in
        the middle and no obstacles, a landing number gives exactly the original game's landing."""
        rng, lv = self.np_random, self.level
        if lv.mission:
            a = int(rng.integers(PAD_SPOTS.start, PAD_SPOTS.stop))
            b = a
            while abs(b - a) < 3:
                b = int(rng.integers(PAD_SPOTS.start, PAD_SPOTS.stop))
            spots = [a, b]
        elif lv.random_pad and self.practice and lv.practice_far_pads and rng.random() < lv.practice_far_pads:
            spots = [int(rng.choice([PAD_SPOTS.start, PAD_SPOTS.stop - 1]))]   # a far pad: room for a wall
        elif lv.random_pad:
            spots = [int(rng.integers(PAD_SPOTS.start, PAD_SPOTS.stop))]
        else:
            spots = [CHUNKS // 2]

        plan = []   # obstacles as (kind, numbers), turned into shapes once the ground exists
        if lv.obstacles:
            pads = [CHUNK_X[s] for s in spots]
            clear_of_pads = lambda x, half: all(abs(x - p) >= PAD_HALF + 0.6 + half for p in pads)
            route = [START_X] + pads
            for x0, x1 in zip(route, route[1:]):        # a wall between far-apart points of the route
                if abs(x1 - x0) >= 6:
                    x, width = x0 + (x1 - x0) * rng.uniform(0.45, 0.55), rng.uniform(0.5, 0.9)
                    height = rng.uniform(*lv.wall_height)
                    if clear_of_pads(x, width / 2):
                        plan.append(("wall", x, width, height))
            for p in pads:                               # one or two rocks right next to each pad
                for side in rng.permutation([-1, 1])[: int(rng.integers(1, 3))]:
                    x, size, turn = p + side * rng.uniform(2.6, 4.2), rng.uniform(0.3, 0.55), rng.uniform(0, 2 * math.pi)
                    if clear_of_pads(x, size) and 0.6 <= x <= W - 0.6:
                        plan.append(("rock", x, size, turn))
        return spots, plan

    def _build_obstacles(self, plan):
        self.obstacles = []      # polygons in world units, for drawing
        if not plan:
            self.obstacle_body = None
            return
        self.obstacle_body = self.world.CreateStaticBody(position=(0, 0))
        for kind, x, size, extra in plan:
            if kind == "wall":
                top = min(self._ground_y(x) + extra, H - 3.0)
                verts = [(x - size / 2, self._ground_y(x) - 0.6), (x + size / 2, self._ground_y(x) - 0.6),
                         (x + size / 2, top), (x - size / 2, top)]
            else:
                y = self._ground_y(x)
                verts = [(x + size * math.cos(extra + k * 2 * math.pi / 6) * (1.0 if k % 2 else 0.8),
                          y + size * math.sin(extra + k * 2 * math.pi / 6) * (1.0 if k % 2 else 0.8))
                         for k in range(6)]
            self.obstacle_body.CreateFixture(fixtureDef(shape=polygonShape(vertices=verts), density=0, friction=0.5,
                                                        categoryBits=0x0001 | OBSTACLE_BIT, maskBits=0xFFFF))
            self.obstacles.append(verts)
        # Points every 10 cm around every obstacle's outline: the nearest point of a solid shape to anything
        # outside it is always on its outline, so these are all the sensors need.
        pts = []
        for verts in self.obstacles:
            for (x0, y0), (x1, y1) in zip(verts, verts[1:] + verts[:1]):
                n = max(1, int(math.hypot(x1 - x0, y1 - y0) / 0.1))
                pts += [(x0 + (x1 - x0) * k / n, y0 + (y1 - y0) * k / n) for k in range(n)]
        self.obstacle_points = np.array(pts)

    # ---- Reset: the original game's reset, plus the layout ------------------------------------------
    def reset(self, *, seed=None, options=None):
        gym.Env.reset(self, seed=seed)
        self._destroy()
        self.world = Box2D.b2World(gravity=(0, self.gravity))
        self.world.contactListener_keepref = _Contacts(self)
        self.world.contactListener = self.world.contactListener_keepref
        self.game_over = False
        self.hit_obstacle = False
        self.prev_shaping = None
        self.particles = []
        self.legs = []

        # The original game's random draws, in the original order
        height = self.np_random.uniform(0, H / 2, size=(CHUNKS + 1,))
        push = (self.np_random.uniform(-ll.INITIAL_RANDOM, ll.INITIAL_RANDOM),
                self.np_random.uniform(-ll.INITIAL_RANDOM, ll.INITIAL_RANDOM))
        if self.enable_wind:
            self.wind_idx = self.np_random.integers(-9999, 9999)
            self.torque_idx = self.np_random.integers(-9999, 9999)
        spots, plan = self._draw_layout()

        # Ground: flat around every pad, then smoothed (as in the original)
        self.helipad_y = H / 4
        for s in spots:
            for k in range(-2, 3):
                height[s + k] = self.helipad_y
        self.smooth_y = [0.33 * (height[i - 1] + height[i + 0] + height[i + 1]) for i in range(CHUNKS)]
        self.moon = self.world.CreateStaticBody(shapes=edgeShape(vertices=[(0, 0), (W, 0)]))
        self.sky_polys = []
        for i in range(CHUNKS - 1):
            p1, p2 = (CHUNK_X[i], self.smooth_y[i]), (CHUNK_X[i + 1], self.smooth_y[i + 1])
            self.moon.CreateEdgeFixture(vertices=[p1, p2], density=0, friction=0.1)
            self.sky_polys.append([p1, p2, (p2[0], H), (p1[0], H)])
        self.moon.color1 = self.moon.color2 = (0.0, 0.0, 0.0)
        self.pads = [CHUNK_X[s] for s in spots]          # pad centres, in mission order
        self.phase = 0
        self.taken_off = False
        self.airborne_steps = 0
        self.waiting_steps = 0
        self.touched_pad_a = False
        self._set_target()

        # The lander and its legs (original code)
        self.lander = self.world.CreateDynamicBody(
            position=(START_X, START_Y), angle=0.0,
            fixtures=fixtureDef(shape=polygonShape(vertices=[(x / ll.SCALE, y / ll.SCALE) for x, y in ll.LANDER_POLY]),
                                density=5.0, friction=0.1, categoryBits=0x0010, maskBits=0x001, restitution=0.0))
        self.lander.color1, self.lander.color2 = (128, 102, 230), (77, 77, 128)
        self.lander.ApplyForceToCenter(push, True)
        for i in [-1, +1]:
            leg = self.world.CreateDynamicBody(
                position=(START_X - i * ll.LEG_AWAY / ll.SCALE, START_Y), angle=(i * 0.05),
                fixtures=fixtureDef(shape=polygonShape(box=(ll.LEG_W / ll.SCALE, ll.LEG_H / ll.SCALE)),
                                    density=1.0, restitution=0.0, categoryBits=0x0020, maskBits=0x001))
            leg.ground_contact = False
            leg.color1, leg.color2 = (128, 102, 230), (77, 77, 128)
            rjd = revoluteJointDef(bodyA=self.lander, bodyB=leg, localAnchorA=(0, 0),
                                   localAnchorB=(i * ll.LEG_AWAY / ll.SCALE, ll.LEG_DOWN / ll.SCALE),
                                   enableMotor=True, enableLimit=True, maxMotorTorque=ll.LEG_SPRING_TORQUE,
                                   motorSpeed=+0.3 * i)
            rjd.lowerAngle, rjd.upperAngle = (0.9 - 0.5, 0.9) if i == -1 else (-0.9, -0.9 + 0.5)
            leg.joint = self.world.CreateJoint(rjd)
            self.legs.append(leg)
        if self.practice and self.level.mission and self.np_random.random() < self.level.practice_parked_start:
            self._park_on_pad_a()
        self._build_obstacles(plan)       # after the lander, so the lander's physics order is unchanged
        self.drawlist = [self.lander] + self.legs

        self.fuel_used = getattr(self, "parked_fuel_used", 0.0) if self.phase == 1 else 0.0
        self.fuel_left = self.level.fuel_tank - self.fuel_used
        self.parked_fuel_used = 0.0
        self.steps = 0
        if self.render_mode == "human":
            self.render()
        return self.step(0)[0], {}

    def _park_on_pad_a(self):
        """PRACTICE ONLY (levels.py, practice_parked_start): start the mission as if pad A were already done.
        The lander is placed just above pad A, still, with the target switched to pad B and part of the tank
        used, so practice sees the "take off now" moment far more often."""
        x, y = self.pads[0], self.helipad_y + 0.95       # legs just clear of the pad; it settles in a moment
        self.world.ClearForces()                         # no random push for a parked start
        self.lander.position, self.lander.angle = (x, y), 0.0
        self.lander.linearVelocity, self.lander.angularVelocity = (0, 0), 0.0
        for i, leg in zip([-1, +1], self.legs):
            leg.position, leg.angle = (x - i * ll.LEG_AWAY / ll.SCALE, y), i * 0.05
            leg.linearVelocity, leg.angularVelocity = (0, 0), 0.0
        self.phase = 1
        self._set_target()
        self.parked_fuel_used = float(self.np_random.uniform(80, 130))   # as if the first landing used it

    def _set_target(self):
        self.target_x = self.pads[min(self.phase, len(self.pads) - 1)]
        self.helipad_x1, self.helipad_x2 = self.target_x - PAD_HALF, self.target_x + PAD_HALF

    def _destroy(self):
        if self.obstacle_body is not None and self.moon:
            self.world.DestroyBody(self.obstacle_body)
            self.obstacle_body = None
        super()._destroy()

    # ---- Sensing --------------------------------------------------------------------------------------
    def legs_on_ground(self):
        """Legs (0-2) physically touching the ground, asked from the physics engine (not the buggy flags)."""
        return sum(any(e.contact.touching for e in leg.contacts) for leg in self.legs)

    def _rays(self):
        """Per sensor slice: closeness of the nearest obstacle, 0 = nothing within RAY_LENGTH, 1 = touching.
        Each point of an obstacle belongs to the sensor whose centre direction is closest to it."""
        if self.obstacle_body is None:
            return [0.0] * N_RAYS
        p = self.lander.position
        d = self.obstacle_points - (p[0], p[1])
        dist = np.hypot(d[:, 0], d[:, 1])
        near = dist < RAY_LENGTH
        out = [0.0] * N_RAYS
        if not near.any():
            return out
        angle = np.degrees(np.arctan2(d[near, 0], -d[near, 1]))           # 0 = straight down, + = right
        gap = np.abs((angle[:, None] - SENSOR_DIRECTIONS[None, :] + 180) % 360 - 180)
        sensor = gap.argmin(axis=1)
        closeness = 1.0 - dist[near] / RAY_LENGTH
        for k in range(N_RAYS):
            mine = closeness[sensor == k]
            if mine.size:
                out[k] = float(mine.max())
        return out

    def _observe(self):
        pos, vel = self.lander.position, self.lander.linearVelocity
        return [
            (pos.x - self.target_x) / (W / 2),
            (pos.y - (self.helipad_y + ll.LEG_DOWN / ll.SCALE)) / (H / 2),
            vel.x * (W / 2) / ll.FPS,
            vel.y * (H / 2) / ll.FPS,
            self.lander.angle,
            20.0 * self.lander.angularVelocity / ll.FPS,
            1.0 if self.legs[0].ground_contact else 0.0,
            1.0 if self.legs[1].ground_contact else 0.0,
            1.0 - self.fuel_left / self.level.fuel_tank,
            (self.target_x - W / 2) / (W / 2),
            *self._rays(),
            float(self.phase),
        ]

    # ---- Step: the original game's step, plus fuel, moving target, obstacles and missions -----------
    def step(self, action):
        action = int(action)
        cost = FUEL_COST[action]
        if cost > 0 and self.fuel_left <= 0:
            action, cost = 0, 0.0                    # empty tank: the engines don't respond
        self.fuel_left = max(0.0, self.fuel_left - cost)
        self.fuel_used += cost

        if self.enable_wind and not (self.legs[0].ground_contact or self.legs[1].ground_contact):
            wind_mag = math.tanh(math.sin(0.02 * self.wind_idx) + math.sin(math.pi * 0.01 * self.wind_idx)) * self.wind_power
            self.wind_idx += 1
            self.lander.ApplyForceToCenter((wind_mag, 0.0), True)
            torque_mag = (math.tanh(math.sin(0.02 * self.torque_idx) + math.sin(math.pi * 0.01 * self.torque_idx))
                          * self.turbulence_power)
            self.torque_idx += 1
            self.lander.ApplyTorque(torque_mag, True)

        tip = (math.sin(self.lander.angle), math.cos(self.lander.angle))
        side = (-tip[1], tip[0])
        dispersion = [self.np_random.uniform(-1.0, +1.0) / ll.SCALE for _ in range(2)]
        main_power, side_power = ll.MAIN_ENGINE_POWER, ll.SIDE_ENGINE_POWER   # read now: PhysicsConditions scales them

        m_power = 0.0
        if action == 2:
            m_power = 1.0
            ox = tip[0] * (ll.MAIN_ENGINE_Y_LOCATION / ll.SCALE + 2 * dispersion[0]) + side[0] * dispersion[1]
            oy = -tip[1] * (ll.MAIN_ENGINE_Y_LOCATION / ll.SCALE + 2 * dispersion[0]) - side[1] * dispersion[1]
            impulse_pos = (self.lander.position[0] + ox, self.lander.position[1] + oy)
            if self.render_mode is not None:
                p = self._create_particle(3.5, impulse_pos[0], impulse_pos[1], m_power)
                p.ApplyLinearImpulse((ox * main_power * m_power, oy * main_power * m_power), impulse_pos, True)
            self.lander.ApplyLinearImpulse((-ox * main_power * m_power, -oy * main_power * m_power), impulse_pos, True)

        s_power = 0.0
        if action in (1, 3):
            direction = action - 2
            s_power = 1.0
            ox = tip[0] * dispersion[0] + side[0] * (3 * dispersion[1] + direction * ll.SIDE_ENGINE_AWAY / ll.SCALE)
            oy = -tip[1] * dispersion[0] - side[1] * (3 * dispersion[1] + direction * ll.SIDE_ENGINE_AWAY / ll.SCALE)
            impulse_pos = (self.lander.position[0] + ox - tip[0] * 17 / ll.SCALE,
                           self.lander.position[1] + oy + tip[1] * ll.SIDE_ENGINE_HEIGHT / ll.SCALE)
            if self.render_mode is not None:
                p = self._create_particle(0.7, impulse_pos[0], impulse_pos[1], s_power)
                p.ApplyLinearImpulse((ox * side_power * s_power, oy * side_power * s_power), impulse_pos, True)
            self.lander.ApplyLinearImpulse((-ox * side_power * s_power, -oy * side_power * s_power), impulse_pos, True)

        self.world.Step(1.0 / ll.FPS, 6 * 30, 2 * 30)
        self.steps += 1
        state = self._observe()

        # The official score, measured to the target pad
        reward = 0
        legs_score = 10 * state[6] + 10 * state[7]
        if self.level.mission and self.phase == 1:
            legs_score = 0          # after pad A: legs on the ground are no longer the goal, so taking off isn't -20
        shaping = (-100 * np.sqrt(state[0] ** 2 + state[1] ** 2) - 100 * np.sqrt(state[2] ** 2 + state[3] ** 2)
                   - 100 * abs(state[4]) + legs_score)
        if self.level.mission and self.phase == 1 and not self.taken_off:
            # Waiting on pad A: the distance/speed/tilt score is paused (it would charge ~-2 points for every
            # puff of upward speed, so trying to lift off always looked bad), and restarts once airborne.
            self.prev_shaping = None
        else:
            if self.prev_shaping is not None:
                reward = shaping - self.prev_shaping
            self.prev_shaping = shaping
        reward -= m_power * 0.30 + s_power * 0.03

        on_target = abs(state[0]) <= PAD_HALF / (W / 2)
        terminated = False
        self.event = None
        self.x_screen = (self.lander.position.x - W / 2) / (W / 2)
        if self.game_over or abs(self.x_screen) >= 1.0:
            terminated, reward = True, -100
        elif self.level.mission and self.phase == 1 and not self.taken_off:
            self.waiting_steps += 1
            if self.legs_on_ground() == 0:
                self.airborne_steps += self.touched_pad_a      # only counts once it has stood on pad A
                self.taken_off = self.airborne_steps >= 10
            else:
                self.airborne_steps, self.touched_pad_a = 0, True
            if not self.taken_off and self.waiting_steps >= TAKEOFF_WINDOW:
                # Sat on pad A too long: the mission failed. Scored like a crash: if giving up cost nothing,
                # never taking off would be the safest choice (the first try at Level 4 learned exactly that).
                terminated, reward = True, -100
        elif not self.lander.awake:                  # came to a complete rest
            if self.level.mission and self.phase == 0:
                if on_target:                        # pad A done: next target is pad B
                    self.phase, self.taken_off, self.airborne_steps, self.waiting_steps = 1, False, 0, 0
                    self.touched_pad_a = True
                    self._set_target()
                    self.prev_shaping = None
                    reward = 100
                    self.event = "first_pad"
                    state = self._observe()
                else:
                    terminated, reward = True, 0
            else:
                terminated = True
                reward = 100 if (on_target or not self.level.mission) else 0

        if self.render_mode == "human":
            self.render()
        return np.array(state, dtype=np.float32), reward, terminated, False, {}

    def outcome(self, obs):
        """How the landing (or mission) ended, judged by the physics engine."""
        mission_b = self.level.mission and self.phase == 1
        if self.game_over:
            return "hit_obstacle" if self.hit_obstacle else "crashed"
        if abs(self.x_screen) >= 1.0:                # off the side of the screen
            return "flew_away"
        if mission_b and not self.taken_off:
            return "stuck_on_first_pad"
        if self.lander.awake:
            return "out_of_time"                     # still flying (or fidgeting) when time ran out
        if self.level.mission and self.phase == 0:
            return "landed_off_pad"                  # came to rest, but not on pad A
        return "landed" if abs(obs[0]) <= PAD_HALF / (W / 2) else "landed_off_pad"
