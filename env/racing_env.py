"""
Straight-road racing env v3 — full racing atmosphere.

Observation (9 floats, UNCHANGED — no retraining needed):
  [0] car_x_norm  [1] dist_left  [2] dist_right
  [3-4] obs1_dist, obs1_gap_rel   [5-6] obs2   [7-8] obs3

Actions: 0=steer left  1=straight  2=steer right

Visual features:
  Day/night sky cycle · rainbow at sunrise · stars + moon at night
  Moving cows · stable audience crowd with posters · 3 competing cars
  Haystacks (alternate barricade) · nitro pickups · headlight cones at night
  "YOU" badge above player car · proper tree rendering
"""
import math
import gymnasium as gym
import numpy as np
import pygame
from gymnasium import spaces
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import config as cfg

_SHIRT  = [(218,55,55),(55,175,55),(55,55,215),(205,170,40),(160,55,160),(55,195,195),(238,115,55)]
_POSTER = [(255,25,25),(25,255,25),(255,255,25),(255,125,25),(185,25,255),(25,175,255)]
_CB     = [(208,32,32),(32,155,32),(205,182,28)]    # competitor body colours
_CW     = [(255,100,100),(100,215,100),(255,238,100)] # competitor windshield


class RacingEnv(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 60}

    def __init__(self, render_mode=None):
        super().__init__()
        self.action_space = spaces.Discrete(3)
        low  = np.array([0.,0.,0.,0.,-1.,0.,-1.,0.,-1.], dtype=np.float32)
        high = np.array([1.,1.,1.,1., 1.,1., 1.,1., 1.], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        self.render_mode = render_mode
        self.window   = None
        self.clock    = None
        self._hl_surf = None   # headlight cone surface, created lazily

        self.road_left  = cfg.ROAD_LEFT
        self.road_right = cfg.ROAD_LEFT + cfg.ROAD_WIDTH
        self.road_cx    = (self.road_left + self.road_right) // 2

        # fixed star positions for night sky
        rng0 = np.random.default_rng(99)
        self._stars = [(int(rng0.integers(0, cfg.WINDOW_WIDTH)),
                        int(rng0.integers(0, cfg.SKY_HEIGHT))) for _ in range(80)]

        self.car_x = self.scroll_y = 0.
        self.steps = self.score = 0
        self._boost_timer = 0
        self.obstacles = self.nitros = self.bg_trees = []
        self.bg_houses = self.bg_cows = self.competitors = []
        self.bg_crowd  = []          # pre-generated; populated in reset()

    # ──────────────────────────────────────────────────────────────────────────
    # World generation
    # ──────────────────────────────────────────────────────────────────────────

    def _gen_obstacles(self, rng):
        margin = cfg.GAP_WIDTH // 2 + 6
        out = []
        for i in range(cfg.N_OBSTACLES_TOTAL):
            wy     = (i + 1) * cfg.OBSTACLE_SPACING
            gap_cx = int(rng.integers(self.road_left + margin, self.road_right - margin))
            out.append({'world_y': float(wy),
                        'gap_l':   float(gap_cx - cfg.GAP_WIDTH // 2),
                        'gap_r':   float(gap_cx + cfg.GAP_WIDTH // 2),
                        'type':    'haystack' if i % 2 else 'barricade',
                        'passed':  False})
        return out

    def _gen_nitros(self):
        out = []
        for i, obs in enumerate(self.obstacles):
            if i % 3 == 0:
                cx = (obs['gap_l'] + obs['gap_r']) / 2.0
                out.append({'world_y': obs['world_y'] - 28., 'x': cx, 'collected': False})
        return out

    def _gen_trees(self, rng, max_wy):
        out = []
        for _ in range(cfg.N_TREES):
            wy   = float(rng.integers(50, int(max_wy)))
            side = int(rng.choice(np.array([-1, 1])))
            if side == -1 and self.road_left - 10 > 10:
                x = float(rng.integers(10, self.road_left - 10))
            elif side == 1 and self.road_right + 10 < cfg.WINDOW_WIDTH - 10:
                x = float(rng.integers(self.road_right + 10, cfg.WINDOW_WIDTH - 10))
            else:
                continue
            out.append({'world_y': wy, 'x': x,
                        'r': int(rng.integers(12, 24)), 'g': int(rng.integers(80, 172))})
        return out

    def _gen_houses(self, rng, max_wy):
        hc = [(220,175,130),(200,148,120),(175,200,158),(210,188,168),(155,178,210)]
        rc = [(140,60,60),(158,78,58),(98,118,78),(118,98,68),(78,88,128)]
        out = []
        for _ in range(cfg.N_HOUSES):
            wy   = float(rng.integers(100, int(max_wy)))
            side = int(rng.choice(np.array([-1, 1])))
            w = int(rng.integers(30, 52)); h = int(rng.integers(25, 44))
            if side == -1:
                hi = self.road_left - w - 15
                if hi <= 5: continue
                x = float(rng.integers(5, hi))
            else:
                lo = self.road_right + 15; hi = cfg.WINDOW_WIDTH - w - 5
                if lo >= hi: continue
                x = float(rng.integers(lo, hi))
            ci = int(rng.integers(0, len(hc)))
            out.append({'world_y': wy, 'x': x, 'w': w, 'h': h,
                        'color': hc[ci], 'roof': rc[ci]})
        return out

    def _gen_cows(self, rng, max_wy):
        # FIX: honour cfg.N_COWS if the user set it; fall back to 30 if not
        n_cows = getattr(cfg, 'N_COWS', 30)
        out = []
        for _ in range(n_cows):
            wy   = float(rng.integers(200, int(max_wy)))
            side = int(rng.choice(np.array([-1, 1])))
            lo   = 20 if side == -1 else self.road_right + 30
            hi   = max(21, self.road_left - 30) if side == -1 else cfg.WINDOW_WIDTH - 20
            if lo >= hi: continue
            x  = float(rng.integers(lo, hi))
            dx = float(rng.uniform(0.15, 0.35)) * (1 if rng.integers(0, 2) else -1)
            out.append({'world_y': wy, 'x': x, 'dx': dx, 'side': side,
                        'phase': float(rng.uniform(0, 6.28))})
        return out

    def _gen_crowd(self, rng, max_wy):
        """Pre-generate crowd members once per episode so positions are fixed.

        The old approach recomputed people every frame via a world_y hash.
        Because scroll_y jumps by CAR_SPEED (6 px) each step, the entire set
        of world_y values shifts every frame → everyone's arm state flickered
        randomly (the 'shaking' bug).  Storing fixed positions here and using
        a per-person sine phase for arm animation eliminates that completely.
        """
        crowd = []
        y = 240.0
        while y < max_wy:
            cluster = int(rng.integers(3, 7))
            for _ in range(cluster):
                wy = y + float(rng.uniform(-18, 18))
                # left-side fan
                lx = self.road_left - int(rng.integers(16, 28))
                if lx > 6:
                    crowd.append({
                        'world_y': wy,
                        'x':       float(lx) + float(rng.uniform(-3, 3)),
                        'shirt':   _SHIRT[int(rng.integers(0, len(_SHIRT)))],
                        'poster':  _POSTER[int(rng.integers(0, len(_POSTER)))]
                                   if int(rng.integers(0, 4)) == 0 else None,
                        'phase':   float(rng.uniform(0, 6.283)),
                    })
                # right-side fan
                rx = self.road_right + int(rng.integers(16, 28))
                if rx < cfg.WINDOW_WIDTH - 6:
                    crowd.append({
                        'world_y': wy + float(rng.uniform(-5, 5)),
                        'x':       float(rx) + float(rng.uniform(-3, 3)),
                        'shirt':   _SHIRT[int(rng.integers(0, len(_SHIRT)))],
                        'poster':  _POSTER[int(rng.integers(0, len(_POSTER)))]
                                   if int(rng.integers(0, 4)) == 0 else None,
                        'phase':   float(rng.uniform(0, 6.283)),
                    })
            y += float(rng.uniform(360, 540))
        return crowd

    def _gen_competitors(self, rng):
        out = []
        for i, (offset, speed, xc) in enumerate([
            (-70,  5.8, self.road_left  + 28),
            ( 55,  6.3, self.road_right - 28),
            (-260, 5.5, self.road_cx),
        ]):
            out.append({'world_y': float(offset), 'x_center': float(xc), 'x': float(xc),
                        'speed': speed, 'ci': i, 'phase': float(rng.uniform(0, 6.28))})
        return out

    # ──────────────────────────────────────────────────────────────────────────
    # Observation
    # ──────────────────────────────────────────────────────────────────────────

    def _get_obs(self):
        norm = (self.car_x - self.road_left) / cfg.ROAD_WIDTH
        feats, n = [], 0
        for obs in self.obstacles:
            if obs['passed']: continue
            fwd = obs['world_y'] - self.scroll_y
            if fwd < 0: continue
            cx = (obs['gap_l'] + obs['gap_r']) / 2.
            feats.extend([float(np.clip(fwd / cfg.MAX_LOOK_AHEAD, 0, 1)),
                           float(np.clip((cx - self.car_x) / cfg.ROAD_WIDTH, -1, 1))])
            n += 1
            if n == 3: break
        while n < 3:
            feats.extend([1., 0.]); n += 1
        return np.array([norm, norm, 1. - norm] + feats, dtype=np.float32)

    # ──────────────────────────────────────────────────────────────────────────
    # Gymnasium API
    # ──────────────────────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        rng = np.random.default_rng(seed)
        self.car_x = float(self.road_cx)
        self.scroll_y = 0.; self.steps = 0; self.score = 0
        self._boost_timer = 0
        max_wy = cfg.N_OBSTACLES_TOTAL * cfg.OBSTACLE_SPACING + 500
        self.obstacles   = self._gen_obstacles(rng)
        self.nitros      = self._gen_nitros()
        self.bg_trees    = self._gen_trees(rng, max_wy)
        self.bg_houses   = self._gen_houses(rng, max_wy)
        self.bg_cows     = self._gen_cows(rng, max_wy)
        self.bg_crowd    = self._gen_crowd(rng, max_wy)   # ← pre-generated crowd
        self.competitors = self._gen_competitors(rng)
        return self._get_obs(), {}

    def step(self, action):
        self.steps    += 1
        self.scroll_y += cfg.CAR_SPEED
        if self._boost_timer > 0: self._boost_timer -= 1

        if action == 0: self.car_x -= cfg.STEER_SPEED
        elif action == 2: self.car_x += cfg.STEER_SPEED
        self.car_x = float(np.clip(self.car_x, self.road_left, self.road_right))

        if (self.car_x - cfg.CAR_HALF_W < self.road_left or
                self.car_x + cfg.CAR_HALF_W > self.road_right):
            return self._get_obs(), cfg.CRASH_PENALTY, True, False, {'score': self.score}

        reward = cfg.ALIVE_REWARD
        for obs in self.obstacles:
            if obs['passed']: continue
            fwd = obs['world_y'] - self.scroll_y
            if abs(fwd) < (cfg.OBSTACLE_HEIGHT / 2 + cfg.CAR_HALF_H):
                if (self.car_x - cfg.CAR_HALF_W < obs['gap_l'] or
                        self.car_x + cfg.CAR_HALF_W > obs['gap_r']):
                    return self._get_obs(), cfg.CRASH_PENALTY, True, False, {'score': self.score}
            if fwd < -cfg.CAR_HALF_H:
                obs['passed'] = True; self.score += 1; reward += cfg.PASS_REWARD

        for n in self.nitros:
            if n['collected']: continue
            if (abs(n['world_y'] - self.scroll_y) < 22 and
                    abs(n['x'] - self.car_x) < 22):
                n['collected'] = True
                reward += cfg.NITRO_REWARD
                self._boost_timer = cfg.NITRO_BOOST_STEPS

        # FIX: clamp each competitor's x strictly inside the road so they never
        # drift onto the grass (the sine swing ±18 px was not previously clamped)
        for c in self.competitors:
            c['world_y'] += c['speed']
            raw_x = c['x_center'] + math.sin(self.steps * 0.04 + c['phase']) * 18
            c['x'] = float(np.clip(raw_x, self.road_left + 11, self.road_right - 11))
            if c['world_y'] < self.scroll_y - (cfg.WINDOW_HEIGHT + 50):
                c['world_y'] = self.scroll_y + 100 + c['ci'] * 90

        for cow in self.bg_cows:
            cow['phase'] = (cow['phase'] + 0.06) % 6.2832
            cow['x'] += cow['dx']
            lo2 = 8 if cow['side'] == -1 else self.road_right + 18
            hi2 = self.road_left - 18 if cow['side'] == -1 else cfg.WINDOW_WIDTH - 8
            if cow['x'] < lo2 or cow['x'] > hi2: cow['dx'] *= -1

        return self._get_obs(), reward, False, self.steps >= cfg.MAX_STEPS, {'score': self.score}

    # ──────────────────────────────────────────────────────────────────────────
    # Colour / sky helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _screen_y(self, wy):
        return int(cfg.CAR_SCREEN_Y - (wy - self.scroll_y))

    def _lerp(self, a, b, t):
        return tuple(int(x + (y - x) * t) for x, y in zip(a, b))

    def _sky_phase(self):
        s = self.scroll_y
        if   s < 1500:  return 'sunrise', s / 1500
        elif s < 8000:  return 'day',     (s - 1500) / 6500
        elif s < 11000: return 'sunset',  (s - 8000) / 3000
        else:           return 'night',   min((s - 11000) / 2000, 1.)

    def _sky_col(self):
        ph, t = self._sky_phase()
        if ph == 'sunrise': return self._lerp((228, 108, 52), (100, 170, 235), t)
        if ph == 'day':     return (100, 170, 235)
        if ph == 'sunset':  return self._lerp((100, 170, 235), (198, 72, 22), t)
        return self._lerp((198, 72, 22), (8, 8, 30), t)

    def _grass_col(self):
        ph, t = self._sky_phase()
        base = (55, 145, 55)
        if ph == 'sunset': return self._lerp(base, (45, 90, 30), t)
        if ph == 'night':  return self._lerp((45, 90, 30), (15, 35, 15), t)
        return base

    def _night_t(self):
        ph, t = self._sky_phase()
        return t if ph == 'night' else 0.

    # ──────────────────────────────────────────────────────────────────────────
    # Draw helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _draw_cow(self, surf, x, y, phase):
        x, y = int(x), int(y + math.sin(phase) * 1.5)
        pygame.draw.ellipse(surf, (192, 152, 92), (x-16, y-7, 32, 14))
        pygame.draw.circle(surf, (192, 152, 92), (x+17, y-5), 7)
        pygame.draw.circle(surf, (28, 28, 28), (x+20, y-7), 2)
        pygame.draw.line(surf, (198, 168, 98), (x+14, y-12), (x+11, y-17), 2)
        pygame.draw.line(surf, (198, 168, 98), (x+20, y-12), (x+23, y-17), 2)
        pygame.draw.ellipse(surf, (58, 38, 18), (x-10, y-6, 9, 7))
        pygame.draw.ellipse(surf, (58, 38, 18), (x+2,  y-4, 7, 5))
        lg = int(math.sin(phase * 2) * 3)
        for lx in [x-10, x-3, x+4, x+10]:
            pygame.draw.line(surf, (152, 112, 72), (lx, y+6), (lx + lg, y+15), 2)

    def _draw_barricade(self, surf, x, y, width):
        if width <= 0: return
        bh = cfg.OBSTACLE_HEIGHT; by = y - bh // 2
        bx, tog = x, False
        while bx < x + width:
            w = min(14, x + width - bx)
            pygame.draw.rect(surf, (28, 28, 28) if tog else (255, 108, 0), (bx, by, w, bh))
            bx += 14; tog = not tog
        pygame.draw.rect(surf, (238, 238, 238), (x, by + bh//2 - 1, width, 3))
        pygame.draw.rect(surf, (198, 78, 0), (x, by, width, bh), 2)

    def _draw_haystack(self, surf, x, y, width):
        if width <= 0: return
        bh = cfg.OBSTACLE_HEIGHT; by = y - bh // 2
        pygame.draw.ellipse(surf, (195, 158, 35), (x, by, width, bh))
        for i in range(1, 4):
            lx = x + width * i // 4
            pygame.draw.line(surf, (152, 115, 25), (lx, by+2), (lx, by+bh-2), 1)
        pygame.draw.ellipse(surf, (152, 115, 25), (x, by, width, bh), 2)
        for i in range(0, width, 10):
            pygame.draw.line(surf, (215, 178, 48), (x+i, by), (x+i+3, by-4), 1)

    def _draw_nitro(self, surf, x, y, steps):
        x, y = int(x), int(y)
        r = int(11 + abs(math.sin(steps * 0.18)) * 4)
        pygame.draw.circle(surf, (0, 190, 255), (x, y), r+3)
        pygame.draw.circle(surf, (180, 248, 255), (x, y), r)
        bolt = [(x+2,y-8),(x-3,y-1),(x+4,y-1),(x-3,y+8),(x+1,y+1),(x-4,y+1)]
        pygame.draw.polygon(surf, (255, 238, 0), bolt)

    def _draw_competitor(self, surf, x, y, ci, night):
        x, y = int(x), int(y)
        cw, ch = 18, 26; xl, yt = x - cw//2, y - ch//2
        pygame.draw.rect(surf, _CB[ci],        (xl, yt, cw, ch),           border_radius=3)
        pygame.draw.rect(surf, _CW[ci],        (xl+3, yt+3, cw-6, 8),      border_radius=2)
        pygame.draw.rect(surf, (100,140,180),  (xl+3, yt+ch-10, cw-6, 6),  border_radius=1)
        hl = (255,255,215) if night else (255,255,175)
        pygame.draw.rect(surf, hl,           (xl+1,    yt+1,    4, 3))
        pygame.draw.rect(surf, hl,           (xl+cw-5, yt+1,    4, 3))
        pygame.draw.rect(surf, (200,30,30),  (xl+1,    yt+ch-4, 4, 3))
        pygame.draw.rect(surf, (200,30,30),  (xl+cw-5, yt+ch-4, 4, 3))

    def _draw_headlight_cone(self, surf, x, y):
        if self._hl_surf is None:
            self._hl_surf = pygame.Surface((84, 88), pygame.SRCALPHA)
            pts = [(42, 85), (42-24, 0), (42+24, 0)]
            pygame.draw.polygon(self._hl_surf, (255, 255, 195, 50), pts)
        surf.blit(self._hl_surf, (int(x) - 42, int(y) - 85))

    def _draw_person(self, surf, x, y, arm_up, sc, pc):
        x, y = int(x), int(y)
        pygame.draw.circle(surf, (245, 210, 175), (x, y-13), 4)
        pygame.draw.rect(surf, sc, (x-3, y-9, 6, 8))
        if arm_up:
            pygame.draw.line(surf, (245,210,175), (x-3,y-7), (x-9,y-14), 2)
            pygame.draw.line(surf, (245,210,175), (x+3,y-7), (x+9,y-14), 2)
            if pc:
                pygame.draw.rect(surf, pc,      (x+9, y-18, 15, 10))
                pygame.draw.rect(surf, (0,0,0), (x+9, y-18, 15, 10), 1)
        else:
            pygame.draw.line(surf, (245,210,175), (x-3,y-7), (x-7,y-3), 2)
            pygame.draw.line(surf, (245,210,175), (x+3,y-7), (x+7,y-3), 2)
        pygame.draw.line(surf, (62,62,108), (x-1,y-1), (x-2,y+6), 2)
        pygame.draw.line(surf, (62,62,108), (x+1,y-1), (x+2,y+6), 2)

    # ──────────────────────────────────────────────────────────────────────────
    # Main render
    # ──────────────────────────────────────────────────────────────────────────

    def render(self):
        if self.render_mode is None: return
        if self.window is None:
            pygame.init()
            if self.render_mode == 'human':
                pygame.display.init()
                self.window = pygame.display.set_mode((cfg.WINDOW_WIDTH, cfg.WINDOW_HEIGHT))
                pygame.display.set_caption('RL Racing Agent v3 – Week 4 CEAM')
                self.font = pygame.font.SysFont('Arial', 18)
            else:
                self.window = pygame.Surface((cfg.WINDOW_WIDTH, cfg.WINDOW_HEIGHT))
            self.clock = pygame.time.Clock()

        surf  = self.window
        sky   = self._sky_col()
        ph, pt = self._sky_phase()
        nt    = self._night_t()
        night = (ph == 'night')

        # ── sky ────────────────────────────────────────────────────────────────
        surf.fill(sky)
        pygame.draw.rect(surf, self._lerp(sky, (255,255,255), 0.22), (0, 0, cfg.WINDOW_WIDTH, 32))

        if night:
            sb = int(200 * pt)
            for sx, sy in self._stars:
                r = 1 if (sx + sy) % 3 == 0 else 0
                pygame.draw.circle(surf, (sb, sb, sb), (sx, sy), r+1)
            pygame.draw.circle(surf, (240, 240, 200), (cfg.WINDOW_WIDTH-80, 30), 16)
            # crescent shadow
            pygame.draw.circle(surf, sky,             (cfg.WINDOW_WIDTH-72, 26), 12)
        else:
            if ph == 'sunrise':
                sxp, syp, sc2 = 80+int(pt*200), 55-int(pt*22), (255,168,58)
            elif ph == 'day':
                sxp, syp, sc2 = 280+int(pt*200), 28, (255,240,78)
            else:
                sxp, syp, sc2 = 480+int(pt*160), 30+int(pt*28), (255,118,28)
            pygame.draw.circle(surf, sc2, (sxp, syp), 20)
            pygame.draw.circle(surf, self._lerp(sc2,(255,255,220),0.5), (sxp,syp), 13)

            # clouds
            if ph in ('day','sunrise'):
                cb = int(180+pt*75) if ph=='sunrise' else 255
                cc = (cb, cb, cb)
                for cxp, cyp, cwp in [(110,22,72),(310,15,82),(530,28,66)]:
                    pygame.draw.ellipse(surf, cc, (cxp-cwp//2, cyp-11, cwp, 22))
                    pygame.draw.ellipse(surf, cc, (cxp-cwp//4, cyp-19, int(cwp*.70), 20))
                    pygame.draw.ellipse(surf, cc, (cxp+cwp//4-8, cyp-13, int(cwp*.58), 18))

            # rainbow during early sunrise
            if ph == 'sunrise' and pt < 0.6:
                ra = 1. - pt / 0.6
                rb_cols = [(148,0,211),(75,0,130),(0,0,255),(0,180,0),(255,255,0),(255,127,0),(255,0,0)]
                for i, rc2 in enumerate(rb_cols):
                    r2 = 88 + i * 12
                    dim = self._lerp(rc2, (rc2[0]//3, rc2[1]//3, rc2[2]//3), 1.-ra)
                    pygame.draw.arc(surf, dim, (cfg.WINDOW_WIDTH//2-r2, 10-r2, r2*2, r2*2), 0, math.pi, 5)

        # ── grass ──────────────────────────────────────────────────────────────
        gc = self._grass_col()
        pygame.draw.rect(surf, gc, (0, cfg.SKY_HEIGHT, cfg.WINDOW_WIDTH, cfg.WINDOW_HEIGHT - cfg.SKY_HEIGHT))

        # ── audience — pre-generated, smooth per-person wave (no more jitter) ──
        # Each fan has a fixed world_y and an independent wave phase.  Arms move
        # with sin(steps * 0.055 + phase) so every person waves at a different
        # time → natural-looking crowd animation instead of everyone snapping
        # simultaneously each frame.
        for p in self.bg_crowd:
            sy2 = self._screen_y(p['world_y'])
            if cfg.SKY_HEIGHT < sy2 < cfg.WINDOW_HEIGHT:
                wave   = math.sin(self.steps * 0.055 + p['phase'])
                arm_up = wave > -0.35   # arms raised ~80 % of the cycle
                self._draw_person(surf, p['x'], sy2, arm_up, p['shirt'], p['poster'])

        # ── trees — FIX: visible brown trunk BELOW canopy ─────────────────────
        # Old code drew the trunk downward from sy2 and the canopy centred at
        # sy2 — the trunk was entirely hidden under the bottom half of the
        # circle.  Now sy2 is treated as ground level: trunk rises UP from it,
        # canopy circle sits above the trunk top.
        for t in self.bg_trees:
            sy2 = self._screen_y(t['world_y'])
            if cfg.SKY_HEIGHT - 50 < sy2 < cfg.WINDOW_HEIGHT + 30:
                tx2 = int(t['x'])
                r   = t['r']
                g   = t['g']
                trunk_col = self._lerp((95, 62, 32),  (32, 18,  8), nt * 0.75)
                leaf_shad = self._lerp((18, max(50, g - 35), 12), (5, g // 6, 4), nt)
                leaf_col  = self._lerp((28, g, 22),   (8,  g // 5, 6), nt)
                leaf_hi   = self._lerp((min(255, 55 + g // 2), min(255, g + 28), 20),
                                       (10, g // 4, 8), nt)
                # trunk: rectangle from ground (sy2) going up
                tw, th = 5, r + 5
                pygame.draw.rect(surf, trunk_col,
                                 (tx2 - tw // 2, sy2 - th, tw, th + 3))
                # canopy: centred above the trunk, slight overlap at join
                cy = sy2 - th - r // 2 + 3
                pygame.draw.circle(surf, leaf_shad, (tx2, cy + 2), r)      # shadow base
                pygame.draw.circle(surf, leaf_col,  (tx2, cy),     r - 1)  # main foliage
                if r > 14:
                    pygame.draw.circle(surf, leaf_hi,
                                       (tx2 - r // 4, cy - r // 4), r // 3)  # sunlit highlight

        # ── houses ─────────────────────────────────────────────────────────────
        for h in self.bg_houses:
            sy2 = self._screen_y(h['world_y'])
            if cfg.SKY_HEIGHT-65 < sy2 < cfg.WINDOW_HEIGHT+65:
                hx2, w2, ht2 = int(h['x']), h['w'], h['h']
                hcl = self._lerp(h['color'], tuple(c//4 for c in h['color']), nt*0.75)
                rcl = self._lerp(h['roof'],  tuple(c//4 for c in h['roof']),  nt*0.75)
                pygame.draw.rect(surf, hcl, (hx2, sy2-ht2, w2, ht2))
                pygame.draw.polygon(surf, rcl, [(hx2-3,sy2-ht2),(hx2+w2+3,sy2-ht2),(hx2+w2//2,sy2-ht2-16)])
                pygame.draw.rect(surf, (88,58,35), (hx2+w2//2-3, sy2-11, 7, 11))
                pygame.draw.rect(surf, (255,220,100) if night else (175,215,250), (hx2+4, sy2-ht2+6, 9, 8))

        # ── cows ───────────────────────────────────────────────────────────────
        for cow in self.bg_cows:
            sy2 = self._screen_y(cow['world_y'])
            if cfg.SKY_HEIGHT-30 < sy2 < cfg.WINDOW_HEIGHT+30:
                self._draw_cow(surf, cow['x'], sy2, cow['phase'])

        # ── road ───────────────────────────────────────────────────────────────
        rd = (62,62,68) if night else (78,78,82)
        pygame.draw.rect(surf, rd, (self.road_left, cfg.SKY_HEIGHT, cfg.ROAD_WIDTH, cfg.WINDOW_HEIGHT-cfg.SKY_HEIGHT))
        el = (255,255,255) if night else (230,230,230)
        pygame.draw.line(surf, el, (self.road_left,  cfg.SKY_HEIGHT), (self.road_left,  cfg.WINDOW_HEIGHT), 2)
        pygame.draw.line(surf, el, (self.road_right, cfg.SKY_HEIGHT), (self.road_right, cfg.WINDOW_HEIGHT), 2)
        dh, dg = 35, 25; period = dh + dg
        offset = int(self.scroll_y) % period; y2 = cfg.SKY_HEIGHT - offset
        while y2 < cfg.WINDOW_HEIGHT:
            if y2+dh > cfg.SKY_HEIGHT:
                dy2 = max(y2, cfg.SKY_HEIGHT); ddh = min(y2+dh, cfg.WINDOW_HEIGHT)-dy2
                pygame.draw.rect(surf, (255,255,255), (self.road_cx-2, dy2, 4, ddh))
            y2 += period

        # ── headlight cones (night only, drawn before cars so car sits on top) ─
        if night:
            for c in self.competitors:
                sy2 = self._screen_y(c['world_y'])
                if -10 < sy2 < cfg.WINDOW_HEIGHT+10:
                    self._draw_headlight_cone(surf, c['x'], sy2)
            self._draw_headlight_cone(surf, self.car_x, cfg.CAR_SCREEN_Y)

        # ── obstacles ──────────────────────────────────────────────────────────
        for obs in self.obstacles:
            if obs['passed']: continue
            sy2 = self._screen_y(obs['world_y'])
            if cfg.SKY_HEIGHT-6 < sy2 < cfg.WINDOW_HEIGHT+6:
                lw = int(obs['gap_l']) - self.road_left
                rw = self.road_right - int(obs['gap_r'])
                fn = self._draw_haystack if obs['type'] == 'haystack' else self._draw_barricade
                if lw > 0: fn(surf, self.road_left, sy2, lw)
                if rw > 0: fn(surf, int(obs['gap_r']), sy2, rw)

        # ── nitro pickups ──────────────────────────────────────────────────────
        for n in self.nitros:
            if n['collected']: continue
            sy2 = self._screen_y(n['world_y'])
            if cfg.SKY_HEIGHT-22 < sy2 < cfg.WINDOW_HEIGHT+22:
                self._draw_nitro(surf, n['x'], sy2, int(self.steps))

        # ── competing cars ─────────────────────────────────────────────────────
        for c in self.competitors:
            sy2 = self._screen_y(c['world_y'])
            if -40 < sy2 < cfg.WINDOW_HEIGHT+40:
                self._draw_competitor(surf, c['x'], sy2, c['ci'], night)

        # ── player car ─────────────────────────────────────────────────────────
        cw2, ch2 = 20, 30
        xl2, yt2 = int(self.car_x)-cw2//2, int(cfg.CAR_SCREEN_Y)-ch2//2
        pygame.draw.rect(surf, (30,120,255),   (xl2,yt2,cw2,ch2), border_radius=4)
        pygame.draw.rect(surf, (155,208,255),  (xl2+3,yt2+3,cw2-6,9), border_radius=2)
        pygame.draw.rect(surf, (108,160,208),  (xl2+3,yt2+ch2-11,cw2-6,6), border_radius=1)
        hl2 = (255,255,225) if night else (255,255,175)
        pygame.draw.rect(surf, hl2,           (xl2+1,    yt2+1,    5, 4))
        pygame.draw.rect(surf, hl2,           (xl2+cw2-6,yt2+1,    5, 4))
        pygame.draw.rect(surf, (220,38,38),   (xl2+1,    yt2+ch2-5,5, 4))
        pygame.draw.rect(surf, (220,38,38),   (xl2+cw2-6,yt2+ch2-5,5, 4))
        pygame.draw.line(surf, (18,78,198),   (xl2+cw2//2,yt2+13),(xl2+cw2//2,yt2+ch2-13),2)

        # ── "YOU" badge above player car ───────────────────────────────────────
        if hasattr(self, 'font'):
            you_txt  = self.font.render("YOU", True, (20, 20, 20))
            badge_w  = you_txt.get_width() + 8
            badge_h  = you_txt.get_height() + 4
            badge_x  = int(self.car_x) - badge_w // 2
            badge_y  = yt2 - badge_h - 6
            pygame.draw.rect(surf, (255, 215, 0),
                             (badge_x, badge_y, badge_w, badge_h), border_radius=4)
            pygame.draw.rect(surf, (190, 148, 0),
                             (badge_x, badge_y, badge_w, badge_h), border_radius=4, width=1)
            surf.blit(you_txt, (badge_x + 4, badge_y + 2))
            # small downward arrow linking badge to car roof
            ax = int(self.car_x)
            ay0 = badge_y + badge_h
            pygame.draw.polygon(surf, (255, 215, 0),
                                 [(ax - 4, ay0), (ax + 4, ay0), (ax, ay0 + 5)])

        # ── boost glow + speed lines ────────────────────────────────────────────
        if self._boost_timer > 0:
            pygame.draw.rect(surf, (0,215,195), (xl2-2,yt2-2,cw2+4,ch2+4), border_radius=5, width=2)
            for i in range(7):
                bx2 = self.road_left + i*(cfg.ROAD_WIDTH//7)+8
                pygame.draw.line(surf,(160,255,215),(bx2,yt2+ch2+4),
                                 (bx2,min(cfg.WINDOW_HEIGHT,yt2+ch2+28+i*3)),1)

        # ── HUD ────────────────────────────────────────────────────────────────
        if self.render_mode == 'human' and hasattr(self, 'font'):
            pl  = {'sunrise':'SUNRISE','day':'DAY','sunset':'SUNSET','night':'NIGHT'}
            bst = ' | BOOST!' if self._boost_timer > 0 else ''
            txt = f"Step:{int(self.steps)}  Cleared:{self.score}  Dist:{int(self.scroll_y)}m  {pl[ph]}{bst}"
            hud = self.font.render(txt, True, (255, 255, 255))
            pygame.draw.rect(surf, (0,0,0), (4,4,hud.get_width()+12,26))
            surf.blit(hud, (10, 7))

        if self.render_mode == 'human':
            pygame.display.flip()
            self.clock.tick(self.metadata['render_fps'])
        elif self.render_mode == 'rgb_array':
            return np.transpose(np.array(pygame.surfarray.pixels3d(surf)), axes=(1,0,2))

    def close(self):
        if self.window is not None:
            pygame.display.quit(); pygame.quit(); self.window = None
