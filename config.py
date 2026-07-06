# ─── Training ────────────────────────────────────────────────────────────────
TRAIN_TIMESTEPS = 150_000
LEARNING_RATE   = 3e-4
N_STEPS         = 1024
BATCH_SIZE      = 64
N_EPOCHS        = 10
GAMMA           = 0.99
MODEL_PATH      = "models/ppo_racer"

# ─── Window ───────────────────────────────────────────────────────────────────
WINDOW_WIDTH  = 700
WINDOW_HEIGHT = 580

# ─── Sky ─────────────────────────────────────────────────────────────────────
SKY_HEIGHT = 80

# ─── Road ─────────────────────────────────────────────────────────────────────
ROAD_WIDTH = 190
ROAD_LEFT  = (WINDOW_WIDTH - ROAD_WIDTH) // 2   # = 255
# ROAD_RIGHT = ROAD_LEFT + ROAD_WIDTH            # = 445

# ─── Car ─────────────────────────────────────────────────────────────────────
CAR_SPEED    = 6.0    # world units scrolled per step
STEER_SPEED  = 3.5    # pixels per step laterally
CAR_HALF_W   = 10     # collision half-width
CAR_HALF_H   = 14     # collision half-height
CAR_SCREEN_Y = 430    # car's fixed y position on screen

# ─── Obstacles ────────────────────────────────────────────────────────────────
OBSTACLE_SPACING  = 220    # world-y distance between barricades
N_OBSTACLES_TOTAL = 80     # pre-generated per episode
OBSTACLE_HEIGHT   = 20     # pixel height of barricade
GAP_WIDTH         = 90     # width of the passable gap in each barricade
MAX_LOOK_AHEAD    = 500    # max forward distance used in observations

# ─── Reward ───────────────────────────────────────────────────────────────────
PASS_REWARD   =  2.0
ALIVE_REWARD  =  0.01
CRASH_PENALTY = -10.0

# ─── Background ───────────────────────────────────────────────────────────────
N_TREES  = 70
N_HOUSES = 55

# ─── Episode ──────────────────────────────────────────────────────────────────
MAX_STEPS = 3000

# ─── Nitro ────────────────────────────────────────────────────────────────────
NITRO_REWARD  = 1.5
NITRO_BOOST_STEPS = 45   # steps the visual boost lasts
