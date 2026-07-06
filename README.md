# Week 4 CEAM – RL Racing Agent

A 2D autonomous racing agent trained with **Proximal Policy Optimization (PPO)** from scratch — no prior knowledge of the track, no human-provided labels. The agent starts completely random and learns through trial, error, and reward signals.

---

## Results

| | Untrained (random) | Trained (150k steps) |
|---|---|---|
| Avg episode reward | −6.81 | **230.20** |
| Obstacles cleared | 1 / 80 | **80 / 80** |
| Training time | — | ~4 minutes (CPU) |

---

## Environment

**Track:** Straight vertical road, 190px wide. World scrolls upward as the car advances.

**Obstacles:** Orange-black barricades and golden haystacks alternate along the road. Each has a randomly-placed 90px gap the agent must steer through.

**Nitro pickups:** Cyan lightning bolt pickups placed at the center of every 3rd gap. Collecting gives +1.5 reward and a 45-step visual boost effect.

**Observation (9 floats):**
- `car_x_norm` — lateral position on road [0=left, 1=right]
- `dist_left, dist_right` — distance to each wall
- `obs1_dist, obs1_gap_rel` — distance + gap position of nearest barricade
- `obs2_dist, obs2_gap_rel` — 2nd nearest
- `obs3_dist, obs3_gap_rel` — 3rd nearest

**Actions (discrete):** 0 = steer left · 1 = straight · 2 = steer right

**Reward:**
```
+2.0  cleared a barricade or haystack
+1.5  collected a nitro pickup
+0.01 survived each step
−10.0 crashed into wall or barricade
```

---

## Visual Features

| Feature | Description |
|---|---|
| Day/night cycle | Sky transitions: Sunrise → Day → Sunset → Night as distance increases |
| Rainbow | Appears during early sunrise phase |
| Stars + moon | Visible during night phase |
| Headlight cones | Player and competitor cars illuminate road ahead at night |
| Competing cars | 3 opponents (red, green, yellow) at different speeds — purely visual |
| Moving cows | Animated cows walk on grass alongside the road |
| Audience crowd | Spectators with posters line both sides of the road |
| YOU label | Yellow label above the player's blue car for easy identification |

---

## Design Choices

| Decision | Chosen | Why not alternative |
|---|---|---|
| Algorithm | PPO | DQN only handles discrete actions cleanly; PPO works for both |
| Library | Stable-Baselines3 | Writing PPO from scratch shifts focus away from env design |
| Environment | Custom Gymnasium + Pygame | Pre-built envs hide the observation/reward decisions |
| Collision | Geometric (not pixel-based) | Pixel checks fail at high speed — car skips wall pixels between frames |

---

## Reward hacking example

First version used only survival reward (+0.01/step) and crash penalty (−10). The agent learned to drive in tight circles near the start — technically maximising the reward, just not solving the problem. Fix: ordered checkpoint/barricade rewards that require forward progress.

---

## Setup

```bash
pip install gymnasium stable-baselines3 pygame-ce numpy tensorboard

# Train (~4 min)
py train.py

# Watch trained agent
py evaluate.py

# Compare random vs trained
py evaluate.py --untrained
```

---

## Project Structure

```
week4-rl-racing/
├── env/
│   ├── __init__.py
│   └── racing_env.py      ← Custom Gymnasium environment (all visual + RL logic)
├── train.py               ← PPO training script
├── evaluate.py            ← Load model + render
├── config.py              ← All hyperparameters in one place
├── models/                ← Saved model (auto-created on train)
├── logs/                  ← Monitor CSVs + TensorBoard logs
└── requirements.txt
```

---

*CEAM Computer Vision and AI Subsystem | Week 4 | July 2026*
