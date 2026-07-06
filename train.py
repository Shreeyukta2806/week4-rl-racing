"""
train.py  –  Train the racing agent with PPO on the straight-road environment.

Run:
    python train.py

Takes ~3-6 minutes on CPU. Saves model to models/ppo_racer.zip.
"""

import os, time
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback

from env.racing_env import RacingEnv
import config as cfg

os.makedirs("models/checkpoints", exist_ok=True)
os.makedirs("logs", exist_ok=True)


class ProgressCallback(BaseCallback):
    """Prints average episode reward every 10 000 steps."""

    def __init__(self, log_every=10_000):
        super().__init__()
        self.log_every  = log_every
        self.ep_rewards = []
        self.last_log   = 0

    def _on_step(self):
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self.ep_rewards.append(info["episode"]["r"])
        if self.num_timesteps - self.last_log >= self.log_every:
            if self.ep_rewards:
                avg = sum(self.ep_rewards) / len(self.ep_rewards)
                print(f"  [step {self.num_timesteps:>7}]  avg reward (last {len(self.ep_rewards)} eps): {avg:.2f}")
                self.ep_rewards = []
            self.last_log = self.num_timesteps
        return True


def main():
    print("=" * 55)
    print("  Week 4 CEAM – RL Racing Agent Training")
    print("  Straight road + barricade obstacles")
    print("=" * 55)

    env = Monitor(RacingEnv(render_mode=None), "logs/monitor")

    model = PPO(
        policy          = "MlpPolicy",
        env             = env,
        learning_rate   = cfg.LEARNING_RATE,
        n_steps         = cfg.N_STEPS,
        batch_size      = cfg.BATCH_SIZE,
        n_epochs        = cfg.N_EPOCHS,
        gamma           = cfg.GAMMA,
        verbose         = 0,
        tensorboard_log = "logs/tensorboard",
    )

    print(f"\nObservation dim : {env.observation_space.shape[0]}")
    print(f"Action space    : {env.action_space.n} (left / straight / right)")
    print(f"Total timesteps : {cfg.TRAIN_TIMESTEPS:,}")
    print("\nTraining… (prints every 10 000 steps)\n")

    checkpoint_cb = CheckpointCallback(
        save_freq   = 50_000,
        save_path   = "models/checkpoints/",
        name_prefix = "ppo_racer",
        verbose     = 0,
    )

    t0 = time.time()
    model.learn(
        total_timesteps = cfg.TRAIN_TIMESTEPS,
        callback        = [checkpoint_cb, ProgressCallback()],
    )
    elapsed = time.time() - t0

    model.save(cfg.MODEL_PATH)
    env.close()

    print(f"\nDone!  Time: {elapsed:.0f}s")
    print(f"Model saved → {cfg.MODEL_PATH}.zip")
    print("Run  py evaluate.py  to watch the trained agent.")


if __name__ == "__main__":
    main()
