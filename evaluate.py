"""
evaluate.py  –  Load a trained model and watch it dodge barricades.

Run:
    py evaluate.py                  # 5 episodes, rendered
    py evaluate.py --untrained      # random agent first, then trained agent
    py evaluate.py --no-render      # headless (for stats only)
"""

import argparse
import numpy as np
import pygame
from stable_baselines3 import PPO

from env.racing_env import RacingEnv
import config as cfg


def run_episodes(model, n_episodes=5, render=True, label="Agent"):
    env = RacingEnv(render_mode="human" if render else None)
    rewards, scores = [], []

    for ep in range(n_episodes):
        obs, _ = env.reset()
        ep_reward, done = 0.0, False

        while not done:
            if model is None:
                action = env.action_space.sample()
            else:
                action, _ = model.predict(obs, deterministic=True)

            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            done = terminated or truncated

            if render:
                env.render()
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        env.close()
                        return rewards, scores

        rewards.append(ep_reward)
        scores.append(info.get("score", 0))
        print(f"  {label}  ep {ep+1:2d}: reward = {ep_reward:8.2f}   obstacles cleared = {info.get('score', 0)}")

    env.close()
    return rewards, scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-render",  action="store_true")
    parser.add_argument("--untrained",  action="store_true")
    parser.add_argument("--episodes",   type=int, default=5)
    args = parser.parse_args()

    render = not args.no_render

    if args.untrained:
        print("\n─── UNTRAINED AGENT (random actions) ───")
        r, s = run_episodes(None, n_episodes=3, render=render, label="Random ")
        print(f"  Average reward            : {np.mean(r):.2f}")
        print(f"  Average obstacles cleared : {np.mean(s):.2f}\n")

    try:
        model = PPO.load(cfg.MODEL_PATH)
    except FileNotFoundError:
        print(f"\nNo trained model found at {cfg.MODEL_PATH}.zip")
        print("Run  py train.py  first.")
        return

    print(f"\n─── TRAINED AGENT ({cfg.MODEL_PATH}.zip) ───")
    r, s = run_episodes(model, n_episodes=args.episodes, render=render, label="Trained")
    print(f"\n  Average reward            : {np.mean(r):.2f}")
    print(f"  Average obstacles cleared : {np.mean(s):.2f}")


if __name__ == "__main__":
    main()
