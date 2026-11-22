"""Record an mp4 of a trained policy (or of a random policy for comparison).

MuJoCo needs an off-screen OpenGL backend on a headless machine; set it
*before* running, e.g. `MUJOCO_GL=egl python record_video.py ...`
(fallback: `MUJOCO_GL=osmesa`).

    python record_video.py --checkpoint runs/mbpo_seed0/best.pt --out videos/mbpo.mp4
    python record_video.py --random --out videos/random.mp4
"""
import argparse
import os

import gymnasium as gym
import imageio
import numpy as np
import torch

from mbpo.sac import SAC


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--random", action="store_true", help="use a uniformly random policy")
    p.add_argument("--env-id", default="HalfCheetah-v5")
    p.add_argument("--out", default="videos/halfcheetah.mp4")
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=360)
    args = p.parse_args()

    env = gym.make(args.env_id, render_mode="rgb_array", width=args.width, height=args.height)
    obs, _ = env.reset(seed=args.seed)
    env.action_space.seed(args.seed)

    agent = None
    if not args.random:
        ckpt = torch.load(args.checkpoint, map_location="cpu")
        agent = SAC(ckpt["obs_dim"], ckpt["act_dim"], torch.device("cpu"),
                    hidden_dim=ckpt["config"]["hidden_dim"])
        agent.load_state_dict(ckpt["agent"])

    frames, total = [], 0.0
    for _ in range(args.steps):
        frames.append(env.render())
        action = env.action_space.sample() if agent is None else agent.act(obs, deterministic=True)
        obs, r, term, trunc, _ = env.step(action)
        total += r
        if term or trunc:
            break
    env.close()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fps = env.metadata.get("render_fps", 20)
    imageio.mimsave(args.out, [np.asarray(f) for f in frames], fps=fps, macro_block_size=1)
    print(f"saved {args.out}: {len(frames)} frames, return {total:.1f}")


if __name__ == "__main__":
    main()
