import torch
import numpy as np
from common_utils import Recorder, Stopwatch
from common_utils import ibrl_utils as utils
from GymnasiumWrapper import create_env

def run_eval_gym(
    agent,
    num_game,
    seed,
    verbose=True,
    eval_mode=True,
) -> list[float]:
    device = "cpu"
    agent = agent.to(device)
    scores = []
    lens = []
    stopwatch = Stopwatch()

    env = create_env()
    with torch.no_grad(), utils.eval_mode(agent):
        for episode_idx in range(num_game):
            step = 0
            rewards = []
            np.random.seed(seed + episode_idx)
            with stopwatch.time("reset"):
                obs, _ = env.reset()

            terminal = False
            while not terminal:

                with stopwatch.time(f"act"):
                    action = agent.act(obs, eval_mode=eval_mode)

                with stopwatch.time("step"):
                    obs, reward, terminal, _, _ = env.step(action)

                rewards.append(reward)
                step += 1

            if verbose:
                print(
                    f"seed: {seed + episode_idx}, "
                    f"reward: {np.sum(rewards)}, steps: {step}"
                )

            scores.append(np.sum(rewards))
            if scores[-1] > 0:
                lens.append(step)

    if verbose:
        print(f"num game: {len(scores)}, seed: {seed}, score: {np.mean(scores)}")
        print(f"average steps for success games: {np.mean(lens)}")
        stopwatch.summary()

    return scores

