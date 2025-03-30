import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    VecEnv,
    VecMonitor,
    is_vecenv_wrapped,
)
from huggingface_sb3 import load_from_hub
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from stable_baselines3.common import type_aliases
import warnings
import gymnasium as gym
import pickle
from tqdm import trange



def generate_offline_data(
    model: "type_aliases.PolicyPredictor",
    env: Union[gym.Env, VecEnv],
    n_eval_episodes: int = 10,
    deterministic: bool = True,
    render: bool = False,
    callback: Optional[Callable[[Dict[str, Any], Dict[str, Any]], None]] = None,
    reward_threshold: Optional[float] = None,
    return_episode_rewards: bool = False,
    warn: bool = True,
) -> Union[Tuple[float, float], Tuple[List[float], List[int]]]:
    """
    Runs policy for ``n_eval_episodes`` episodes and returns average reward.
    If a vector env is passed in, this divides the episodes to evaluate onto the
    different elements of the vector env. This static division of work is done to
    remove bias. See https://github.com/DLR-RM/stable-baselines3/issues/402 for more
    details and discussion.

    .. note::
        If environment has not been wrapped with ``Monitor`` wrapper, reward and
        episode lengths are counted as it appears with ``env.step`` calls. If
        the environment contains wrappers that modify rewards or episode lengths
        (e.g. reward scaling, early episode reset), these will affect the evaluation
        results as well. You can avoid this by wrapping environment with ``Monitor``
        wrapper before anything else.

    :param model: The RL agent you want to evaluate. This can be any object
        that implements a `predict` method, such as an RL algorithm (``BaseAlgorithm``)
        or policy (``BasePolicy``).
    :param env: The gym environment or ``VecEnv`` environment.
    :param n_eval_episodes: Number of episode to evaluate the agent
    :param deterministic: Whether to use deterministic or stochastic actions
    :param render: Whether to render the environment or not
    :param callback: callback function to do additional checks,
        called after each step. Gets locals() and globals() passed as parameters.
    :param reward_threshold: Minimum expected reward per episode,
        this will raise an error if the performance is not met
    :param return_episode_rewards: If True, a list of rewards and episode lengths
        per episode will be returned instead of the mean.
    :param warn: If True (default), warns user about lack of a Monitor wrapper in the
        evaluation environment.
    :return: Mean reward per episode, std of reward per episode.
        Returns ([float], [int]) when ``return_episode_rewards`` is True, first
        list containing per-episode rewards and second containing per-episode lengths
        (in number of steps).
    """
    is_monitor_wrapped = False
    # Avoid circular import
    from stable_baselines3.common.monitor import Monitor

    if not isinstance(env, VecEnv):
        env = DummyVecEnv([lambda: env])

    is_monitor_wrapped = (
        is_vecenv_wrapped(env, VecMonitor) or env.env_is_wrapped(Monitor)[0]
    )

    if not is_monitor_wrapped and warn:
        warnings.warn(
            "Evaluation environment is not wrapped with a ``Monitor`` wrapper. "
            "This may result in reporting modified episode lengths and rewards, if other wrappers happen to modify these. "
            "Consider wrapping environment first with ``Monitor`` wrapper.",
            UserWarning,
        )

    n_envs = env.num_envs
    episode_rewards = []
    episode_lengths = []
    obs_list, actions_list, dones_list, rewards_list, infos_list, next_obs_list = (
        [],
        [],
        [],
        [],
        [],
        [],
    )
    episode_counts = np.zeros(n_envs, dtype="int")
    # Divides episodes among different sub environments in the vector as evenly as possible
    episode_count_targets = np.array(
        [(n_eval_episodes + i) // n_envs for i in range(n_envs)], dtype="int"
    )

    current_rewards = np.zeros(n_envs)
    current_lengths = np.zeros(n_envs, dtype="int")
    observations = env.reset()
    states = None
    episode_starts = np.ones((env.num_envs,), dtype=bool)
    while (episode_counts < episode_count_targets).any():
        obs_list.append(observations[0])

        actions, states = model.predict(
            observations,
            state=states,
            episode_start=episode_starts,
            deterministic=deterministic,
        )
        observations, rewards, dones, infos = env.step(actions)

        actions_list.append(actions[0])
        rewards_list.append(rewards[0])
        dones_list.append(dones[0])
        infos_list.append(infos[0])
        next_obs_list.append(observations[0])

        current_rewards += rewards
        current_lengths += 1
        for i in range(n_envs):
            if episode_counts[i] < episode_count_targets[i]:
                # unpack values so that the callback can access the local variables
                reward = rewards[i]
                done = dones[i]
                info = infos[i]
                episode_starts[i] = done

                if callback is not None:
                    callback(locals(), globals())

                if dones[i]:
                    if is_monitor_wrapped:
                        # Atari wrapper can send a "done" signal when
                        # the agent loses a life, but it does not correspond
                        # to the true end of episode
                        if "episode" in info.keys():
                            # Do not trust "done" with episode endings.
                            # Monitor wrapper includes "episode" key in info if environment
                            # has been wrapped with it. Use those rewards instead.
                            episode_rewards.append(info["episode"]["r"])
                            episode_lengths.append(info["episode"]["l"])
                            # Only increment at the real end of an episode
                            episode_counts[i] += 1
                    else:
                        episode_rewards.append(current_rewards[i])
                        episode_lengths.append(current_lengths[i])
                        episode_counts[i] += 1
                    current_rewards[i] = 0
                    current_lengths[i] = 0

    print(f"Sum of rewards in trajectory: {np.sum(rewards_list)}")
    # print(f"Mean rewards in trajectory: {np.mean(episode_rewards)}")
    assert sum(episode_lengths) == len(rewards_list)
    print(f"Length: {len(rewards_list)}")

    trajectory_dict = {
        "obs": {"state": np.array(obs_list)},
        "action": {"action": np.array(actions_list)},
        "next_obs": {"state": np.array(next_obs_list)},
        "dones": np.array(dones_list),
        "terminals": np.array(dones_list),
        "rewards": np.array(rewards_list),
        "infos": np.array(infos_list),
    }
    return trajectory_dict


if __name__ == "__main__":
    N_ENVS = 1
    env = make_vec_env("Humanoid-v5", n_envs=N_ENVS)
    print(env.envs[0].spec)
    # Load a pretrained model from Hugging Face
    checkpoint = load_from_hub(
        repo_id="farama-minari/Humanoid-v5-SAC-expert",
        filename="humanoid-v5-sac-expert.zip",
    )
    model = SAC.load(checkpoint)
    trajectories: List = []
    avg_return = 0
    num_episodes = 100
    max_episode_steps = env.envs[0].spec.max_episode_steps
    print(f"Max Episode Steps: {max_episode_steps}")
    # Continue generating rollouts until we have enough valid demos
    while len(trajectories) < num_episodes:
        trajectory_dict = generate_offline_data(model, env, n_eval_episodes=1)
        ep_rewards = trajectory_dict["rewards"].sum()
        ep_lengths = len(trajectory_dict["rewards"])
        if ep_rewards > 8000 and ep_lengths == max_episode_steps:
            trajectories.append(trajectory_dict)
            avg_return += ep_rewards
            print(f"Accepted demo #{len(trajectories)}: Reward {ep_rewards}, Length {ep_lengths}")
        else:
            print(f"Rejected demo: Reward {ep_rewards}, Length {ep_lengths}")
    avg_return /= len(trajectories)
    print(f"Average Return: {avg_return}")

    import os
    fname = "./checkpoints/gym/humanoid-expert-v5.pkl"
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    fname = "./checkpoints/gym/humanoid-expert-v5.pkl"
    with open(fname, "wb") as fp:
        pickle.dump(trajectories, fp, protocol=pickle.HIGHEST_PROTOCOL)
