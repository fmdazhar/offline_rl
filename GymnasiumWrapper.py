import gymnasium as gym
from gymnasium.wrappers import FrameStackObservation, FlattenObservation
import numpy as np
import torch

class StateDictWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Dict({
            "state": env.observation_space
        })

    def observation(self, observation):
        return {"state": observation}

class SuccessWrapper(gym.Wrapper):
    def __init__(self, env, success_threshold=8000):
        super().__init__(env)
        self.success_threshold = success_threshold
        self.episode_reward = 0
        self.time_step = 0
    
    def reset(self, **kwargs):
        self.episode_reward = 0
        self.time_step = 0
        obs, info = self.env.reset(**kwargs)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.time_step += 1
        self.episode_reward += reward
        success = self.episode_reward >= self.success_threshold
        # Add success flag to the returned tuple
        return obs, reward, terminated, success, info

class ToTensorWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
    def observation(self, obs):
        # assumes obs is a dict {"state": np.ndarray}
        return {k: torch.from_numpy(v).float() for k, v in obs.items()}

def create_env(render_mode="rgb_array", env_name="Humanoid-v5", max_episode_steps=1000):
    env = gym.make(env_name, render_mode=render_mode, max_episode_steps=max_episode_steps)
    env = FrameStackObservation(env, stack_size=1)
    env = FlattenObservation(env)
    env = StateDictWrapper(env)
    env = ToTensorWrapper(env)
    env = SuccessWrapper(env, success_threshold=5000)
    env.observation_shape = env.observation_space["state"].shape

    return env


if __name__ == "__main__":
    env = create_env()
    print(env.observation_space)
    print(env.action_space.shape[0])
    # Test the wrapper chain
    obs, info = env.reset(seed=123)
    
    for _ in range(1000):
        obs, reward, terminated, success, info = env.step(env.action_space.sample())
        if terminated:
            obs, info = env.reset(seed=123)
        if success:
            print("Success!")
            break
    print("Done.")
    env.close()
