import os
from collections import defaultdict
import json
import torch
import numpy as np
import pickle
from common_utils import rela
from common_utils import ibrl_utils as utils


class Batch:
    def __init__(self, obs, next_obs, action, reward, bootstrap):
        self.obs = obs
        self.next_obs = next_obs
        self.action = action
        self.reward = reward
        self.bootstrap = bootstrap

    @classmethod
    def merge_batches(cls, batch0, batch1):
        obs = {k: torch.cat([v, batch0.obs[k]], dim=0) for k, v in batch1.obs.items()}
        next_obs = {
            k: torch.cat([v, batch0.next_obs[k]], dim=0) for k, v in batch1.next_obs.items()
        }
        action = {k: torch.cat([v, batch0.action[k]], dim=0) for k, v in batch1.action.items()}
        reward = torch.cat([batch1.reward, batch0.reward], dim=0)
        bootstrap = torch.cat([batch1.bootstrap, batch0.bootstrap], dim=0)
        return cls(obs, next_obs, action, reward, bootstrap)


class ReplayBuffer:
    def __init__(
        self,
        nstep,
        gamma,
        frame_stack,
        max_episode_length,
        replay_size,
        use_bc,
        bc_max_len=-1,
        save_per_success=-1,
        save_dir=None,
    ):
        self.replay_size = replay_size

        self.episode = rela.Episode(nstep, max_episode_length, gamma)
        self.replay = rela.SingleStepTransitionReplay(
            frame_stack=frame_stack,
            n_step=nstep,
            capacity=replay_size,
            seed=1,
            prefetch=3,
            extra=0.1,
        )

        self.bc_replay = None
        self.freeze_bc_replay = False
        if use_bc:
            self.bc_replay = rela.SingleStepTransitionReplay(
                frame_stack=frame_stack,
                n_step=nstep,
                capacity=replay_size,
                seed=1,
                prefetch=3,
                extra=0.1,
            )
        self.bc_max_len = bc_max_len
        self.save_per_success = save_per_success
        self.save_dir = save_dir

        self.episode_image_obs = defaultdict(list)
        self.num_success = 0
        self.num_episode = 0

    def new_episode(self, obs: dict[str, torch.Tensor]):
        self.episode_image_obs = defaultdict(list)
        self.episode.init({})
        self.episode.push_obs(obs)

    def append_obs(self, obs: dict[str, torch.Tensor]):
        self.episode.push_obs(obs)

    def append_reply(self, reply: dict[str, torch.Tensor]):
        self.episode.push_action(reply)

    def append_reward_terminal(self, reward: float, terminal: bool, success: bool):
        self.episode.push_reward(reward)
        self.episode.push_terminal(float(terminal))

        if terminal:
            self._push_episode(success)

    def add(
        self,
        obs: dict[str, torch.Tensor],
        reply: dict[str, torch.Tensor],
        reward: float,
        terminal: bool,
        success: bool,
    ):
        self.episode.push_action(reply)
        self.episode.push_reward(reward)
        self.episode.push_terminal(float(terminal))

        if not terminal:
            self.episode.push_obs(obs)
            return

        self._push_episode(success)

    def _push_episode(self, success):
        transition = self.episode.pop_transition()
        self.replay.add(transition)
        self.num_episode += 1

        if not success:
            return
        self.num_success += 1

        if self.bc_replay is None or self.freeze_bc_replay:
            return
        seq_len = int(transition.seq_len.item())

        if self.bc_max_len > 0 and seq_len > self.bc_max_len:
            print(f"episode too long {seq_len}, max={self.bc_max_len}, ignore")
            return
        self.bc_replay.add(transition)

        # dump the bc dataset for training
        if self.save_per_success <= 0 or self.num_success % self.save_per_success != 0:
            return
        # store the most recent n trajectories
        print(f"Saving bc replay; @{self.num_success} games.")
        size = self.bc_replay.size()
        episodes = self.bc_replay.get_range(size - self.save_per_success, size, "cpu")

        assert self.save_dir is not None
        save_id = self.num_success // self.save_per_success
        filename = os.path.join(self.save_dir, f"data{save_id}.h5")
        self._save_replay(filename, episodes)

    def save_replay(self, filename):
        size = self.replay.size()
        episodes = self.replay.get_range(0, size, "cpu")
        self._save_replay(filename, episodes)

    def _save_replay(self, filename, episodes):
        print(f"writing replay buffer to {filename}")
        size = episodes.seq_len.size(0)
        trajectories = []

        for i in range(size):
            ep_len = int(episodes.seq_len[i].item())

            traj = {
                "obs": {k: episodes.obs[k][:ep_len, i].numpy() for k in episodes.obs},
                "action": {k: episodes.action[k][:ep_len, i].numpy() for k in episodes.action},
                "rewards": episodes.reward[:ep_len, i].numpy(),
                "dones": np.concatenate([np.zeros(ep_len - 1, dtype=bool), np.ones(1, dtype=bool)]),
            }
            trajectories.append(traj)

        # Ensure .pkl extension
        if not filename.endswith(".pkl"):
            filename = filename.rstrip(".h5") + ".pkl"

        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, "wb") as fp:
            pickle.dump(trajectories, fp, protocol=pickle.HIGHEST_PROTOCOL)

        print(f"Saved {size} trajectories to {filename}")


    def sample(self, batchsize, device):
        return self.replay.sample(batchsize, device)

    def sample_bc(self, batchsize, device):
        assert self.bc_replay is not None
        assert self.num_success > 0
        return self.bc_replay.sample(batchsize, device)

    def sample_rl_bc(self, rl_bsize, bc_bsize, device):
        rl_batch = self.sample(rl_bsize, device)
        bc_batch = self.sample_bc(bc_bsize, device)

        batch = Batch.merge_batches(rl_batch, bc_batch)
        return batch

    def size(self, bc=False):
        if bc:
            assert self.bc_replay is not None
            return self.bc_replay.size()
        else:
            return self.replay.size()


def add_demos_to_replay(
    replay: ReplayBuffer,
    data_path: str,
    num_data: int,
    reward_scale: float = 1.0,
    is_demo: bool = True,
):
    """
    Loads offline demonstration data from a pickle file (list of trajectories),
    and pushes it into the ReplayBuffer. Each trajectory in the pickle file
    must be a dict with keys at least:
        "obs":    {"state": np.ndarray of shape [T, obs_dim]}
        "action": {"action": np.ndarray of shape [T, act_dim]}
        "rewards": np.ndarray of shape [T,]
        "dones":   np.ndarray of shape [T,] (bools)
        ... possibly "terminals", "infos", etc. ...
    """

    with open(data_path, "rb") as f:
        trajectories = pickle.load(f)

    # If num_data <= 0, we load all of them
    if num_data <= 0 or num_data > len(trajectories):
        num_data = len(trajectories)

    print(f"Loading first {num_data} episodes from {data_path}. Total stored: {len(trajectories)}")

    # Go through each trajectory and add it to the replay buffer
    for i, traj in enumerate(trajectories[:num_data]):
        obs_array = traj["obs"]["state"]        # shape (T, obs_dim)
        actions_array = traj["action"]["action"] # shape (T, act_dim)
        rewards_array = traj["rewards"]         # shape (T,)
        dones_array = traj["dones"]             # shape (T,) (bool)


        # Typically, T = number of steps in the trajectory
        T = len(rewards_array)

        if is_demo:
            assert rewards_array[-1] == 1
            terminals = rewards_array
        else:
            terminals = rewards_array[:]
            terminals[-1] = 1
        # We'll do 1 extra iteration so that at t=0 we can call new_episode(...)
        # and at t>0 we process the t-1 action and reward.
        for t in range(T + 1):
            if t < T:
                # Current observation
                current_obs = {
                    "state": torch.from_numpy(obs_array[t]).float()
                }

            if t == 0:
                # Start a new episode with the first observation
                replay.new_episode(current_obs)
                continue

            # Action at step (t-1)
            action_dict = {
                "action": torch.from_numpy(actions_array[t - 1]).float()
            }
            r = float(rewards_array[t - 1]) * reward_scale
            done = bool(dones_array[t - 1])

            # If these are demonstration episodes that only end when
            # the task is successful, you might define success as:
            success = bool(rewards_array[t - 1] == 1)
            terminal = bool(terminals[t - 1])

            # Add transition
            replay.add(
                obs=current_obs,
                reply=action_dict,
                reward=r,
                terminal=terminal,
                success=success,
            )

            # If environment says done, break out of this episode
            if done:
                break

    print(f"Size of main replay buffer: {replay.size()}")
    print(f"Number of success episodes so far: {replay.num_success}")
    if replay.bc_replay is not None:
        print(f"Size of BC replay buffer: {replay.bc_replay.size()}")



if __name__ == "__main__":
    replay = ReplayBuffer(
        nstep=3,
        gamma=0.99,
        frame_stack=1,
        max_episode_length=1000,
        replay_size=100,
        use_bc=True,
        save_per_success=-1,
        save_dir="exps/rl/run1",
    )
    add_demos_to_replay(
        replay,
        "../release/data/mujoco/humanoid-expert-v5.pkl",
        num_data=2,
        reward_scale=1,
        is_demo=True,
    )
