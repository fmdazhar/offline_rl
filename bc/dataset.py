import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from bc_policy import StateBcPolicy, StateBcPolicyConfig
from collections import namedtuple, defaultdict
import numpy as np
from dataclasses import dataclass


Batch = namedtuple("Batch", ["obs", "action"])

@dataclass
class DatasetConfig:
    path: str = ""
    num_data: int = -1
    max_len: int = -1
    eval_episode_len: int = 300
    use_state: int = 0
    prop_stack: int = 1
    norm_action: int = 0
    obs_stack: int = 1
    state_stack: int = 1

class BcDataset:
    def __init__(self, raw_data: list):
        self.data = []
        self.idx2entry = []
        self.prop_stack = 1

        # raw_data is a list of episodes; each episode is a dict of arrays
        for ep_idx, episode in enumerate(raw_data):
            states = episode["obs"]["state"]        # shape (T, state_dim)
            actions = episode["action"]["action"]   # shape (T, action_dim)
            T = states.shape[0]

            ep_entries = []
            for t in range(T):
                entry = {
                    "state": torch.tensor(states[t], dtype=torch.float32),
                    "action": torch.tensor(actions[t], dtype=torch.float32),
                }
                ep_entries.append(entry)
                self.idx2entry.append((ep_idx, t))

            self.data.append(ep_entries)

        # meta‑info (mimics RobomimicDataset)
        last = self.data[-1][-1]
        self.obs_shape = last["state"].shape
        self.action_dim = last["action"].shape[0]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def _convert_to_batch(self, samples, device):
        batch = {k: torch.stack(v).to(device) for k, v in samples.items() if k != "action"}
        action = {"action": torch.stack(samples["action"]).to(device)}
        return Batch(obs=batch, action=action)

    def sample_bc(self, batchsize, device):
        indices = np.random.choice(len(self.idx2entry), batchsize)
        samples = defaultdict(list)
        for idx in indices:
            ep_idx, t = self.idx2entry[idx]
            entry = self.data[ep_idx][t]
            for k, v in entry.items():
                samples[k].append(v)
        return self._convert_to_batch(samples, device)
    
