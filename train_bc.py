import os
import sys
from dataclasses import dataclass, field
from typing import Optional, List

import torch
import numpy as np
import pyrallis
import yaml
from  GymnasiumWrapper import create_env
import common_utils
import wandb
from evaluate import run_eval_gym


@dataclass
class DatasetConfig:
    path: str = "./checkpoints/gym/humanoid-expert-v5.pkl"
    num_data: int = -1
    max_len: int = -1
    eval_episode_len: int = 300
    use_state: int = 1
    prop_stack: int = 1
    norm_action: int = 0
    obs_stack: int = 1
    state_stack: int = 1



from collections import defaultdict, namedtuple

Batch = namedtuple("Batch", ["obs", "action"])

class GymStateDataset:
    """
    A simple dataset for offline BC from a Gym environment with state-only observations.
    This matches the interface of 'RobomimicDataset' from your original code.
    """
    def __init__(self, cfg: DatasetConfig):
        self.cfg = cfg
        # Load the raw data from the pickle (or any other) file
        import pickle
        with open(cfg.path, "rb") as f:
            raw_data = pickle.load(f)  # expecting a list of episodes

        self.data = []
        self.idx2entry = []

        for ep_id, episode in enumerate(raw_data):
            states = episode["obs"]["state"]    # shape (T, state_dim)
            actions = episode["action"]["action"]  # shape (T, action_dim)
            T = states.shape[0]

            ep_entries = []
            for t in range(T):
                entry = {
                    "state": torch.tensor(states[t], dtype=torch.float32),
                    "action": torch.tensor(actions[t], dtype=torch.float32),
                }
                ep_entries.append(entry)
                self.idx2entry.append((ep_id, t))
            self.data.append(ep_entries)

        # For convenience, pick the last item to determine shapes
        last = self.data[-1][-1]
        self.obs_shape = last["state"].shape
        self.action_dim = last["action"].shape[0]

        print(f"Dataset loaded: {len(self.data)} episodes, {len(self.idx2entry)} total steps.")
        print(f"Obs shape: {self.obs_shape}, Action dim: {self.action_dim}")

    def __len__(self):
        return len(self.data)  # number of episodes (not total steps!)

    def __getitem__(self, idx):
        return self.data[idx]

    def sample_bc(self, batch_size, device):
        """
        Randomly sample transitions for supervised BC.
        """
        indices = np.random.choice(len(self.idx2entry), batch_size)
        collected = defaultdict(list)
        for idx in indices:
            ep_id, t = self.idx2entry[idx]
            entry = self.data[ep_id][t]
            for k, v in entry.items():
                collected[k].append(v)
        # Convert to batch
        obs_dict = {}
        for k, v in collected.items():
            if k == "action":
                continue
            obs_dict[k] = torch.stack(v).to(device)
        actions = {"action": torch.stack(collected["action"]).to(device)}
        return Batch(obs_dict, actions)


from bc.bc_policy import StateBcPolicy, StateBcPolicyConfig

@dataclass
class MainConfig(common_utils.RunConfig):
    dataset: DatasetConfig = field(default_factory=lambda: DatasetConfig())
    state_policy: StateBcPolicyConfig = field(default_factory=lambda: StateBcPolicyConfig())
    # training
    seed: int = 1
    device: str = "cpu"
    load_model: str = "none"
    num_epoch: int = 20
    epoch_len: int = 10000
    batch_size: int = 256
    lr: float = 1e-4
    grad_clip: float = 5
    weight_decay: float = 0
    # logging
    save_dir: str = "exps/bc/gym_run1"
    use_wb: int = 0
    save_per: int = -1

    # Here just for compatibility with prior usage
    task_name: str = ""
    robots: List[str] = field(default_factory=lambda: [])
    image_size: int = -1
    rl_image_size: int = -1

def run(cfg: MainConfig, policy=None):
    # Build dataset
    dataset = GymStateDataset(cfg.dataset)
    print(f"Obs shape: {dataset.obs_shape}, Action dim: {dataset.action_dim}")

    # Dump the config for record-keeping
    pyrallis.dump(cfg, open(cfg.cfg_path, "w"))
    print(common_utils.wrap_ruler("config"))
    with open(cfg.cfg_path, "r") as f:
        print(f.read())
    cfg_dict = yaml.safe_load(open(cfg.cfg_path, "r"))

    # Create policy if none was loaded
    if policy is None:
        # We'll assume it's always a state-based policy for this environment
        policy = StateBcPolicy(dataset.obs_shape, dataset.action_dim, cfg.state_policy)
    policy = policy.to(cfg.device)
    print(common_utils.wrap_ruler("policy"))
    print(policy)
    common_utils.count_parameters(policy)

    # Create optimizer
    if cfg.weight_decay == 0:
        print("Using Adam optimizer")
        optim = torch.optim.Adam(policy.parameters(), lr=cfg.lr)
    else:
        print("Using AdamW optimizer")
        optim = torch.optim.AdamW(policy.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    # Setup logging
    stat = common_utils.MultiCounter(
        cfg.save_dir,
        bool(cfg.use_wb),
        wb_exp_name=cfg.wb_exp,
        wb_run_name=cfg.wb_run,
        wb_group_name=cfg.wb_group,
        config=cfg_dict,
    )
    saver = common_utils.TopkSaver(cfg.save_dir, 2)
    stopwatch = common_utils.Stopwatch()

    best_loss = 1e9
    saved = False

    # Training
    for epoch in range(cfg.num_epoch):
        print(f"Epoch {epoch+1}/{cfg.num_epoch}")
        stopwatch.reset()

        # Run epoch
        for i in range(cfg.epoch_len):
            with stopwatch.time("sample"):
                batch = dataset.sample_bc(cfg.batch_size, cfg.device)

            with stopwatch.time("train"):
                loss = policy.loss(batch)
                optim.zero_grad()
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg.grad_clip)
                optim.step()

            stat["train/loss"].append(loss.item())
            stat["train/grad_norm"].append(grad_norm.item())

            # Just a quick progress update
            if (i + 1) % (cfg.epoch_len // 5) == 0:
                print(f"  step {i+1}/{cfg.epoch_len}, loss = {loss.item():.4f}")

        epoch_time = stopwatch.elapsed_time_since_reset
        stat["other/speed"].append(cfg.epoch_len / epoch_time)
        print(f"Finished epoch {epoch+1} in {epoch_time:.2f}s, speed {cfg.epoch_len / epoch_time:.2f} it/s")

        # Optionally save if this is the best so far
        # (or implement your own metric as needed)
        if loss.item() < best_loss:
            best_loss = loss.item()
            saved = saver.save(policy.state_dict(), best_loss, save_latest=True)
            if saved:
                print("New best model saved!")
        else:
            # always keep 'latest' around
            saver.save(policy.state_dict(), loss.item(), save_latest=True)

        # summary to logs
        stat.summary(epoch)
        stopwatch.summary()

    # If desired, load best model and do final checks
    # but we skip evaluation if not needed
    if saved:
        best_model = saver.get_best_model()
        policy.load_state_dict(torch.load(best_model))
        print("Loaded best model weights for final usage.")

    # done
    print("Done with training, exiting.")
    # force exit like your code
    assert False


def evaluate(policy, seed, num_game):
    return run_eval_gym(
        policy, num_game=num_game, seed=seed, verbose=False
    )


def _load_model(weight_file, env, device, cfg: Optional[MainConfig] = None):
    if cfg is None:
        cfg_path = os.path.join(os.path.dirname(weight_file), f"cfg.yaml")
        cfg = pyrallis.load(MainConfig, open(cfg_path, "r"))  # type: ignore

    print("observation shape: ", env.observation_shape)
    policy = StateBcPolicy(env.observation_shape, env.action_space.shape[0], cfg.state_policy)
    policy.load_state_dict(torch.load(weight_file, map_location=device))
    return policy.to(device)


# function to load bc models
def load_model(weight_file, device, *, verbose=True):
    run_folder = os.path.dirname(weight_file)
    cfg_path = os.path.join(run_folder, f"cfg.yaml")
    if verbose:
        print(common_utils.wrap_ruler("config of loaded agent"))
        with open(cfg_path, "r") as f:
            print(f.read(), end="")
        print(common_utils.wrap_ruler(""))

    cfg = pyrallis.load(MainConfig, open(cfg_path, "r"))  # type: ignore

    env = create_env()
    print(f"state_stack: {cfg.dataset.state_stack}, observation shape: {env.observation_shape}")
    policy = _load_model(weight_file, env, device, cfg)
    return policy, env


if __name__ == "__main__":
    import rich.traceback
    rich.traceback.install()
    torch.set_printoptions(linewidth=100)

    # parse
    cfg = pyrallis.parse(config_class=MainConfig)
    common_utils.set_all_seeds(cfg.seed)

    # optional: redirect stdout to a file
    log_path = os.path.join(cfg.save_dir, "train.log")
    sys.stdout = common_utils.Logger(log_path, print_to_stdout=True)

    # (Optional) Load model if wanted
    policy = None
    if cfg.load_model and cfg.load_model != "none":
        # You can implement or reuse a load function if your policy structure is the same
        state_dict = torch.load(cfg.load_model, map_location=cfg.device)
        policy = StateBcPolicy((1,), 1, cfg.state_policy)  # dummy shape, real shape is set once loaded
        policy.load_state_dict(state_dict)
        policy = policy.to(cfg.device)

    # run
    run(cfg, policy)

    # finish wandb if used
    if cfg.use_wb:
        wandb.finish()
