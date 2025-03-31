# Offline RL for Humanoid with Behavioral Cloning & Finetuning

This repository demonstrates how to:

1. **Generate Offline Data** using a pretrained policy (e.g., from Hugging Face Hub).
2. **Train a Reinforcement Learning (RL) agent** from offline data, with optional pretraining or fine-tuning using Behavioral Cloning (BC).

---

## 📜 Key Scripts

- **`generate_offline_data.py`**: Generates offline trajectories by rolling out a pretrained Stable-Baselines policy (e.g. from the Hugging Face Hub) and saves them as `.pkl`.
- **`train_bc.py`**: Trains a behavioral cloning policy using the generated offline dataset.
- **`train_rl.py`**: Trains an RL agent (e.g. SAC, or a custom QAgent) using either the replay buffer or a combination of replay buffer + BC.

---

## 1. 🛠️ Setup Environment

### 1.1 Install MuJoCo (if not already installed)

Download the [MuJoCo 2.1 binaries](https://mujoco.org/download) and extract them to:

```bash
~/.mujoco/mujoco210
```

### 1.2 Create Conda Environment (Recommended)

```bash
conda create --name offline_rl python=3.9
conda activate offline_rl
```

### 1.3 Install Dependencies

Adjust the CUDA version if needed.

```bash
# Example: PyTorch 2.1 with CUDA 12.1
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121

# Then install the rest
pip install -r requirements.txt
```

(Optional) Build the C++ extension in `common_utils`:

```bash
cd common_utils
make
cd ..
```

### 1.4 Set Up Environment Variables (Optional)

Update and source `set_env.sh` if it contains environment variables:

```bash
source set_env.sh
```

---

## 2. 📦 Generate Offline Data

Generate offline data from a pretrained policy (e.g., from Hugging Face Hub):

```bash
python generate_offline_data.py
```

By default:
- Uses a Farama-Minari SAC expert policy on Humanoid-v5.
- Filters out poor trajectories (e.g., rewards < 8000).
- Saves data to `release/data/humanoid-expert-v5.pkl`.

You can modify:
- Environment name.
- Save location.
- Number of episodes and reward threshold.

---

## 3. 🧠 Train a Behavioral Cloning (BC) Policy

Run:

```bash
python train_bc.py --config_path path/to/config.yaml
```

Example minimal config (`humanoid_bc.yaml`):

```yaml
seed: 1
device: "cuda"
preload_datapath: "release/data/humanoid-expert-v5.pkl"
batch_size: 256
...
```

Checkpoints and logs will be saved to `save_dir`.

---

## 4. 🏋️ Train RL with Offline Data (+ Optional BC Policy)

Train SAC or custom Q-learning RL agent from offline data:

```bash
python train_rl.py --config_path path/to/config.yaml
```

Example config options:

```yaml
preload_datapath: "release/data/humanoid-expert-v5.pkl"
preload_num_data: 100

bc_policy: "path/to/bc_policy_checkpoint.pt"
mix_rl_rate: 0.5

pretrain_num_epoch: 10
pretrain_epoch_len: 10000
pretrain_only: 0
```

---


## ✅ Example Commands

### Generate Data

```bash
python generate_offline_data.py
```

### Train BC Policy

```bash
python train_bc.py --config_path release/cfgs/mujoco_bc/humanoid_bc.yaml
```

### Train RL (SAC) from Offline Data

```bash
python train_rl.py --config_path release/cfgs/mujoco_rl/humanoid_rl.yaml \
                   --preload_datapath release/data/humanoid-expert-v5.pkl \
                   --preload_num_data 50 \
                   --num_train_step 200000 \
                   --use_wb 1
```

### Train RL with BC Initialization

```bash
python train_rl.py --config_path release/cfgs/mujoco_rl/humanoid_rl.yaml \
                   --bc_policy path/to/bc_policy.pt \
                   --pretrain_num_epoch 5 \
                   --pretrain_epoch_len 10000
```

---
