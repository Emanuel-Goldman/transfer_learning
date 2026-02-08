# Transfer Learning - Training Scripts

This repository contains training scripts for three reinforcement learning environments using Actor-Critic (A2C) algorithm.

## Environments

1. **Acrobot-v1** (discrete actions)
2. **CartPole-v1** (discrete actions)
3. **MountainCarContinuous-v0** (continuous actions)

## Installation

### 1. Create a Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

## Usage

Each environment has its own training script. You can run them directly or import and call the `train_actor_critic` function.

### Direct Execution

**CartPole-v1:**
```bash
python cartpole.py
```

**Acrobot-v1:**
```bash
python acrobot.py
```

**MountainCarContinuous-v0:**
```bash
python mountaincar.py
```

### Using the Function Directly

You can also import and call the training function with custom parameters:

**CartPole-v1:**
```python
from cartpole import train_actor_critic
train_actor_critic("CartPole-v1", episodes=1000, max_steps=500, seed=0)
```

**Acrobot-v1:**
```python
from acrobot import train_actor_critic
train_actor_critic("Acrobot-v1", episodes=1500, max_steps=1000, seed=1)
```

**MountainCarContinuous-v0:**
```python
from mountaincar import train_actor_critic
train_actor_critic("MountainCarContinuous-v0", episodes=1500, max_steps=500, seed=0)
```

### Command-Line Arguments

All scripts support command-line arguments:

**Normal training:**
```bash
python acrobot.py --episodes 1500 --max_steps 1000 --seed 1 --lr_p 5e-4 --lr_v 1e-3 --gamma 0.99
```

**Optuna hyperparameter search:**
```bash
python acrobot.py --optuna --n_trials 50 --episodes 1500 --max_steps 1000 --seed 1
```

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--optuna` | flag | `False` | Run Optuna hyperparameter search |
| `--n_trials` | int | `50` | Number of Optuna trials (only with `--optuna`) |
| `--episodes` | int | varies | Number of training episodes |
| `--max_steps` | int | varies | Maximum steps per episode |
| `--seed` | int | varies | Random seed |
| `--lr_p` | float | `5e-4` | Policy learning rate (only without `--optuna`) |
| `--lr_v` | float | `1e-3` | Value learning rate (only without `--optuna`) |
| `--gamma` | float | `0.99` | Discount factor (only without `--optuna`) |

## Optuna Hyperparameter Search

Optuna will search for optimal hyperparameters:
- **lr_p**: Policy learning rate (log scale: 1e-5 to 1e-2)
- **lr_v**: Value learning rate (log scale: 1e-4 to 1e-2)
- **gamma**: Discount factor (0.9 to 0.999)
- **hidden**: Hidden layer size (64, 128, or 256)

The search maximizes the average return over the last 100 episodes and uses pruning to stop unpromising trials early.

**Example:**
```bash
python acrobot.py --optuna --n_trials 50
```

## Function Parameters

The `train_actor_critic` function accepts the following parameters:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `env_name` | string | required | Environment name ("CartPole-v1", "Acrobot-v1", or "MountainCarContinuous-v0") |
| `episodes` | int | varies | Number of training episodes |
| `max_steps` | int | varies | Maximum steps per episode |
| `gamma` | float | `0.99` | Discount factor |
| `lr_p` | float | `5e-4` | Policy learning rate |
| `lr_v` | float | `1e-3` | Value learning rate |
| `seed` | int | varies | Random seed for reproducibility |

## Architecture

- **PADDED_STATE_DIM**: 6 (observations are padded to the maximum dimension across all tasks)
- **POLICY_OUT_DIM**: 5 
  - First 3 outputs: logits for discrete actions (CartPole, Acrobot)
  - Last 2 outputs: mean and log_std for continuous actions (MountainCar)
- **Network**: Single hidden layer with configurable size (64, 128, or 256) and ReLU activation
- **Device**: Automatically uses CUDA if available, otherwise CPU

## Output

The training function:
- Prints progress during training (episode reward and 100-episode average)
- Displays plots showing:
  - Reward per episode with 100-episode moving average
  - Policy and value losses per episode
- Returns a list of episode rewards

## Notes

- Training uses step-based updates (not episode-based)
- Episode endings (both `terminated` and `truncated`) are handled correctly - no bootstrapping when done
- The code automatically handles device selection (CPU/GPU)
