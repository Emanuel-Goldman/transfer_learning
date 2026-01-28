# Transfer Learning - HW3

This repository contains implementations for:
- **Section 1**: Training individual networks using Advantage Actor-Critic (A2C)
- **Section 2**: Fine-tuning / Transfer learning between environments

## Overview

The implementation trains a unified Actor-Critic network on three environments:
1. **CartPole-v1** (discrete actions)
2. **Acrobot-v1** (discrete actions)
3. **MountainCarContinuous-v0** (continuous actions)

All networks share **identical input and output dimensions** to support later transfer learning:
- **Input dimension**: 6 (observations are padded to the maximum dimension across all tasks)
- **Output dimension**: 3 (for discrete actions, dummy actions are masked; for continuous actions, only the first dimension is used)

## Features

- ✅ Unified architecture supporting both discrete and continuous action spaces
- ✅ Action masking for dummy actions (prevents invalid action selection)
- ✅ Observation padding to fixed dimensions
- ✅ Deterministic seeding for reproducibility
- ✅ Automatic convergence detection with environment-specific stopping criteria
- ✅ Comprehensive logging and visualization

## Requirements

- Python 3.7+
- See `requirements.txt` for package dependencies

## Installation

### 1. Create a Virtual Environment

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# On Linux/Mac:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate
```

### 2. Install Dependencies

```bash
# Install all required packages
pip install -r requirements.txt
```

### 3. Verify Installation

```bash
# Check that packages are installed
python -c "import gymnasium, numpy, torch, matplotlib; print('All packages installed successfully!')"
```

**Note**: Remember to activate the virtual environment (`source venv/bin/activate`) each time you work on this project. To deactivate, simply run `deactivate`.

## Usage

### Section 1: Training Individual Networks

Each environment has its own dedicated training script with Optuna hyperparameter optimization support.

#### CartPole-v1

**Run Optuna hyperparameter search:**
```bash
python train_cartpole.py --optuna --n_trials 30 --seed 0 --max_episodes 500
```

**Train with specific hyperparameters:**
```bash
python train_cartpole.py --seed 0 \
  --lr_policy 0.001 --lr_value 0.003 \
  --gamma 0.99 --entropy_coef 0.02 \
  --normalize_advantages --update_mode episode \
  --hidden_sizes 128,128 --max_episodes 1000
```

#### Acrobot-v1

**Run Optuna hyperparameter search:**
```bash
python train_acrobot.py --optuna --n_trials 30 --seed 0 --max_episodes 500
```

**Train with specific hyperparameters:**
```bash
python train_acrobot.py --seed 0 \
  --lr_policy 0.001 --lr_value 0.003 \
  --gamma 0.99 --entropy_coef 0.02 \
  --normalize_advantages --update_mode episode \
  --hidden_sizes 128,128 --max_episodes 1000
```

#### MountainCarContinuous-v0

**Run Optuna hyperparameter search:**
```bash
python train_mountaincar.py --optuna --n_trials 30 --seed 0 --max_episodes 500
```

**Train with specific hyperparameters:**
```bash
python train_mountaincar.py --seed 0 \
  --lr_policy 0.001 --lr_value 0.003 \
  --gamma 0.99 --entropy_coef 0.02 \
  --normalize_advantages --update_mode step \
  --hidden_sizes 256,256 --max_episodes 5000
```

### Command-Line Arguments

All training scripts support the following arguments:

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--seed` | int | `0` | Random seed for reproducibility |
| `--lr_policy` | float | `3e-4` | Policy learning rate |
| `--lr_value` | float | `1e-3` | Value learning rate |
| `--gamma` | float | `0.99` | Discount factor |
| `--entropy_coef` | float | `0.01` | Entropy regularization coefficient |
| `--normalize_advantages` | flag | `False` | Normalize advantages |
| `--update_mode` | string | `episode` | Update mode: `episode` or `step` |
| `--hidden_sizes` | string | `128,128` | Hidden layer sizes (comma-separated) |
| `--max_episodes` | int | `None` | Maximum training episodes |
| `--eval_interval` | int | `100` | Evaluation interval (episodes) |
| `--eval_episodes` | int | `10` | Number of episodes for evaluation |
| `--artifact_dir` | string | `results` | Output directory for results |
| `--optuna` | flag | `False` | Run Optuna hyperparameter search |
| `--n_trials` | int | `30` | Number of Optuna trials (only with `--optuna`) |

### Optuna Hyperparameter Search

Each training script includes Optuna optimization that searches for the best hyperparameters:

- **Search space**: Learning rates (policy/value), gamma, entropy coefficient, normalize advantages, update mode, hidden sizes
- **Output**: Prints each trial's hyperparameters and results to terminal, then prints best hyperparameters at the end
- **Objective**: Maximizes final average return (last 100 episodes)

Example Optuna output:
```
Trial 0:
  lr_policy=0.000234, lr_value=0.001456, gamma=0.95, ...
  Final avg return (last 100): 475.23
  Converged: True
  Episodes to converge: 245
  ...

Best hyperparameters:
  lr_policy: 0.001234
  lr_value: 0.003456
  ...
```

### Help

View all available options for each script:

```bash
python train_cartpole.py --help
python train_acrobot.py --help
python train_mountaincar.py --help
```

## Convergence Criteria

Training stops when the average return over the last 100 episodes meets the target:

- **CartPole-v1**: ≥ 475
- **Acrobot-v1**: ≥ -100
- **MountainCarContinuous-v0**: ≥ 90

A maximum episode limit of 10,000 is enforced to prevent infinite training.

## Output

All results are saved in the `results/` directory:

### Summary JSON

For each run, a summary file is created: `{env_name}_seed{seed}_summary.json`

Contains:
- Environment name and seed
- Hyperparameters used
- Runtime in seconds
- Number of episodes to convergence
- Final average return
- Convergence status

### Training Curves

Training curve plots are saved as: `{env_name}_seed{seed}_training_curve.png`

Each plot shows:
- Episode returns (blue, semi-transparent)
- 100-episode moving average (red line)
- Target return threshold (green dashed line)

## Architecture Details

### Unified Network Design

- **Shared feature extractor**: 2-layer MLP with ReLU activations (128 hidden units)
- **Actor head**: Outputs fixed `ACTION_DIM = 3` logits/values
  - Discrete: Full logits with masking for dummy actions
  - Continuous: Mean values (first `action_dim` outputs) with learnable log_std
- **Critic head**: Single value estimate

### Observation Padding

Observations are automatically padded to `OBS_DIM = 6`:
- CartPole-v1: 4 → 6 (padded with zeros)
- Acrobot-v1: 6 → 6 (no padding needed)
- MountainCarContinuous-v0: 2 → 6 (padded with zeros)

### Action Masking

For discrete action spaces:
- Valid actions have logits as normal
- Dummy actions (beyond actual action space) have logits set to `-inf`
- This ensures dummy actions are never sampled

## Implementation Notes

- All random seeds are set deterministically (numpy, torch, Python random, environment)
- The implementation uses on-policy A2C with episode-based updates
- Advantages are computed using TD(0) returns
- Gradient clipping is applied (max norm: 0.5)

---

## Section 2 – Fine-tuning / Transfer Learning

Section 2 implements transfer learning by fine-tuning a pre-trained model from one environment (source) to another (target). Only the output layers (policy head and value head) are reinitialized; the feature extractor weights are transferred from the source model.

### Overview

The fine-tuning process:
1. **Load** a trained source model checkpoint
2. **Reinitialize** only the output layers (policy and value heads) while keeping feature extractor weights
3. **Train** on the target environment using the same training procedure as Section 1
4. **Compare** results with training from scratch (Section 1)

### Transfer Pairs

Two transfer pairs are supported:
1. **Acrobot-v1 → CartPole-v1** (discrete → discrete)
2. **CartPole-v1 → MountainCarContinuous-v0** (discrete → continuous)

### Step-by-Step Instructions

#### Step 1: Train Source Environment (Section 1)

First, train a model on the source environment and save the checkpoint:

```bash
# Train Acrobot-v1 (source for first transfer pair)
python section1_actor_critic.py --env Acrobot-v1 --seed 0 \
  --lr_policy 0.001957699823689008 \
  --lr_value 0.005729747681869129 \
  --gamma 0.9168770457543605 \
  --entropy_coef 0.04991493043969712 \
  --normalize_advantages \
  --update_mode episode \
  --hidden_sizes 64,128 \
  --artifact_dir results/section1/acrobot_seed0

# Train CartPole-v1 (source for second transfer pair)
python section1_actor_critic.py --env CartPole-v1 --seed 0 \
  --lr_policy 0.001957699823689008 \
  --lr_value 0.005729747681869129 \
  --gamma 0.9168770457543605 \
  --entropy_coef 0.04991493043969712 \
  --normalize_advantages \
  --update_mode episode \
  --hidden_sizes 64,128 \
  --artifact_dir results/section1/cartpole_seed0
```

Checkpoints will be saved to:
- `results/section1/acrobot_seed0/checkpoints/agent_final_policy.pt`
- `results/section1/acrobot_seed0/checkpoints/agent_final_value.pt`

#### Step 2: Fine-tune on Target Environment (Section 2)

**Transfer Pair 1: Acrobot → CartPole**

```bash
python section2_finetune.py \
  --source_env Acrobot-v1 \
  --target_env CartPole-v1 \
  --source_checkpoint results/section1/acrobot_seed0/checkpoints/agent_final \
  --seed 0 \
  --lr_policy 0.001957699823689008 \
  --lr_value 0.005729747681869129 \
  --gamma 0.9168770457543605 \
  --entropy_coef 0.04991493043969712 \
  --normalize_advantages \
  --update_mode episode \
  --hidden_sizes 64,128 \
  --max_episodes 1000
```

**Transfer Pair 2: CartPole → MountainCarContinuous**

```bash
python section2_finetune.py \
  --source_env CartPole-v1 \
  --target_env MountainCarContinuous-v0 \
  --source_checkpoint results/section1/cartpole_seed0/checkpoints/agent_final \
  --seed 0 \
  --lr_policy 0.001957699823689008 \
  --lr_value 0.005729747681869129 \
  --gamma 0.9168770457543605 \
  --entropy_coef 0.04991493043969712 \
  --normalize_advantages \
  --update_mode episode \
  --hidden_sizes 64,128 \
  --max_episodes 1000
```

### Section 2 Command-Line Arguments

| Argument | Type | Description |
|----------|------|-------------|
| `--source_env` | string | **required** | Source environment name |
| `--target_env` | string | **required** | Target environment name |
| `--source_checkpoint` | string | **required** | Path prefix to source checkpoint (without `_policy.pt`/`_value.pt` suffix) |
| `--seed` | int | `0` | Random seed |
| `--lr_policy` | float | `3e-4` | Policy learning rate |
| `--lr_value` | float | `1e-3` | Value learning rate |
| `--gamma` | float | `0.99` | Discount factor |
| `--entropy_coef` | float | `0.01` | Entropy coefficient |
| `--normalize_advantages` | flag | Normalize advantages |
| `--update_mode` | string | `episode` | Update mode: `episode` or `step` |
| `--hidden_sizes` | string | `128,128` | Hidden layer sizes (comma-separated) |
| `--max_episodes` | int | `None` | Maximum training episodes |
| `--output_dir` | string | `None` | Output directory (default: `results/section2/<source>_to_<target>/seed_<seed>/`) |
| `--compare_with_section1` | string | `None` | Path to Section 1 stats.json for comparison |

### Output Structure

Section 2 results are saved to: `results/section2/<source_env>_to_<target_env>/seed_<seed>/`

```
results/section2/
├── Acrobot-v1_to_CartPole-v1/
│   └── seed_0/
│       ├── stats.json              # Convergence statistics
│       ├── metrics.json             # Training metrics
│       ├── checkpoints/
│       │   ├── agent_best_policy.pt
│       │   ├── agent_best_value.pt
│       │   ├── agent_final_policy.pt
│       │   └── agent_final_value.pt
│       └── plots/
│           └── CartPole-v1_seed0_training_curve.png
└── CartPole-v1_to_MountainCarContinuous-v0/
    └── seed_0/
        └── ...
```

### Comparing Results

Section 2 automatically compares fine-tuning results with Section 1 training from scratch if Section 1 stats are found at `results/<target_env>/stats.json`. The comparison shows:

- Episodes to converge (from scratch vs fine-tuning)
- Time to converge (from scratch vs fine-tuning)
- Speedup factor (if both converged)

Example output:
```
============================================================
COMPARISON: From Scratch vs Fine-tuning
============================================================
Environment: CartPole-v1

From Scratch (Section 1):
  Episodes to converge: 250
  Time to converge: 45.32 seconds
  Converged: True

Fine-tuning (Section 2):
  Episodes to converge: 180
  Time to converge: 32.15 seconds
  Converged: True

Speedup: 1.41x
============================================================
```

### Key Implementation Details

- **Output Layer Reinitialization**: Only the final Linear layers (policy head and value head) are reinitialized using Xavier uniform initialization for weights and zeros for bias
- **Feature Extractor Transfer**: All hidden layers (feature extractor) are kept from the source model
- **Same Training Procedure**: Uses identical training loop, convergence criteria, and logging as Section 1
- **Deterministic Seeding**: Ensures reproducibility with proper seed handling

---

## File Structure

```
transfer_learning/
├── common.py                      # Shared code (models, utilities, training function)
├── train_cartpole.py              # CartPole-v1 training script with Optuna
├── train_acrobot.py               # Acrobot-v1 training script with Optuna
├── train_mountaincar.py           # MountainCarContinuous-v0 training script with Optuna
├── section1_actor_critic.py      # Original unified training script (deprecated)
├── section2_finetune.py          # Section 2: Fine-tuning / Transfer learning
├── requirements.txt               # Python dependencies
├── README.md                      # This file
├── venv/                          # Virtual environment (created during setup)
└── results/                       # Output directory (created automatically)
    ├── CartPole-v1/               # CartPole results
    ├── Acrobot-v1/                # Acrobot results
    ├── MountainCarContinuous-v0/  # MountainCar results
    └── section2/                  # Section 2 results
        ├── Acrobot-v1_to_CartPole-v1/
        └── CartPole-v1_to_MountainCarContinuous-v0/
```

**Note**: The `venv/` directory should be added to `.gitignore` if using version control.

## License

This code is part of a university assignment.
