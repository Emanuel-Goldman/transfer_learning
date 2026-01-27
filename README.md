# Transfer Learning - HW3 Section 1

This repository contains the implementation for **Section 1: Training individual networks** using Advantage Actor-Critic (A2C) on three different environments.

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

### Basic Commands

Train on each environment:

```bash
# CartPole-v1
python section1_actor_critic.py --env CartPole-v1 --seed 0

# Acrobot-v1
python section1_actor_critic.py --env Acrobot-v1 --seed 0

# MountainCarContinuous-v0
python section1_actor_critic.py --env MountainCarContinuous-v0 --seed 0
```

### Command-Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--env` | string | **required** | Environment name: `CartPole-v1`, `Acrobot-v1`, or `MountainCarContinuous-v0` |
| `--seed` | int | `0` | Random seed for reproducibility |
| `--lr` | float | `3e-4` | Learning rate |
| `--gamma` | float | `0.99` | Discount factor |
| `--value_coef` | float | `0.5` | Value loss coefficient |
| `--entropy_coef` | float | `0.01` | Entropy regularization coefficient |
| `--device` | string | `cpu` | Device: `cpu` or `cuda` |

### Examples

```bash
# Run with custom learning rate
python section1_actor_critic.py --env CartPole-v1 --seed 0 --lr 1e-3

# Run on GPU (if available)
python section1_actor_critic.py --env Acrobot-v1 --seed 0 --device cuda

# Run with custom hyperparameters
python section1_actor_critic.py --env MountainCarContinuous-v0 --seed 0 --gamma 0.95 --entropy_coef 0.02
```

### Help

View all available options:

```bash
python section1_actor_critic.py --help
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

## File Structure

```
transfer_learning/
├── section1_actor_critic.py  # Main implementation file
├── requirements.txt           # Python dependencies
├── README.md                  # This file
├── venv/                      # Virtual environment (created during setup)
└── results/                   # Output directory (created automatically)
    ├── CartPole-v1_seed0_summary.json
    ├── CartPole-v1_seed0_training_curve.png
    └── ...
```

**Note**: The `venv/` directory should be added to `.gitignore` if using version control.

## License

This code is part of a university assignment.
