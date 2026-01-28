"""
Common code shared across all training scripts.
Contains models, utilities, and training functions.
"""

import json
import os
import random
import time
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical, Normal
import matplotlib.pyplot as plt


# ============================================================================
# Constants and Configuration
# ============================================================================

# Fixed dimensions across all environments
OBS_DIM = 6  # Max observation dimension (Acrobot-v1)
ACTION_DIM = 3  # Max discrete action dimension (Acrobot-v1)

# Environment-specific configurations
ENV_CONFIGS = {
    'CartPole-v1': {
        'obs_dim': 4,
        'action_dim': 2,  # discrete
        'action_type': 'discrete',
        'target_return': 475,
        'max_episodes': 10000,
    },
    'Acrobot-v1': {
        'obs_dim': 6,
        'action_dim': 3,  # discrete
        'action_type': 'discrete',
        'target_return': -100,
        'max_episodes': 10000,
    },
    'MountainCarContinuous-v0': {
        'obs_dim': 2,
        'action_dim': 1,  # continuous
        'action_type': 'continuous',
        'target_return': 90,
        'max_episodes': 10000,
    }
}


# ============================================================================
# Utility Functions
# ============================================================================

def set_seed(seed: int, deterministic: bool = True) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def pad_observation(obs: np.ndarray) -> np.ndarray:
    """Pad observation to fixed OBS_DIM."""
    obs = np.array(obs).flatten()
    current_dim = obs.shape[0]
    if current_dim < OBS_DIM:
        padding = np.zeros(OBS_DIM - current_dim)
        obs_padded = np.concatenate([obs, padding])
    else:
        obs_padded = obs[:OBS_DIM]
    return obs_padded


def parse_hidden_sizes(hidden_sizes_str: str) -> List[int]:
    """Parse comma-separated hidden sizes string."""
    return [int(s.strip()) for s in hidden_sizes_str.split(",") if s.strip()]


def get_solved_threshold(env: gym.Env, env_name: str) -> float:
    """Determine solved threshold from env.spec.reward_threshold or fallback dict."""
    # Try to get from env.spec.reward_threshold
    if hasattr(env, 'spec') and env.spec is not None:
        if hasattr(env.spec, 'reward_threshold') and env.spec.reward_threshold is not None:
            return env.spec.reward_threshold
    
    # Fallback to per-env dict
    fallback_thresholds = {
        'CartPole-v1': 475,
        'Acrobot-v1': -100,
        'MountainCarContinuous-v0': 90
    }
    
    if env_name in fallback_thresholds:
        return fallback_thresholds[env_name]
    
    # Default fallback (shouldn't happen for our envs)
    raise ValueError(f"No threshold found for environment {env_name}")


# ============================================================================
# Neural Networks
# ============================================================================

class MLP(nn.Module):
    """Multi-layer perceptron with configurable hidden layers."""
    
    def __init__(self, input_dim: int, output_dim: int, hidden_sizes: List[int], 
                 activation: str = "relu", output_activation: Optional[str] = None):
        super().__init__()
        layers = []
        prev_size = input_dim
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            if activation == "relu":
                layers.append(nn.ReLU())
            elif activation == "tanh":
                layers.append(nn.Tanh())
            elif activation == "elu":
                layers.append(nn.ELU())
            else:
                raise ValueError(f"Unknown activation: {activation}")
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, output_dim))
        if output_activation == "tanh":
            layers.append(nn.Tanh())
        elif output_activation == "sigmoid":
            layers.append(nn.Sigmoid())
        elif output_activation is not None:
            raise ValueError(f"Unknown output activation: {output_activation}")
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PolicyNetwork(nn.Module):
    """Policy network with fixed output dimension and action masking support."""
    
    def __init__(self, obs_dim: int, action_dim: int, action_type: str, hidden_sizes: List[int] = [128, 128]):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.action_type = action_type
        
        # Network always takes padded observations
        self.mlp = MLP(
            input_dim=OBS_DIM,
            output_dim=ACTION_DIM,
            hidden_sizes=hidden_sizes,
            activation="relu",
            output_activation=None
        )
        
        # For continuous actions: learnable log_std
        if action_type == 'continuous':
            self.log_std = nn.Parameter(torch.zeros(action_dim))
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Forward pass. Returns logits for discrete or mean for continuous."""
        # obs should already be padded
        return self.mlp(obs)
    
    def get_action_and_log_prob(self, obs: np.ndarray, action_mask: Optional[np.ndarray] = None, 
                                deterministic: bool = False) -> Tuple[Any, float]:
        """Sample an action and return log probability."""
        obs_padded = pad_observation(obs)
        obs_tensor = torch.FloatTensor(obs_padded).unsqueeze(0)
        
        with torch.no_grad():
            if self.action_type == 'discrete':
                logits = self.forward(obs_tensor)
                
                # Apply action mask: set invalid actions to -inf
                if action_mask is not None:
                    mask_tensor = torch.FloatTensor(action_mask).unsqueeze(0)
                    logits = logits.masked_fill(mask_tensor == 0, float('-inf'))
                
                dist = Categorical(logits=logits)
                if deterministic:
                    action = torch.argmax(logits, dim=-1)
                else:
                    action = dist.sample()
                log_prob = dist.log_prob(action)
                return action.item(), log_prob.item()
            else:  # continuous
                mean = self.forward(obs_tensor)[:, :self.action_dim]
                std = torch.exp(self.log_std.clamp(-20, 2))
                dist = Normal(mean, std)
                if deterministic:
                    action = mean
                else:
                    action = dist.sample()
                log_prob = dist.log_prob(action).sum(dim=-1)
                # Clamp action to valid range
                action = torch.clamp(action, -1.0, 1.0)
                action_np = action.squeeze().cpu().numpy().flatten()
                return action_np, log_prob.item()
    
    def get_log_prob(self, obs: torch.Tensor, actions: torch.Tensor, 
                     action_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Get log probability of actions."""
        if self.action_type == 'discrete':
            logits = self.forward(obs)
            if action_mask is not None:
                logits = logits.masked_fill(action_mask == 0, float('-inf'))
            dist = Categorical(logits=logits)
            return dist.log_prob(actions)
        else:  # continuous
            mean = self.forward(obs)[:, :self.action_dim]
            std = torch.exp(self.log_std.clamp(-20, 2))
            dist = Normal(mean, std)
            return dist.log_prob(actions).sum(dim=-1)
    
    def get_entropy(self, obs: torch.Tensor, action_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute entropy of the action distribution."""
        if self.action_type == 'discrete':
            logits = self.forward(obs)
            if action_mask is not None:
                logits = logits.masked_fill(action_mask == 0, float('-inf'))
            dist = Categorical(logits=logits)
            return dist.entropy()
        else:  # continuous
            mean = self.forward(obs)[:, :self.action_dim]
            std = torch.exp(self.log_std.clamp(-20, 2))
            dist = Normal(mean, std)
            return dist.entropy().sum(dim=-1)


class ValueNetwork(nn.Module):
    """Value network that estimates V(s) with fixed input dimension."""
    
    def __init__(self, obs_dim: int, hidden_sizes: List[int] = [128, 128]):
        super().__init__()
        self.mlp = MLP(
            input_dim=OBS_DIM,
            output_dim=1,
            hidden_sizes=hidden_sizes,
            activation="relu",
            output_activation=None
        )
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Forward pass. obs should already be padded."""
        return self.mlp(obs).squeeze(-1)


# ============================================================================
# Buffer
# ============================================================================

class TrajectoryBuffer:
    """Buffer for storing a single episode trajectory."""
    
    def __init__(self):
        self.states: List[np.ndarray] = []
        self.actions: List[Any] = []
        self.rewards: List[float] = []
        self.log_probs: List[float] = []
        self.values: List[float] = []
        self.dones: List[bool] = []
    
    def add(self, state: np.ndarray, action: Any, reward: float, log_prob: float,
            value: float = 0.0, done: bool = False) -> None:
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.dones.append(done)
    
    def clear(self) -> None:
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.log_probs.clear()
        self.values.clear()
        self.dones.clear()


# ============================================================================
# Actor-Critic Algorithm
# ============================================================================

class ActorCritic:
    """Actor-Critic algorithm with TD-error as advantage."""
    
    def __init__(self, obs_dim: int, action_dim: int, action_type: str, hidden_sizes: list = [128, 128],
                 lr_policy: float = 3e-4, lr_value: float = 1e-3, gamma: float = 0.99,
                 normalize_advantages: bool = False, entropy_coef: float = 0.01, device: str = "cpu",
                 lr_decay: Optional[float] = None, lr_decay_steps: int = 100):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.action_type = action_type
        self.gamma = gamma
        self.normalize_advantages = normalize_advantages
        self.entropy_coef = entropy_coef
        self.device = device
        self.lr_decay = lr_decay
        self.lr_decay_steps = lr_decay_steps
        
        self.policy = PolicyNetwork(obs_dim, action_dim, action_type, hidden_sizes).to(device)
        self.value = ValueNetwork(obs_dim, hidden_sizes).to(device)
        
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=lr_policy)
        self.value_optimizer = optim.Adam(self.value.parameters(), lr=lr_value)
        
        # Learning rate schedulers
        if lr_decay is not None:
            self.policy_scheduler = optim.lr_scheduler.ExponentialLR(self.policy_optimizer, gamma=lr_decay)
            self.value_scheduler = optim.lr_scheduler.ExponentialLR(self.value_optimizer, gamma=lr_decay)
        else:
            self.policy_scheduler = None
            self.value_scheduler = None
        
        self.buffer = TrajectoryBuffer()
    
    def select_action(self, obs: np.ndarray, action_mask: Optional[np.ndarray] = None, 
                     deterministic: bool = False) -> tuple:
        """Select an action and estimate value."""
        obs_padded = pad_observation(obs)
        obs_tensor = torch.FloatTensor(obs_padded).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action, log_prob = self.policy.get_action_and_log_prob(obs, action_mask, deterministic)
            value = self.value(obs_tensor).item()
        
        return action, value
    
    def store_transition(self, obs: np.ndarray, action: Any, reward: float,
                        log_prob: float, value: float, next_obs: Optional[np.ndarray] = None,
                        next_value: Optional[float] = None, done_bootstrap: bool = False) -> None:
        """Store a transition in the buffer."""
        self.buffer.add(obs, action, reward, log_prob, value=value, done=done_bootstrap)
    
    def update_step(self, obs: np.ndarray, action: Any, reward: float, log_prob: float,
                   value: float, next_obs: np.ndarray, next_value: float, done_bootstrap: bool,
                   action_mask: Optional[np.ndarray] = None) -> dict:
        """Update networks using TD-error from a single step."""
        obs_padded = pad_observation(obs)
        next_obs_padded = pad_observation(next_obs)
        obs_tensor = torch.FloatTensor(obs_padded).unsqueeze(0).to(self.device)
        next_obs_tensor = torch.FloatTensor(next_obs_padded).unsqueeze(0).to(self.device)
        
        if self.action_type == 'discrete':
            action_tensor = torch.LongTensor([action]).to(self.device)
        else:
            action_tensor = torch.FloatTensor(np.array(action)).unsqueeze(0).to(self.device)
        
        reward_tensor = torch.FloatTensor([reward]).to(self.device)
        done_bootstrap_tensor = torch.BoolTensor([done_bootstrap]).to(self.device)
        
        with torch.no_grad():
            next_value_tensor = torch.FloatTensor([next_value]).to(self.device)
            target = reward_tensor + self.gamma * (1 - done_bootstrap_tensor.float()) * next_value_tensor
        
        value_tensor = self.value(obs_tensor)
        td_error = target - value_tensor
        
        # Get action mask tensor if discrete
        action_mask_tensor = None
        if self.action_type == 'discrete' and action_mask is not None:
            action_mask_tensor = torch.FloatTensor(action_mask).unsqueeze(0).to(self.device)
        
        log_prob_tensor = self.policy.get_log_prob(obs_tensor, action_tensor, action_mask_tensor)
        policy_loss = -(log_prob_tensor * td_error.detach()).mean()
        
        entropy = self.policy.get_entropy(obs_tensor, action_mask_tensor).mean()
        entropy_bonus = self.entropy_coef * entropy
        policy_loss_total = policy_loss - entropy_bonus
        
        value_loss = td_error.pow(2).mean()
        
        self.policy_optimizer.zero_grad()
        policy_loss_total.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
        self.policy_optimizer.step()
        
        self.value_optimizer.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.value.parameters(), max_norm=0.5)
        self.value_optimizer.step()
        
        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "td_error": td_error.item(),
            "entropy": entropy.item()
        }
    
    def update_episode(self, action_mask: Optional[np.ndarray] = None) -> dict:
        """Update networks using all transitions from the episode."""
        if len(self.buffer.rewards) == 0:
            return {}
        
        # Pad all observations
        states_padded = [pad_observation(s) for s in self.buffer.states[:-1]]
        next_states_padded = [pad_observation(s) for s in self.buffer.states[1:]]
        
        states = torch.FloatTensor(np.array(states_padded)).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states_padded)).to(self.device)
        
        if self.action_type == 'discrete':
            actions = torch.LongTensor(self.buffer.actions).to(self.device)
        else:
            actions = torch.FloatTensor(np.array(self.buffer.actions)).to(self.device)
        
        rewards = torch.FloatTensor(self.buffer.rewards).to(self.device)
        dones = torch.BoolTensor(self.buffer.dones).to(self.device)
        
        # Current values need gradients for value loss
        current_values = self.value(states)
        
        # Next values should be detached for bootstrapping (target)
        with torch.no_grad():
            next_values = self.value(next_states)
        
        targets = rewards + self.gamma * (1 - dones.float()) * next_values
        td_errors = targets - current_values
        
        # Separate advantages for policy loss (normalized if needed) from raw td_errors for value loss
        advantages = td_errors.detach()
        if self.normalize_advantages:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Get action mask tensor if discrete
        action_mask_tensor = None
        if self.action_type == 'discrete' and action_mask is not None:
            action_mask_tensor = torch.FloatTensor(action_mask).unsqueeze(0).expand(len(actions), -1).to(self.device)
        
        log_probs = self.policy.get_log_prob(states, actions, action_mask_tensor)
        policy_loss = -(log_probs * advantages).mean()
        
        entropy = self.policy.get_entropy(states, action_mask_tensor).mean()
        entropy_bonus = self.entropy_coef * entropy
        policy_loss_total = policy_loss - entropy_bonus
        
        # Value loss always uses raw td_errors (un-normalized)
        value_loss = td_errors.pow(2).mean()
        
        self.policy_optimizer.zero_grad()
        policy_loss_total.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
        self.policy_optimizer.step()
        
        self.value_optimizer.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.value.parameters(), max_norm=0.5)
        self.value_optimizer.step()
        
        self.buffer.clear()
        
        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "mean_td_error": td_errors.mean().item(),
            "std_td_error": td_errors.std().item(),
            "entropy": entropy.item()
        }
    
    def save(self, filepath_prefix: str) -> None:
        """Save both networks."""
        torch.save(self.policy.state_dict(), f"{filepath_prefix}_policy.pt")
        torch.save(self.value.state_dict(), f"{filepath_prefix}_value.pt")
    
    def load(self, filepath_prefix: str) -> None:
        """Load both networks."""
        self.policy.load_state_dict(torch.load(f"{filepath_prefix}_policy.pt", map_location=self.device))
        self.value.load_state_dict(torch.load(f"{filepath_prefix}_value.pt", map_location=self.device))
    
    def step_lr(self, step: int) -> None:
        """Step learning rate schedulers if enabled."""
        if self.policy_scheduler is not None and self.value_scheduler is not None:
            # Step every lr_decay_steps
            if step % self.lr_decay_steps == 0 and step > 0:
                self.policy_scheduler.step()
                self.value_scheduler.step()
    
    def get_current_lrs(self) -> dict:
        """Get current learning rates."""
        return {
            "lr_policy": self.policy_optimizer.param_groups[0]['lr'],
            "lr_value": self.value_optimizer.param_groups[0]['lr']
        }


# ============================================================================
# Evaluation
# ============================================================================

def evaluate_policy(policy: PolicyNetwork, env: gym.Env, action_type: str, 
                   n_episodes: int = 10, deterministic: bool = True, device: str = "cpu", 
                   render: bool = False) -> dict:
    """Evaluate a policy over multiple episodes."""
    episode_returns = []
    episode_lengths = []
    
    # Get action mask for discrete actions
    def get_action_mask():
        if action_type == 'discrete':
            mask = np.ones(ACTION_DIM, dtype=np.float32)
            if hasattr(env, 'action_space') and hasattr(env.action_space, 'n'):
                mask[env.action_space.n:] = 0
            return mask
        return None
    
    for episode in range(n_episodes):
        obs, info = env.reset()
        episode_return = 0.0
        episode_length = 0
        done = False
        
        while not done:
            if render:
                env.render()
            
            action_mask = get_action_mask()
            action, _ = policy.get_action_and_log_prob(obs, action_mask, deterministic)
            
            if action_type == 'continuous':
                if isinstance(action, np.ndarray):
                    action_env = action if action.ndim > 0 else np.array([action], dtype=np.float32)
                else:
                    action_env = np.array([action], dtype=np.float32)
            else:
                action_env = int(action)
            
            obs, reward, terminated, truncated, info = env.step(action_env)
            done = terminated or truncated
            episode_return += reward
            episode_length += 1
        
        episode_returns.append(episode_return)
        episode_lengths.append(episode_length)
    
    return {
        "mean_return": np.mean(episode_returns),
        "std_return": np.std(episode_returns),
        "min_return": np.min(episode_returns),
        "max_return": np.max(episode_returns),
        "mean_length": np.mean(episode_lengths),
        "std_length": np.std(episode_lengths),
        "episode_returns": episode_returns,
        "episode_lengths": episode_lengths
    }


# ============================================================================
# Logging and Plotting
# ============================================================================

class MetricsLogger:
    """Logs metrics and saves to JSON."""
    
    def __init__(self, artifact_dir: Optional[str] = None, run_name: str = "run", no_save: bool = False):
        self.no_save = no_save
        self.run_name = run_name
        
        if no_save:
            self.artifact_dir = None
        else:
            self.artifact_dir = Path(artifact_dir) if artifact_dir else None
            
            if self.artifact_dir:
                self.artifact_dir.mkdir(parents=True, exist_ok=True)
                (self.artifact_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
                (self.artifact_dir / "plots").mkdir(parents=True, exist_ok=True)
        
        self.metrics: Dict[str, list] = {}
    
    def log_scalar(self, tag: str, value: float, step: int) -> None:
        """Log a scalar value and store in memory."""
        if tag not in self.metrics:
            self.metrics[tag] = []
        self.metrics[tag].append({"step": step, "value": value})
    
    def save_metrics(self, filename: str = "metrics.json") -> None:
        """Save all logged metrics to JSON file."""
        if self.no_save or not self.artifact_dir:
            return
        metrics_file = self.artifact_dir / filename
        with open(metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        print(f"Metrics saved to {metrics_file}")
    
    def get_metrics(self) -> Dict[str, list]:
        """Get all logged metrics."""
        return self.metrics.copy()


def moving_average(data: np.ndarray, window: int = 100) -> np.ndarray:
    """Compute moving average of data."""
    if len(data) < window:
        return np.convolve(data, np.ones(window) / window, mode='valid')
    return np.convolve(data, np.ones(window) / window, mode='valid')


def plot_metrics(metrics: Dict[str, List[Dict[str, float]]], output_dir: Optional[Path], 
                 env_name: str, seed: int, target_return: float, window: int = 100) -> None:
    """Generate plots from metrics dictionary."""
    if output_dir is None:
        return
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if "episode_return" in metrics:
        returns = [m["value"] for m in metrics["episode_return"]]
        steps = [m["step"] for m in metrics["episode_return"]]
        
        plt.figure(figsize=(10, 6))
        plt.plot(steps, returns, alpha=0.3, label="Episode Return", color="blue")
        
        if len(returns) >= window:
            ma_returns = moving_average(np.array(returns), window)
            ma_steps = steps[window-1:]
            plt.plot(ma_steps, ma_returns, label=f"{window}-Episode Moving Average", 
                    color="red", linewidth=2)
        
        plt.axhline(y=target_return, color='green', linestyle='--', label=f'Target Return ({target_return})')
        plt.xlabel("Episode")
        plt.ylabel("Return")
        plt.title(f'Training Curve: {env_name} (seed {seed})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / f"{env_name}_seed{seed}_training_curve.png", dpi=150)
        plt.close()
    
    print(f"Plots saved to {output_dir}")


# ============================================================================
# Training Function
# ============================================================================

def train(config: dict) -> Dict[str, Any]:
    """
    Train Actor-Critic agent with given configuration.
    
    Returns:
        Dictionary with training results
    """
    set_seed(config['seed'])
    
    env = gym.make(config['env'])
    env.reset(seed=config['seed'])
    
    env_config = ENV_CONFIGS[config['env']]
    obs_dim = env_config['obs_dim']
    action_dim = env_config['action_dim']
    action_type = env_config['action_type']
    max_episodes = config.get('max_episodes') or env_config['max_episodes']
    
    # Get solved threshold from env.spec or fallback
    solved_threshold = get_solved_threshold(env, config['env'])
    
    if not config.get('no_save', False):
        print(f"Environment: {config['env']}")
        print(f"Observation dim: {obs_dim} (padded to {OBS_DIM})")
        print(f"Action dim: {action_dim} (output dim: {ACTION_DIM})")
        print(f"Action type: {action_type}")
        print(f"Solved threshold: {solved_threshold}")
    
    agent = ActorCritic(
        obs_dim=obs_dim,
        action_dim=action_dim,
        action_type=action_type,
        hidden_sizes=config['hidden_sizes'],
        lr_policy=config['lr_policy'],
        lr_value=config['lr_value'],
        gamma=config['gamma'],
        normalize_advantages=config['normalize_advantages'],
        entropy_coef=config['entropy_coef'],
        device="cpu",
        lr_decay=config.get('lr_decay'),
        lr_decay_steps=config.get('lr_decay_steps', 100)
    )
    
    artifact_dir = None if config.get('no_save') else config.get('artifact_dir', 'results')
    logger = MetricsLogger(
        artifact_dir=artifact_dir,
        run_name=f"actor_critic_{config['env']}_{config['seed']}",
        no_save=config.get('no_save', False)
    )
    
    # Get action mask function
    def get_action_mask():
        if action_type == 'discrete':
            mask = np.ones(ACTION_DIM, dtype=np.float32)
            mask[action_dim:] = 0  # Mask dummy actions
            return mask
        return None
    
    best_mean_return = float('-inf')
    episodes_to_converge = None
    updates_to_converge = None
    env_steps_to_converge = None
    total_time_seconds_to_converge = None
    
    step_count = 0
    env_steps = 0
    episode_returns = []
    episode_returns_deque = deque(maxlen=100)  # For moving average calculation
    
    start_time = time.perf_counter()  # Use perf_counter for better precision
    
    if not config.get('no_save', False):
        print("\nStarting training...")
        print(f"Max episodes: {max_episodes}")
        print(f"Update mode: {config['update_mode']}")
        print(f"Gamma: {config['gamma']}, Policy LR: {config['lr_policy']}, Value LR: {config['lr_value']}")
        print("-" * 50)
    
    for episode in range(1, max_episodes + 1):
        obs, info = env.reset()
        episode_return = 0.0
        episode_length = 0
        done = False
        final_done_bootstrap = False
        
        while not done and episode_length < config['max_steps']:
            action_mask = get_action_mask()
            action, value = agent.select_action(obs, action_mask, deterministic=False)
            
            # Get log prob for storage
            obs_padded = pad_observation(obs)
            obs_tensor = torch.FloatTensor(obs_padded).unsqueeze(0)
            if action_type == 'discrete':
                action_tensor = torch.LongTensor([action])
            else:
                action_tensor = torch.FloatTensor(np.array(action)).unsqueeze(0)
            action_mask_tensor = None
            if action_type == 'discrete' and action_mask is not None:
                action_mask_tensor = torch.FloatTensor(action_mask).unsqueeze(0)
            log_prob = agent.policy.get_log_prob(obs_tensor, action_tensor, action_mask_tensor).item()
            
            # Format action for environment
            if action_type == 'discrete':
                action_env = int(action)
            else:
                # Continuous: ensure it's a numpy array
                if isinstance(action, np.ndarray):
                    action_env = action if action.ndim > 0 else np.array([action], dtype=np.float32)
                else:
                    action_env = np.array([action], dtype=np.float32)
            
            next_obs, reward, terminated, truncated, info = env.step(action_env)
            done_env = terminated or truncated
            done_bootstrap = terminated  # Only true termination, not truncation
            final_done_bootstrap = done_bootstrap
            
            if not done_env:
                _, next_value = agent.select_action(next_obs, action_mask, deterministic=False)
            else:
                next_value = 0.0
            
            if config['update_mode'] == "step":
                metrics = agent.update_step(
                    obs, action, reward, log_prob, value,
                    next_obs, next_value, done_bootstrap, action_mask
                )
                if metrics:
                    logger.log_scalar("td_error", metrics.get("td_error", 0), step_count)
                    logger.log_scalar("policy_loss", metrics.get("policy_loss", 0), step_count)
                    logger.log_scalar("value_loss", metrics.get("value_loss", 0), step_count)
                step_count += 1
                agent.step_lr(step_count)
            else:
                agent.store_transition(
                    obs, action, reward, log_prob, value,
                    next_obs=next_obs, next_value=next_value, done_bootstrap=done_bootstrap
                )
            
            obs = next_obs
            episode_return += reward
            episode_length += 1
            done = done_env
        
        # Track environment steps
        env_steps += episode_length
            
        if config['update_mode'] == "episode":
            # Add final state and value
            if not final_done_bootstrap:
                _, final_value = agent.select_action(obs, get_action_mask(), deterministic=False)
                agent.buffer.states.append(obs)
                agent.buffer.values.append(final_value)
            else:
                agent.buffer.states.append(obs)
                agent.buffer.values.append(0.0)
            
            metrics = agent.update_episode(get_action_mask())
            if metrics:
                logger.log_scalar("mean_td_error", metrics.get("mean_td_error", 0), episode)
                logger.log_scalar("std_td_error", metrics.get("std_td_error", 0), episode)
                logger.log_scalar("policy_loss", metrics.get("policy_loss", 0), episode)
                logger.log_scalar("value_loss", metrics.get("value_loss", 0), episode)
                if "entropy" in metrics:
                    logger.log_scalar("entropy", metrics["entropy"], episode)
            agent.step_lr(episode)
        
        episode_returns.append(episode_return)
        episode_returns_deque.append(episode_return)  # Add to deque for moving average
        logger.log_scalar("episode_return", episode_return, episode)
        logger.log_scalar("episode_length", episode_length, episode)
        
        # Log learning rates if decay is enabled
        if config.get('lr_decay') is not None:
            current_lrs = agent.get_current_lrs()
            logger.log_scalar("lr_policy", current_lrs['lr_policy'], episode)
            logger.log_scalar("lr_value", current_lrs['lr_value'], episode)
        
        # Check convergence: moving average of last 100 episodes >= solved_threshold
        # Only check after episode >= 100
        if episode >= 100 and len(episode_returns_deque) == 100:
            ma_last_100 = sum(episode_returns_deque) / 100
            if episodes_to_converge is None and ma_last_100 >= solved_threshold:
                # Convergence achieved!
                episodes_to_converge = episode
                env_steps_to_converge = env_steps
                total_time_seconds_to_converge = time.perf_counter() - start_time
                
                # Determine updates_to_converge
                if config['update_mode'] == "step":
                    updates_to_converge = step_count
                else:
                    updates_to_converge = episode
                
                # Print convergence summary block
                if not config.get('no_save', False):
                    print("\n" + "=" * 60)
                    print("CONVERGENCE ACHIEVED")
                    print("=" * 60)
                    print(f"Environment: {config['env']}")
                    print(f"Seed: {config['seed']}")
                    print(f"Threshold: {solved_threshold}")
                    print(f"Episodes to converge: {episodes_to_converge}")
                    print(f"Env steps to converge: {env_steps_to_converge}")
                    print(f"Updates to converge: {updates_to_converge}")
                    print(f"Time to converge (s): {total_time_seconds_to_converge:.2f}")
                    print(f"Final avg return (last 100): {ma_last_100:.2f}")
                    print("=" * 60 + "\n")
                
                # Save convergence stats to JSON
                if artifact_dir:
                    stats_dir = Path(artifact_dir) / config['env']
                    os.makedirs(stats_dir, exist_ok=True)
                    stats_path = stats_dir / "stats.json"
                    
                    stats = {
                        'env_name': config['env'],
                        'seed': config['seed'],
                        'threshold': solved_threshold,
                        'episodes_to_converge': episodes_to_converge,
                        'updates_to_converge': updates_to_converge,
                        'env_steps_to_converge': env_steps_to_converge,
                        'total_time_seconds_to_converge': total_time_seconds_to_converge,
                        'final_avg_return_last_100': ma_last_100,
                        'converged': True
                    }
                    
                    with open(stats_path, 'w') as f:
                        json.dump(stats, f, indent=2)
                    if not config.get('no_save', False):
                        print(f"Convergence stats saved to: {stats_path}")
                
                break
        
        if episode % config['eval_interval'] == 0 or episode == max_episodes:
            eval_results = evaluate_policy(
                agent.policy,
                env,
                action_type,
                n_episodes=config['eval_episodes'],
                deterministic=True,
                device="cpu",
                render=False
            )
            mean_return = eval_results["mean_return"]
            logger.log_scalar("eval_mean_return", mean_return, episode)
            
            if len(episode_returns_deque) == 100:
                ma_last_100 = sum(episode_returns_deque) / 100
            elif len(episode_returns) >= 100:
                ma_last_100 = sum(episode_returns[-100:]) / 100
            else:
                ma_last_100 = sum(episode_returns) / len(episode_returns) if episode_returns else 0.0
            
            current_lrs = agent.get_current_lrs()
            if not config.get('no_save', False):
                print(f"Episode {episode:4d} | "
                      f"Return: {episode_return:7.2f} | "
                      f"MA (last 100): {ma_last_100:7.2f} | "
                      f"Eval Return: {mean_return:7.2f} ± {eval_results['std_return']:.2f} | "
                      f"Policy Loss: {metrics.get('policy_loss', 0):.4f} | "
                      f"Value Loss: {metrics.get('value_loss', 0):.4f}")
                
                if config['update_mode'] == "episode" and "mean_td_error" in metrics:
                    print(f"  TD Error: {metrics['mean_td_error']:.4f} ± {metrics['std_td_error']:.4f}")
                
                if config.get('lr_decay') is not None:
                    print(f"  LR: policy={current_lrs['lr_policy']:.6f}, value={current_lrs['lr_value']:.6f}")
            
            if mean_return > best_mean_return:
                best_mean_return = mean_return
                if artifact_dir:
                    agent.save(f"{artifact_dir}/checkpoints/agent_best")
    
    elapsed_time = time.perf_counter() - start_time
    
    if artifact_dir:
        agent.save(f"{artifact_dir}/checkpoints/agent_final")
    
    logger.save_metrics()
    plot_metrics(logger.get_metrics(), Path(artifact_dir) / "plots" if artifact_dir else None,
                 config['env'], config['seed'], solved_threshold)
    
    # Save summary
    if artifact_dir:
        # Calculate final average return
        if len(episode_returns_deque) == 100:
            final_avg_return = sum(episode_returns_deque) / 100
        elif len(episode_returns) >= 100:
            final_avg_return = sum(episode_returns[-100:]) / 100
        else:
            final_avg_return = sum(episode_returns) / len(episode_returns) if episode_returns else 0.0
        
        summary = {
            'env_name': config['env'],
            'seed': config['seed'],
            'threshold': solved_threshold,
            'hyperparameters': {
                'lr_policy': config['lr_policy'],
                'lr_value': config['lr_value'],
                'gamma': config['gamma'],
                'entropy_coef': config['entropy_coef'],
                'normalize_advantages': config['normalize_advantages'],
                'hidden_sizes': config['hidden_sizes'],
                'max_episodes': max_episodes,
                'update_mode': config['update_mode'],
            },
            'runtime_seconds': elapsed_time,
            'episodes_to_converge': episodes_to_converge if episodes_to_converge else max_episodes,
            'updates_to_converge': updates_to_converge,
            'env_steps_to_converge': env_steps_to_converge if env_steps_to_converge else env_steps,
            'total_time_seconds_to_converge': total_time_seconds_to_converge,
            'final_avg_return': final_avg_return,
            'converged': episodes_to_converge is not None
        }
        
        summary_path = Path(artifact_dir) / f"{config['env']}_seed{config['seed']}_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
    
    # Calculate final average return
    if len(episode_returns_deque) == 100:
        final_avg_return = sum(episode_returns_deque) / 100
    elif len(episode_returns) >= 100:
        final_avg_return = sum(episode_returns[-100:]) / 100
    else:
        final_avg_return = sum(episode_returns) / len(episode_returns) if episode_returns else 0.0
    
    if not config.get('no_save', False):
        print("\n" + "=" * 50)
        print("Training Summary")
        print("=" * 50)
        print(f"Environment: {config['env']}")
        print(f"Seed: {config['seed']}")
        print(f"Best eval return: {best_mean_return:.2f}")
        if episodes_to_converge:
            print(f"Episodes to converge: {episodes_to_converge}")
            print(f"Env steps to converge: {env_steps_to_converge}")
            print(f"Updates to converge: {updates_to_converge}")
            print(f"Time to converge: {total_time_seconds_to_converge:.2f} seconds")
        else:
            print(f"Did not reach threshold {solved_threshold}")
            print(f"Final episodes: {len(episode_returns)}")
            print(f"Final env steps: {env_steps}")
        print(f"Total runtime: {elapsed_time:.2f} seconds")
        print(f"Final avg return (last 100): {final_avg_return:.2f}")
        if artifact_dir:
            print(f"Results saved to: {artifact_dir}/")
        print("=" * 50)
    
    env.close()
    
    return {
        'episodes': episodes_to_converge if episodes_to_converge else len(episode_returns),
        'updates': updates_to_converge,
        'env_steps': env_steps_to_converge if env_steps_to_converge else env_steps,
        'runtime_seconds': elapsed_time,
        'time_to_converge_seconds': total_time_seconds_to_converge,
        'final_avg_return': final_avg_return,
        'converged': episodes_to_converge is not None
    }
