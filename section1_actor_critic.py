"""
Section 1: Training individual networks with Advantage Actor-Critic (A2C)

Unified implementation supporting:
- Discrete action spaces (CartPole-v1, Acrobot-v1) with action masking
- Continuous action space (MountainCarContinuous-v0)
- Fixed input/output dimensions across all environments
"""

import argparse
import json
import os
import time
from collections import deque
from typing import Tuple, Dict, Any

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt


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


class ActorCritic(nn.Module):
    """Unified Actor-Critic network with fixed input/output dimensions."""
    
    def __init__(self, obs_dim: int, action_dim: int, action_type: str, hidden_dim: int = 128):
        super(ActorCritic, self).__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.action_type = action_type
        
        # Shared feature extractor
        self.feature = nn.Sequential(
            nn.Linear(OBS_DIM, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        # Actor head: outputs action_dim values
        # For discrete: logits for each action
        # For continuous: mean (and we'll use a learnable std or fixed std)
        self.actor = nn.Linear(hidden_dim, ACTION_DIM)
        
        # Critic head: outputs value estimate
        self.critic = nn.Linear(hidden_dim, 1)
        
        # For continuous actions: learnable log_std (one per action dimension)
        if action_type == 'continuous':
            self.log_std = nn.Parameter(torch.zeros(action_dim))
    
    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        Returns: (action_distribution, value)
        """
        # Pad observation to OBS_DIM
        obs_padded = self._pad_observation(obs)
        
        features = self.feature(obs_padded)
        value = self.critic(features)
        
        if self.action_type == 'discrete':
            logits = self.actor(features)
            return logits, value
        else:  # continuous
            mean = self.actor(features)[:, :self.action_dim]  # Use first action_dim outputs
            std = torch.exp(self.log_std.clamp(-20, 2))  # Clamp for numerical stability
            return (mean, std), value
    
    def _pad_observation(self, obs: torch.Tensor) -> torch.Tensor:
        """Pad observation to fixed OBS_DIM."""
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        current_dim = obs.shape[-1]
        if current_dim < OBS_DIM:
            padding = torch.zeros(obs.shape[0], OBS_DIM - current_dim, device=obs.device)
            obs_padded = torch.cat([obs, padding], dim=-1)
        else:
            obs_padded = obs
        return obs_padded
    
    def get_action(self, obs: np.ndarray, action_mask: np.ndarray = None) -> Tuple[Any, torch.Tensor, torch.Tensor]:
        """
        Sample an action from the policy.
        Returns: (action, log_prob, value)
        """
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
        
        with torch.no_grad():
            if self.action_type == 'discrete':
                logits, value = self.forward(obs_tensor)
                
                # Apply action mask: set invalid actions to -inf
                if action_mask is not None:
                    mask_tensor = torch.FloatTensor(action_mask).unsqueeze(0)
                    logits = logits.masked_fill(mask_tensor == 0, float('-inf'))
                
                dist = torch.distributions.Categorical(logits=logits)
                action = dist.sample()
                log_prob = dist.log_prob(action)
                
                # Return action as numpy scalar
                return action.item(), log_prob, value.squeeze()
            else:  # continuous
                (mean, std), value = self.forward(obs_tensor)
                dist = torch.distributions.Normal(mean, std)
                action = dist.sample()
                log_prob = dist.log_prob(action).sum(dim=-1)
                # Clamp action to valid range (typically [-1, 1] for MountainCarContinuous)
                action = torch.clamp(action, -1.0, 1.0)
                # Return as numpy array, preserving shape (1,) for MountainCarContinuous
                action_np = action.cpu().numpy().flatten()
                return action_np, log_prob, value.squeeze()
    
    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor, action_mask: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions for computing policy loss.
        Returns: (log_probs, values, entropy)
        """
        obs_padded = self._pad_observation(obs)
        features = self.feature(obs_padded)
        values = self.critic(features).squeeze()
        
        if self.action_type == 'discrete':
            logits = self.actor(features)
            
            # Apply action mask
            if action_mask is not None:
                logits = logits.masked_fill(action_mask == 0, float('-inf'))
            
            dist = torch.distributions.Categorical(logits=logits)
            log_probs = dist.log_prob(actions)
            entropy = dist.entropy()
        else:  # continuous
            mean = self.actor(features)[:, :self.action_dim]
            std = torch.exp(self.log_std.clamp(-20, 2))
            dist = torch.distributions.Normal(mean, std)
            log_probs = dist.log_prob(actions).sum(dim=-1)
            entropy = dist.entropy().sum(dim=-1)
        
        return log_probs, values, entropy


class A2CAgent:
    """Advantage Actor-Critic agent."""
    
    def __init__(
        self,
        env_name: str,
        seed: int = 0,
        lr: float = 3e-4,
        gamma: float = 0.99,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        device: str = 'cpu'
    ):
        self.env_name = env_name
        self.seed = seed
        self.gamma = gamma
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.device = device
        
        # Set seeds for reproducibility
        self._set_seeds(seed)
        
        # Create environment
        self.env = gym.make(env_name)
        self.env.reset(seed=seed)
        
        # Get environment config
        self.config = ENV_CONFIGS[env_name]
        self.action_type = self.config['action_type']
        self.target_return = self.config['target_return']
        self.max_episodes = self.config['max_episodes']
        
        # Create network
        self.network = ActorCritic(
            obs_dim=self.config['obs_dim'],
            action_dim=self.config['action_dim'],
            action_type=self.action_type
        ).to(device)
        
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        
        # Training statistics
        self.episode_returns = deque(maxlen=100)
        self.all_returns = []
        self.episode_lengths = []
        
    def _set_seeds(self, seed: int):
        """Set seeds for reproducibility."""
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        import random
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
    
    def _get_action_mask(self) -> np.ndarray:
        """Get action mask for discrete actions (1 = valid, 0 = invalid/dummy)."""
        if self.action_type == 'discrete':
            mask = np.ones(ACTION_DIM, dtype=np.float32)
            # Mask out dummy actions (actions beyond the actual action space)
            mask[self.config['action_dim']:] = 0
            return mask
        return None
    
    def train(self) -> Dict[str, Any]:
        """Train the agent until convergence or max episodes."""
        start_time = time.time()
        episode = 0
        
        print(f"Training {self.env_name} with seed {self.seed}")
        print(f"Target return: {self.target_return}")
        
        while episode < self.max_episodes:
            episode_return = 0
            episode_length = 0
            
            obs, _ = self.env.reset()
            done = False
            
            # Storage for one episode
            observations = []
            actions = []
            rewards = []
            log_probs = []
            values = []
            dones = []
            
            # Collect one episode
            while not done:
                action_mask = self._get_action_mask()
                action, log_prob, value = self.network.get_action(obs, action_mask)
                
                observations.append(obs)
                actions.append(action)
                log_probs.append(log_prob)
                values.append(value)
                
                # Step environment
                if self.action_type == 'continuous':
                    # Continuous action: MountainCarContinuous expects array-like with shape (1,)
                    # action is already a numpy array from get_action
                    action_env = action if action.ndim > 0 else np.array([action], dtype=np.float32)
                else:
                    action_env = int(action)
                
                obs, reward, terminated, truncated, _ = self.env.step(action_env)
                done = terminated or truncated
                
                rewards.append(reward)
                dones.append(done)
                
                episode_return += reward
                episode_length += 1
            
            # Store episode statistics
            self.episode_returns.append(episode_return)
            self.all_returns.append(episode_return)
            self.episode_lengths.append(episode_length)
            episode += 1
            
            # Update network
            self._update(observations, actions, rewards, log_probs, values, dones)
            
            # Logging
            if episode % 100 == 0:
                avg_return = np.mean(self.episode_returns) if len(self.episode_returns) > 0 else 0
                print(f"Episode {episode}, Avg Return (last 100): {avg_return:.2f}, "
                      f"Last Return: {episode_return:.2f}")
            
            # Check convergence
            if len(self.episode_returns) >= 100:
                avg_return = np.mean(self.episode_returns)
                if avg_return >= self.target_return:
                    elapsed_time = time.time() - start_time
                    print(f"\nConverged! Episode {episode}, Avg Return: {avg_return:.2f}")
                    print(f"Time elapsed: {elapsed_time:.2f} seconds")
                    break
        
        elapsed_time = time.time() - start_time
        final_avg_return = np.mean(self.episode_returns) if len(self.episode_returns) > 0 else 0
        
        return {
            'episodes': episode,
            'runtime_seconds': elapsed_time,
            'final_avg_return': final_avg_return,
            'converged': episode < self.max_episodes
        }
    
    def _update(self, observations, actions, rewards, log_probs, values, dones):
        """Update network using A2C algorithm."""
        # Convert to tensors
        obs_tensor = torch.FloatTensor(np.array(observations)).to(self.device)
        if self.action_type == 'discrete':
            action_tensor = torch.LongTensor(actions).to(self.device)
        else:
            action_tensor = torch.FloatTensor(np.array(actions)).to(self.device)
        reward_tensor = torch.FloatTensor(rewards).to(self.device)
        done_tensor = torch.FloatTensor(dones).to(self.device)
        
        # Get action mask if discrete
        action_mask = None
        if self.action_type == 'discrete':
            mask = self._get_action_mask()
            action_mask = torch.FloatTensor(mask).unsqueeze(0).expand(len(actions), -1).to(self.device)
        
        # Compute returns and advantages
        returns = self._compute_returns(rewards, dones)
        returns_tensor = torch.FloatTensor(returns).to(self.device)
        
        # Evaluate current policy
        new_log_probs, new_values, entropy = self.network.evaluate_actions(
            obs_tensor, action_tensor, action_mask
        )
        
        # Compute advantages
        advantages = returns_tensor - new_values.detach()
        
        # Compute losses
        policy_loss = -(new_log_probs * advantages).mean()
        value_loss = F.mse_loss(new_values, returns_tensor)
        entropy_loss = -entropy.mean()
        
        total_loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss
        
        # Update
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
        self.optimizer.step()
    
    def _compute_returns(self, rewards: list, dones: list) -> list:
        """Compute discounted returns."""
        returns = []
        G = 0
        for reward, done in zip(reversed(rewards), reversed(dones)):
            if done:
                G = reward
            else:
                G = reward + self.gamma * G
            returns.insert(0, G)
        return returns
    
    def save_results(self, results: Dict[str, Any], output_dir: str = 'results'):
        """Save training results and plots."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save summary
        summary = {
            'env_name': self.env_name,
            'seed': self.seed,
            'hyperparameters': {
                'lr': 3e-4,
                'gamma': self.gamma,
                'value_coef': self.value_coef,
                'entropy_coef': self.entropy_coef,
                'max_grad_norm': self.max_grad_norm,
            },
            'runtime_seconds': results['runtime_seconds'],
            'episodes_to_converge': results['episodes'],
            'final_avg_return': results['final_avg_return'],
            'converged': results['converged'],
        }
        
        summary_path = os.path.join(output_dir, f'{self.env_name}_seed{self.seed}_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Save training curve
        self._plot_training_curve(output_dir)
        
        print(f"Results saved to {output_dir}/")
    
    def _plot_training_curve(self, output_dir: str):
        """Plot and save training curve."""
        returns = np.array(self.all_returns)
        episodes = np.arange(1, len(returns) + 1)
        
        # Compute moving average
        window = 100
        if len(returns) >= window:
            moving_avg = np.convolve(returns, np.ones(window)/window, mode='valid')
            moving_episodes = episodes[window-1:]
        else:
            moving_avg = returns
            moving_episodes = episodes
        
        plt.figure(figsize=(10, 6))
        plt.plot(episodes, returns, alpha=0.3, label='Episode Return', color='blue')
        plt.plot(moving_episodes, moving_avg, label=f'Moving Average ({window} episodes)', color='red', linewidth=2)
        plt.axhline(y=self.target_return, color='green', linestyle='--', label=f'Target Return ({self.target_return})')
        plt.xlabel('Episode')
        plt.ylabel('Return')
        plt.title(f'Training Curve: {self.env_name} (seed {self.seed})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plot_path = os.path.join(output_dir, f'{self.env_name}_seed{self.seed}_training_curve.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()


def main():
    parser = argparse.ArgumentParser(description='Train A2C on specified environment')
    parser.add_argument('--env', type=str, required=True,
                        choices=['CartPole-v1', 'Acrobot-v1', 'MountainCarContinuous-v0'],
                        help='Environment name')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--value_coef', type=float, default=0.5, help='Value loss coefficient')
    parser.add_argument('--entropy_coef', type=float, default=0.01, help='Entropy coefficient')
    parser.add_argument('--device', type=str, default='cpu', help='Device (cpu/cuda)')
    
    args = parser.parse_args()
    
    # Create agent
    agent = A2CAgent(
        env_name=args.env,
        seed=args.seed,
        lr=args.lr,
        gamma=args.gamma,
        value_coef=args.value_coef,
        entropy_coef=args.entropy_coef,
        device=args.device
    )
    
    # Train
    results = agent.train()
    
    # Save results
    agent.save_results(results)
    
    print("\nTraining completed!")


if __name__ == '__main__':
    main()
