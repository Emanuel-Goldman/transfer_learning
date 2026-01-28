"""
Training script for MountainCarContinuous-v0 with Optuna hyperparameter optimization.
"""

import argparse
import optuna
from optuna.pruners import MedianPruner
from optuna.exceptions import TrialPruned

from common import (
    train, ENV_CONFIGS, parse_hidden_sizes, set_seed, ActorCritic, 
    pad_observation, get_solved_threshold, evaluate_policy, OBS_DIM, ACTION_DIM
)
import gymnasium as gym
import numpy as np
import torch
import time
from collections import deque


ENV_NAME = 'MountainCarContinuous-v0'


def train_with_pruning(trial, seed: int, max_episodes: int, eval_interval: int = 50):
    """Train agent with Optuna pruning support."""
    # Sample hyperparameters
    lr_policy = trial.suggest_float("lr_policy", 1e-5, 1e-2, log=True)
    lr_value = trial.suggest_float("lr_value", 1e-5, 1e-2, log=True)
    gamma = trial.suggest_float("gamma", 0.9, 0.99)
    entropy_coef = trial.suggest_float("entropy_coef", 0.0, 0.1)
    normalize_advantages = trial.suggest_categorical("normalize_advantages", [True, False])
    
    # Hidden sizes choices
    hidden_size_choices = [
        [64, 64],
        [128, 128],
        [256, 256],
        [64, 128],
        [128, 256],
    ]
    hidden_sizes_idx = trial.suggest_int("hidden_sizes_idx", 0, len(hidden_size_choices) - 1)
    hidden_sizes = hidden_size_choices[hidden_sizes_idx]
    
    # Update mode
    update_mode = trial.suggest_categorical("update_mode", ["episode", "step"])
    
    # Learning rate decay
    use_lr_decay = trial.suggest_categorical("use_lr_decay", [True, False])
    if use_lr_decay:
        lr_decay = trial.suggest_float("lr_decay", 0.98, 0.999, log=False)
        lr_decay_steps = trial.suggest_int("lr_decay_steps", 50, 200, step=50)
    else:
        lr_decay = None
        lr_decay_steps = 100
    
    # Print sampled parameters
    lr_decay_str = f"lr_decay={lr_decay:.4f}, lr_decay_steps={lr_decay_steps}" if use_lr_decay else "lr_decay=None"
    print(f"\nTrial {trial.number} - Sampled params:")
    print(f"  lr_policy={lr_policy:.6f}, lr_value={lr_value:.6f}, gamma={gamma:.4f}, "
          f"entropy_coef={entropy_coef:.4f}, normalize_advantages={normalize_advantages}, "
          f"update_mode={update_mode}, hidden_sizes={hidden_sizes}, {lr_decay_str}")
    
    # Set seed for reproducibility
    set_seed(seed)
    
    # Create environment
    env = gym.make(ENV_NAME)
    env.reset(seed=seed)
    
    env_config = ENV_CONFIGS[ENV_NAME]
    obs_dim = env_config['obs_dim']
    action_dim = env_config['action_dim']
    action_type = env_config['action_type']
    solved_threshold = get_solved_threshold(env, ENV_NAME)
    
    # Create agent
    agent = ActorCritic(
        obs_dim=obs_dim,
        action_dim=action_dim,
        action_type=action_type,
        hidden_sizes=hidden_sizes,
        lr_policy=lr_policy,
        lr_value=lr_value,
        gamma=gamma,
        normalize_advantages=normalize_advantages,
        entropy_coef=entropy_coef,
        device="cpu",
        lr_decay=lr_decay,
        lr_decay_steps=lr_decay_steps
    )
    
    # Get action mask function
    def get_action_mask():
        if action_type == 'discrete':
            mask = np.ones(ACTION_DIM, dtype=np.float32)
            mask[action_dim:] = 0
            return mask
        return None
    
    step_count = 0
    episode_returns = []
    best_eval_return = float('-inf')
    max_steps = 500
    eval_episodes = 10
    
    for episode in range(1, max_episodes + 1):
        obs, info = env.reset()
        episode_return = 0.0
        episode_length = 0
        done = False
        final_done_bootstrap = False
        
        while not done and episode_length < max_steps:
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
                if isinstance(action, np.ndarray):
                    action_env = action if action.ndim > 0 else np.array([action], dtype=np.float32)
                else:
                    action_env = np.array([action], dtype=np.float32)
            
            next_obs, reward, terminated, truncated, info = env.step(action_env)
            done_env = terminated or truncated
            done_bootstrap = terminated
            final_done_bootstrap = done_bootstrap
            
            if not done_env:
                _, next_value = agent.select_action(next_obs, action_mask, deterministic=False)
            else:
                next_value = 0.0
            
            if update_mode == "step":
                agent.update_step(
                    obs, action, reward, log_prob, value,
                    next_obs, next_value, done_bootstrap, action_mask
                )
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
        
        if update_mode == "episode":
            if not final_done_bootstrap:
                _, final_value = agent.select_action(obs, get_action_mask(), deterministic=False)
                agent.buffer.states.append(obs)
                agent.buffer.values.append(final_value)
            else:
                agent.buffer.states.append(obs)
                agent.buffer.values.append(0.0)
            
            agent.update_episode(get_action_mask())
            agent.step_lr(episode)
        
        episode_returns.append(episode_return)
        
        # Evaluate and report to Optuna at intervals
        if episode % eval_interval == 0 or episode == max_episodes:
            eval_results = evaluate_policy(
                agent.policy,
                env,
                action_type,
                n_episodes=eval_episodes,
                deterministic=True,
                device="cpu",
                render=False
            )
            mean_eval_return = eval_results["mean_return"]
            
            # Update best eval return
            if mean_eval_return > best_eval_return:
                best_eval_return = mean_eval_return
            
            # Report to Optuna
            trial.report(mean_eval_return, step=episode)
            
            # Check for pruning
            if trial.should_prune():
                print(f"  PRUNED at episode {episode}, eval_return: {mean_eval_return:.2f}")
                env.close()
                raise TrialPruned()
            
            # Print progress
            print(f"  Episode {episode:4d} | Eval Return: {mean_eval_return:7.2f} | Best so far: {best_eval_return:7.2f}")
    
    env.close()
    return best_eval_return


def run_optuna_search(n_trials: int = 30, seed: int = 0, max_episodes: int = 500, eval_interval: int = 50):
    """Run Optuna hyperparameter search for MountainCarContinuous-v0 with pruning."""
    print("=" * 60)
    print("Optuna Hyperparameter Search: MountainCarContinuous-v0")
    print("=" * 60)
    print(f"Number of trials: {n_trials}")
    print(f"Seed: {seed}")
    print(f"Max episodes per trial: {max_episodes}")
    print(f"Evaluation interval: {eval_interval} episodes")
    print("-" * 60)
    
    def objective(trial):
        return train_with_pruning(trial, seed, max_episodes, eval_interval)
    
    # Create study with MedianPruner
    study = optuna.create_study(
        direction="maximize",
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=100, interval_steps=eval_interval)
    )
    
    # Run optimization
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    # Print results
    print("\n" + "=" * 60)
    print("Optuna Search Complete")
    print("=" * 60)
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best value (best eval return): {study.best_value:.2f}")
    print("\nBest hyperparameters:")
    for key, value in study.best_params.items():
        if key == "hidden_sizes_idx":
            hidden_size_choices = [
                [64, 64], [128, 128], [256, 256], [64, 128], [128, 256]
            ]
            print(f"  {key}: {hidden_size_choices[value]}")
        elif key == "use_lr_decay" and value == False:
            print(f"  {key}: {value}")
            print(f"  lr_decay: None")
        else:
            print(f"  {key}: {value}")
    print("=" * 60)
    
    return study.best_params


def main():
    parser = argparse.ArgumentParser(description="Train Actor-Critic on MountainCarContinuous-v0")
    
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--max_steps", type=int, default=500, help="Max steps per episode")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--lr_policy", type=float, default=3e-4, help="Policy learning rate")
    parser.add_argument("--lr_value", type=float, default=1e-3, help="Value learning rate")
    parser.add_argument("--hidden_sizes", type=str, default="128,128", help="Hidden layer sizes (comma-separated)")
    parser.add_argument("--normalize_advantages", action="store_true", help="Normalize advantages")
    parser.add_argument("--entropy_coef", type=float, default=0.01, help="Entropy bonus coefficient")
    parser.add_argument("--max_episodes", type=int, default=None, help="Maximum training episodes")
    parser.add_argument("--eval_interval", type=int, default=100, help="Evaluation interval (episodes)")
    parser.add_argument("--eval_episodes", type=int, default=10, help="Number of episodes for evaluation")
    parser.add_argument("--update_mode", type=str, default="episode", choices=["step", "episode"],
                       help="Update after each step or after episode")
    parser.add_argument("--lr_decay", type=float, default=None, help="Learning rate decay factor")
    parser.add_argument("--lr_decay_steps", type=int, default=100, help="Number of episodes/steps between LR decay updates")
    parser.add_argument("--artifact_dir", type=str, default="results", help="Artifact output directory")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="Device to use (cpu or cuda)")
    
    # Optuna flags
    parser.add_argument("--optuna", action="store_true", help="Run Optuna hyperparameter search")
    parser.add_argument("--n_trials", type=int, default=30, help="Number of Optuna trials")
    parser.add_argument("--optuna_eval_interval", type=int, default=50, help="Evaluation interval for Optuna pruning (episodes)")
    
    args = parser.parse_args()
    
    if args.optuna:
        # Run Optuna search
        best_params = run_optuna_search(
            n_trials=args.n_trials,
            seed=args.seed,
            max_episodes=args.max_episodes or ENV_CONFIGS[ENV_NAME]['max_episodes'],
            eval_interval=args.optuna_eval_interval
        )
    else:
        # Train with given hyperparameters
        config = {
            'env': ENV_NAME,
            'seed': args.seed,
            'max_steps': args.max_steps,
            'gamma': args.gamma,
            'lr_policy': args.lr_policy,
            'lr_value': args.lr_value,
            'hidden_sizes': parse_hidden_sizes(args.hidden_sizes),
            'normalize_advantages': args.normalize_advantages,
            'entropy_coef': args.entropy_coef,
            'update_mode': args.update_mode,
            'max_episodes': args.max_episodes,
            'eval_interval': args.eval_interval,
            'eval_episodes': args.eval_episodes,
            'lr_decay': args.lr_decay,
            'lr_decay_steps': args.lr_decay_steps,
            'device': args.device,
            'artifact_dir': args.artifact_dir,
            'no_save': False
        }
        train(config)


if __name__ == "__main__":
    main()
