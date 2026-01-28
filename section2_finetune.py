"""
Section 2: Fine-tuning / Transfer Learning

Transfer learning for Actor-Critic between environments using trained source model weights.
Only output layers (policy head and value head) are reinitialized; feature extractor is kept.
"""

import argparse
import json
import os
import time
from collections import deque
from pathlib import Path
from typing import Dict, Optional, Any

import numpy as np
import torch
import torch.nn as nn

# Import from Section 1
from section1_actor_critic import (
    ActorCritic, PolicyNetwork, ValueNetwork, ENV_CONFIGS,
    set_seed, pad_observation, get_solved_threshold, parse_hidden_sizes,
    train as train_agent, evaluate_policy, MetricsLogger, plot_metrics, moving_average,
    OBS_DIM, ACTION_DIM
)
import gymnasium as gym


# ============================================================================
# Model Loading and Reinitialization
# ============================================================================

def load_source_model(checkpoint_path: str, obs_dim: int, action_dim: int, 
                     action_type: str, hidden_sizes: list, device: str = "cpu") -> ActorCritic:
    """
    Load a trained source model from checkpoint.
    
    Args:
        checkpoint_path: Path prefix to checkpoint files (without _policy.pt/_value.pt suffix)
        obs_dim: Observation dimension
        action_dim: Action dimension
        action_type: 'discrete' or 'continuous'
        hidden_sizes: Hidden layer sizes
        device: Device to load model on
    
    Returns:
        Loaded ActorCritic model
    """
    # Create model with same architecture
    model = ActorCritic(
        obs_dim=obs_dim,
        action_dim=action_dim,
        action_type=action_type,
        hidden_sizes=hidden_sizes,
        device=device
    )
    
    # Load weights
    policy_path = f"{checkpoint_path}_policy.pt"
    value_path = f"{checkpoint_path}_value.pt"
    
    if not os.path.exists(policy_path):
        raise FileNotFoundError(f"Policy checkpoint not found: {policy_path}")
    if not os.path.exists(value_path):
        raise FileNotFoundError(f"Value checkpoint not found: {value_path}")
    
    model.policy.load_state_dict(torch.load(policy_path, map_location=device))
    model.value.load_state_dict(torch.load(value_path, map_location=device))
    
    print(f"Loaded source model from: {checkpoint_path}")
    return model


def reinit_output_layers(model: ActorCritic) -> None:
    """
    Reinitialize ONLY the output layers (policy head and value head).
    
    The output layers are:
    - Policy: last Linear layer in policy.mlp.net
    - Value: last Linear layer in value.mlp.net
    
    Uses xavier_uniform_ for weights and zeros for bias.
    Does NOT touch feature extractor layers.
    """
    # Reinitialize policy output layer
    policy_mlp = model.policy.mlp.net
    # Find the last Linear layer (before any output activation)
    policy_output_layer = None
    for i in range(len(policy_mlp) - 1, -1, -1):
        if isinstance(policy_mlp[i], nn.Linear):
            policy_output_layer = policy_mlp[i]
            break
    
    if policy_output_layer is None:
        raise ValueError("Could not find policy output Linear layer")
    
    nn.init.xavier_uniform_(policy_output_layer.weight)
    nn.init.zeros_(policy_output_layer.bias)
    
    # Reinitialize value output layer
    value_mlp = model.value.mlp.net
    # Find the last Linear layer
    value_output_layer = None
    for i in range(len(value_mlp) - 1, -1, -1):
        if isinstance(value_mlp[i], nn.Linear):
            value_output_layer = value_mlp[i]
            break
    
    if value_output_layer is None:
        raise ValueError("Could not find value output Linear layer")
    
    nn.init.xavier_uniform_(value_output_layer.weight)
    nn.init.zeros_(value_output_layer.bias)
    
    # For continuous actions, also reinitialize log_std
    if model.action_type == 'continuous':
        nn.init.zeros_(model.policy.log_std)
    
    print("Reinitialized output layers (policy head and value head)")


# ============================================================================
# Fine-tuning Function
# ============================================================================

def finetune_on_target(model: ActorCritic, target_env_name: str, training_args: dict, 
                      output_dir: str, seed: int) -> Dict[str, Any]:
    """
    Fine-tune model on target environment.
    
    Args:
        model: ActorCritic model (with source weights loaded and output layers reinitialized)
        target_env_name: Target environment name
        training_args: Dictionary with training hyperparameters
        output_dir: Output directory for results
        seed: Random seed
    
    Returns:
        Dictionary with convergence statistics
    """
    # Set seed
    set_seed(seed)
    
    # Create target environment
    env = gym.make(target_env_name)
    env.reset(seed=seed)
    
    # Get environment config
    env_config = ENV_CONFIGS[target_env_name]
    target_obs_dim = env_config['obs_dim']
    target_action_dim = env_config['action_dim']
    target_action_type = env_config['action_type']
    
    # Get solved threshold
    solved_threshold = get_solved_threshold(env, target_env_name)
    
    print(f"\nFine-tuning on target environment: {target_env_name}")
    print(f"Target observation dim: {target_obs_dim} (padded to {model.obs_dim})")
    print(f"Target action dim: {target_action_dim} (output dim: {model.action_dim})")
    print(f"Target action type: {target_action_type}")
    print(f"Solved threshold: {solved_threshold}")
    
    # Update model optimizers with new learning rates
    model.policy_optimizer = torch.optim.Adam(model.policy.parameters(), lr=training_args['lr_policy'])
    model.value_optimizer = torch.optim.Adam(model.value.parameters(), lr=training_args['lr_value'])
    
    # Update learning rate schedulers if needed
    if training_args.get('lr_decay') is not None:
        model.lr_decay = training_args['lr_decay']
        model.lr_decay_steps = training_args.get('lr_decay_steps', 100)
        model.policy_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            model.policy_optimizer, gamma=training_args['lr_decay'])
        model.value_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            model.value_optimizer, gamma=training_args['lr_decay'])
    else:
        model.policy_scheduler = None
        model.value_scheduler = None
    
    # Create logger
    logger = MetricsLogger(
        artifact_dir=output_dir,
        run_name=f"finetune_{target_env_name}_{seed}",
        no_save=False
    )
    
    # Get action mask function
    def get_action_mask():
        if target_action_type == 'discrete':
            mask = np.ones(ACTION_DIM, dtype=np.float32)
            mask[target_action_dim:] = 0  # Mask dummy actions
            return mask
        return None
    
    # Training loop (reuse logic from Section 1)
    best_mean_return = float('-inf')
    episodes_to_converge = None
    updates_to_converge = None
    env_steps_to_converge = None
    total_time_seconds_to_converge = None
    
    step_count = 0
    env_steps = 0
    episode_returns = []
    episode_returns_deque = deque(maxlen=100)
    
    start_time = time.perf_counter()
    
    max_episodes = training_args.get('max_episodes', 500)
    max_steps = training_args.get('max_steps', 500)
    update_mode = training_args.get('update_mode', 'episode')
    eval_interval = training_args.get('eval_interval', 100)
    eval_episodes = training_args.get('eval_episodes', 10)
    
    print("\nStarting fine-tuning...")
    print(f"Max episodes: {max_episodes}")
    print(f"Update mode: {update_mode}")
    print(f"Gamma: {training_args['gamma']}, Policy LR: {training_args['lr_policy']}, Value LR: {training_args['lr_value']}")
    print("-" * 50)
    
    for episode in range(1, max_episodes + 1):
        obs, info = env.reset()
        episode_return = 0.0
        episode_length = 0
        done = False
        final_done_bootstrap = False
        
        while not done and episode_length < max_steps:
            action_mask = get_action_mask()
            action, value = model.select_action(obs, action_mask, deterministic=False)
            
            # Get log prob for storage
            obs_padded = pad_observation(obs)
            obs_tensor = torch.FloatTensor(obs_padded).unsqueeze(0).to(model.device)
            if target_action_type == 'discrete':
                action_tensor = torch.LongTensor([action])
            else:
                action_tensor = torch.FloatTensor(torch.tensor(action)).unsqueeze(0)
            action_mask_tensor = None
            if target_action_type == 'discrete' and action_mask is not None:
                action_mask_tensor = torch.FloatTensor(action_mask).unsqueeze(0)
            log_prob = model.policy.get_log_prob(obs_tensor, action_tensor, action_mask_tensor).item()
            
            # Format action for environment
            if target_action_type == 'discrete':
                action_env = int(action)
            else:
                # Continuous: ensure it's a numpy array
                if isinstance(action, np.ndarray):
                    action_env = action if action.ndim > 0 else np.array([action], dtype=np.float32)
                else:
                    action_env = np.array([action], dtype=np.float32)
            
            next_obs, reward, terminated, truncated, info = env.step(action_env)
            done_env = terminated or truncated
            done_bootstrap = terminated
            final_done_bootstrap = done_bootstrap
            
            if not done_env:
                _, next_value = model.select_action(next_obs, action_mask, deterministic=False)
            else:
                next_value = 0.0
            
            if update_mode == "step":
                metrics = model.update_step(
                    obs, action, reward, log_prob, value,
                    next_obs, next_value, done_bootstrap, action_mask
                )
                if metrics:
                    logger.log_scalar("td_error", metrics.get("td_error", 0), step_count)
                    logger.log_scalar("policy_loss", metrics.get("policy_loss", 0), step_count)
                    logger.log_scalar("value_loss", metrics.get("value_loss", 0), step_count)
                step_count += 1
                model.step_lr(step_count)
            else:
                model.store_transition(
                    obs, action, reward, log_prob, value,
                    next_obs=next_obs, next_value=next_value, done_bootstrap=done_bootstrap
                )
            
            obs = next_obs
            episode_return += reward
            episode_length += 1
            done = done_env
        
        # Track environment steps
        env_steps += episode_length
        
        if update_mode == "episode":
            # Add final state and value
            if not final_done_bootstrap:
                _, final_value = model.select_action(obs, get_action_mask(), deterministic=False)
                model.buffer.states.append(obs)
                model.buffer.values.append(final_value)
            else:
                model.buffer.states.append(obs)
                model.buffer.values.append(0.0)
            
            metrics = model.update_episode(get_action_mask())
            if metrics:
                logger.log_scalar("mean_td_error", metrics.get("mean_td_error", 0), episode)
                logger.log_scalar("std_td_error", metrics.get("std_td_error", 0), episode)
                logger.log_scalar("policy_loss", metrics.get("policy_loss", 0), episode)
                logger.log_scalar("value_loss", metrics.get("value_loss", 0), episode)
                if "entropy" in metrics:
                    logger.log_scalar("entropy", metrics["entropy"], episode)
            model.step_lr(episode)
        
        episode_returns.append(episode_return)
        episode_returns_deque.append(episode_return)
        logger.log_scalar("episode_return", episode_return, episode)
        logger.log_scalar("episode_length", episode_length, episode)
        
        # Log learning rates if decay is enabled
        if training_args.get('lr_decay') is not None:
            current_lrs = model.get_current_lrs()
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
                if update_mode == "step":
                    updates_to_converge = step_count
                else:
                    updates_to_converge = episode
                
                # Print convergence summary block
                print("\n" + "=" * 60)
                print("CONVERGENCE ACHIEVED (Fine-tuning)")
                print("=" * 60)
                print(f"Environment: {target_env_name}")
                print(f"Seed: {seed}")
                print(f"Threshold: {solved_threshold}")
                print(f"Episodes to converge: {episodes_to_converge}")
                print(f"Env steps to converge: {env_steps_to_converge}")
                print(f"Updates to converge: {updates_to_converge}")
                print(f"Time to converge (s): {total_time_seconds_to_converge:.2f}")
                print(f"Final avg return (last 100): {ma_last_100:.2f}")
                print("=" * 60 + "\n")
                
                break
        
        if episode % eval_interval == 0 or episode == max_episodes:
            eval_results = evaluate_policy(
                model.policy,
                env,
                target_action_type,
                n_episodes=eval_episodes,
                deterministic=True,
                device=model.device,
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
            
            current_lrs = model.get_current_lrs()
            print(f"Episode {episode:4d} | "
                  f"Return: {episode_return:7.2f} | "
                  f"MA (last 100): {ma_last_100:7.2f} | "
                  f"Eval Return: {mean_return:7.2f} ± {eval_results['std_return']:.2f} | "
                  f"Policy Loss: {metrics.get('policy_loss', 0):.4f} | "
                  f"Value Loss: {metrics.get('value_loss', 0):.4f}")
            
            if update_mode == "episode" and "mean_td_error" in metrics:
                print(f"  TD Error: {metrics['mean_td_error']:.4f} ± {metrics['std_td_error']:.4f}")
            
            if training_args.get('lr_decay') is not None:
                print(f"  LR: policy={current_lrs['lr_policy']:.6f}, value={current_lrs['lr_value']:.6f}")
            
            if mean_return > best_mean_return:
                best_mean_return = mean_return
                model.save(f"{output_dir}/checkpoints/agent_best")
    
    elapsed_time = time.perf_counter() - start_time
    
    # Save final model
    model.save(f"{output_dir}/checkpoints/agent_final")
    
    # Save metrics and plots
    logger.save_metrics()
    plot_metrics(logger.get_metrics(), Path(output_dir) / "plots", 
                 target_env_name, seed, solved_threshold)
    
    # Calculate final average return
    if len(episode_returns_deque) == 100:
        final_ma100_return = sum(episode_returns_deque) / 100
    elif len(episode_returns) >= 100:
        final_ma100_return = sum(episode_returns[-100:]) / 100
    else:
        final_ma100_return = sum(episode_returns) / len(episode_returns) if episode_returns else 0.0
    
    # Prepare stats dictionary
    stats = {
        'env_name': target_env_name,
        'seed': seed,
        'source_env': training_args.get('source_env'),
        'threshold': solved_threshold,
        'converged': episodes_to_converge is not None,
        'episodes_to_converge': episodes_to_converge,
        'updates_to_converge': updates_to_converge,
        'env_steps_to_converge': env_steps_to_converge if env_steps_to_converge else env_steps,
        'total_time_seconds_to_converge': total_time_seconds_to_converge if total_time_seconds_to_converge else elapsed_time,
        'runtime_seconds': elapsed_time,
        'best_eval_return': best_mean_return,
        'final_ma100_return': final_ma100_return,
        'hyperparameters': {
            'lr_policy': training_args['lr_policy'],
            'lr_value': training_args['lr_value'],
            'gamma': training_args['gamma'],
            'entropy_coef': training_args.get('entropy_coef', 0.01),
            'normalize_advantages': training_args.get('normalize_advantages', False),
            'hidden_sizes': training_args.get('hidden_sizes', [128, 128]),
            'update_mode': update_mode,
        }
    }
    
    # Save stats to JSON
    stats_path = Path(output_dir) / "stats.json"
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print("\n" + "=" * 50)
    print("Fine-tuning Summary")
    print("=" * 50)
    print(f"Environment: {target_env_name}")
    print(f"Seed: {seed}")
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
    print(f"Final avg return (last 100): {final_ma100_return:.2f}")
    print(f"Results saved to: {output_dir}/")
    print("=" * 50)
    
    env.close()
    
    return stats


# ============================================================================
# Results Comparison Helper
# ============================================================================

def compare_with_section1(target_env_name: str, section1_stats_path: str, 
                         finetune_stats: Dict[str, Any]) -> None:
    """
    Compare fine-tuning results with Section 1 training from scratch.
    
    Args:
        target_env_name: Target environment name
        section1_stats_path: Path to Section 1 stats.json
        finetune_stats: Fine-tuning statistics dictionary
    """
    if not os.path.exists(section1_stats_path):
        print(f"\nNote: Section 1 stats not found at {section1_stats_path}")
        print("Skipping comparison.")
        return
    
    try:
        with open(section1_stats_path, 'r') as f:
            section1_stats = json.load(f)
        
        print("\n" + "=" * 60)
        print("COMPARISON: From Scratch vs Fine-tuning")
        print("=" * 60)
        print(f"Environment: {target_env_name}")
        print(f"\nFrom Scratch (Section 1):")
        print(f"  Episodes to converge: {section1_stats.get('episodes_to_converge', 'N/A')}")
        print(f"  Time to converge: {section1_stats.get('total_time_seconds_to_converge', section1_stats.get('runtime_seconds', 'N/A')):.2f} seconds")
        print(f"  Converged: {section1_stats.get('converged', False)}")
        
        print(f"\nFine-tuning (Section 2):")
        print(f"  Episodes to converge: {finetune_stats.get('episodes_to_converge', 'N/A')}")
        print(f"  Time to converge: {finetune_stats.get('total_time_seconds_to_converge', finetune_stats.get('runtime_seconds', 'N/A')):.2f} seconds")
        print(f"  Converged: {finetune_stats.get('converged', False)}")
        
        # Calculate speedup if both converged
        if section1_stats.get('converged') and finetune_stats.get('converged'):
            s1_time = section1_stats.get('total_time_seconds_to_converge', section1_stats.get('runtime_seconds', 0))
            ft_time = finetune_stats.get('total_time_seconds_to_converge', finetune_stats.get('runtime_seconds', 0))
            if s1_time > 0:
                speedup = s1_time / ft_time
                print(f"\nSpeedup: {speedup:.2f}x")
        
        print("=" * 60)
    except Exception as e:
        print(f"\nError comparing with Section 1: {e}")


# ============================================================================
# Main
# ============================================================================

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Fine-tune Actor-Critic model on target environment")
    
    parser.add_argument("--source_env", type=str, required=True,
                        choices=['CartPole-v1', 'Acrobot-v1', 'MountainCarContinuous-v0'],
                        help="Source environment name")
    parser.add_argument("--target_env", type=str, required=True,
                        choices=['CartPole-v1', 'Acrobot-v1', 'MountainCarContinuous-v0'],
                        help="Target environment name")
    parser.add_argument("--source_checkpoint", type=str, required=True,
                        help="Path prefix to source checkpoint files (without _policy.pt/_value.pt suffix)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    
    # Training hyperparameters (same as Section 1)
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
    
    # Output
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Output directory (default: results/section2/<source>_to_<target>/seed_<seed>/)")
    parser.add_argument("--compare_with_section1", type=str, default=None,
                       help="Path to Section 1 stats.json for comparison (optional)")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Validate transfer pair
    if args.source_env == args.target_env:
        raise ValueError("Source and target environments must be different")
    
    # Get source environment config
    source_config = ENV_CONFIGS[args.source_env]
    source_obs_dim = source_config['obs_dim']
    source_action_dim = source_config['action_dim']
    source_action_type = source_config['action_type']
    
    # Get target environment config
    target_config = ENV_CONFIGS[args.target_env]
    target_obs_dim = target_config['obs_dim']
    target_action_dim = target_config['action_dim']
    target_action_type = target_config['action_type']
    
    # Parse hidden sizes
    hidden_sizes = parse_hidden_sizes(args.hidden_sizes)
    
    # Set output directory
    if args.output_dir is None:
        output_dir = f"results/section2/{args.source_env}_to_{args.target_env}/seed_{args.seed}"
    else:
        output_dir = args.output_dir
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/checkpoints", exist_ok=True)
    os.makedirs(f"{output_dir}/plots", exist_ok=True)
    
    print("=" * 60)
    print("Section 2: Fine-tuning / Transfer Learning")
    print("=" * 60)
    print(f"Source environment: {args.source_env}")
    print(f"Target environment: {args.target_env}")
    print(f"Source checkpoint: {args.source_checkpoint}")
    print(f"Seed: {args.seed}")
    print(f"Output directory: {output_dir}")
    print("=" * 60)
    
    # Load source model
    print("\nLoading source model...")
    model = load_source_model(
        checkpoint_path=args.source_checkpoint,
        obs_dim=source_obs_dim,
        action_dim=source_action_dim,
        action_type=source_action_type,
        hidden_sizes=hidden_sizes,
        device="cpu"
    )
    
    # Reinitialize output layers
    print("\nReinitializing output layers...")
    reinit_output_layers(model)
    
    # Prepare training arguments
    training_args = {
        'source_env': args.source_env,
        'lr_policy': args.lr_policy,
        'lr_value': args.lr_value,
        'gamma': args.gamma,
        'entropy_coef': args.entropy_coef,
        'normalize_advantages': args.normalize_advantages,
        'hidden_sizes': hidden_sizes,
        'max_episodes': args.max_episodes or target_config['max_episodes'],
        'max_steps': args.max_steps,
        'eval_interval': args.eval_interval,
        'eval_episodes': args.eval_episodes,
        'update_mode': args.update_mode,
        'lr_decay': args.lr_decay,
        'lr_decay_steps': args.lr_decay_steps,
    }
    
    # Fine-tune on target
    finetune_stats = finetune_on_target(
        model=model,
        target_env_name=args.target_env,
        training_args=training_args,
        output_dir=output_dir,
        seed=args.seed
    )
    
    # Compare with Section 1 if requested
    if args.compare_with_section1:
        compare_with_section1(args.target_env, args.compare_with_section1, finetune_stats)
    else:
        # Try to find Section 1 stats automatically
        section1_stats_path = f"results/{args.target_env}/stats.json"
        if os.path.exists(section1_stats_path):
            compare_with_section1(args.target_env, section1_stats_path, finetune_stats)


if __name__ == "__main__":
    main()
