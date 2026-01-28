"""
Training script for Acrobot-v1 with Optuna hyperparameter optimization.
"""

import argparse
import optuna
from optuna.pruners import MedianPruner

from common import (
    train, ENV_CONFIGS, parse_hidden_sizes, set_seed
)


ENV_NAME = 'Acrobot-v1'


def run_optuna_search(n_trials: int = 30, seed: int = 0, max_episodes: int = 500):
    """Run Optuna hyperparameter search for Acrobot-v1."""
    print("=" * 60)
    print("Optuna Hyperparameter Search: Acrobot-v1")
    print("=" * 60)
    print(f"Number of trials: {n_trials}")
    print(f"Seed: {seed}")
    print(f"Max episodes per trial: {max_episodes}")
    print("-" * 60)
    
    def objective(trial):
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
            lr_decay_steps = 100  # Default, won't be used
        
        # Create config
        config = {
            'env': ENV_NAME,
            'seed': seed,
            'max_steps': 500,
            'gamma': gamma,
            'lr_policy': lr_policy,
            'lr_value': lr_value,
            'hidden_sizes': hidden_sizes,
            'normalize_advantages': normalize_advantages,
            'entropy_coef': entropy_coef,
            'update_mode': update_mode,
            'max_episodes': max_episodes,
            'eval_interval': 100,
            'eval_episodes': 10,
            'lr_decay': lr_decay,
            'lr_decay_steps': lr_decay_steps,
            'artifact_dir': None,
            'no_save': True  # Don't save during Optuna search
        }
        
        # Train and get final average return
        results = train(config)
        
        # Return average of last 100 episodes (or all if less than 100)
        final_return = results['final_avg_return']
        
        # Print trial results
        print(f"\nTrial {trial.number}:")
        lr_decay_str = f"lr_decay={lr_decay:.4f}, lr_decay_steps={lr_decay_steps}" if use_lr_decay else "lr_decay=None"
        print(f"  lr_policy={lr_policy:.6f}, lr_value={lr_value:.6f}, gamma={gamma:.4f}, "
              f"entropy_coef={entropy_coef:.4f}, normalize_advantages={normalize_advantages}, "
              f"update_mode={update_mode}, hidden_sizes={hidden_sizes}, {lr_decay_str}")
        print(f"  Final avg return (last 100): {final_return:.2f}")
        print(f"  Converged: {results['converged']}")
        if results['converged']:
            print(f"  Episodes to converge: {results['episodes']}")
            print(f"  Time to converge: {results['time_to_converge_seconds']:.2f} seconds")
        
        return final_return
    
    # Create study
    study = optuna.create_study(
        direction="maximize",  # For Acrobot, higher is better (less negative)
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=10)
    )
    
    # Run optimization
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    
    # Print results
    print("\n" + "=" * 60)
    print("Optuna Search Complete")
    print("=" * 60)
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best value (final avg return): {study.best_value:.2f}")
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
    parser = argparse.ArgumentParser(description="Train Actor-Critic on Acrobot-v1")
    
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
    
    # Optuna flags
    parser.add_argument("--optuna", action="store_true", help="Run Optuna hyperparameter search")
    parser.add_argument("--n_trials", type=int, default=30, help="Number of Optuna trials")
    
    args = parser.parse_args()
    
    if args.optuna:
        # Run Optuna search
        best_params = run_optuna_search(
            n_trials=args.n_trials,
            seed=args.seed,
            max_episodes=args.max_episodes or ENV_CONFIGS[ENV_NAME]['max_episodes']
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
            'artifact_dir': args.artifact_dir,
            'no_save': False
        }
        train(config)


if __name__ == "__main__":
    main()
