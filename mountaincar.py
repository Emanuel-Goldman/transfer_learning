"""
Training script for MountainCarContinuous-v0 environment.
"""

import argparse
import os
import random
import sys
import time
from typing import List

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch import distributions as dist
import matplotlib.pyplot as plt
import optuna

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PADDED_STATE_DIM = 6
POLICY_OUT_DIM = 5

ENV_NAME = 'MountainCarContinuous-v0'


def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pad_state(s):
    s = np.asarray(s, dtype=np.float32)
    out = np.zeros(PADDED_STATE_DIM, dtype=np.float32)
    out[: len(s)] = s
    return out


class PolicyNet(nn.Module):
    def __init__(self, hidden=128, init_log_std=-0.5):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(PADDED_STATE_DIM, hidden),
            nn.ReLU(),
        )
        self.head = nn.Linear(hidden, POLICY_OUT_DIM)
        self.log_std_param = nn.Parameter(torch.tensor(float(init_log_std)))

    def forward(self, x):
        h = self.body(x)
        return self.head(h)


class ValueNet(nn.Module):
    def __init__(self, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(PADDED_STATE_DIM, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def select_action(policy: PolicyNet, state_padded: np.ndarray, env_name: str):
    s = torch.tensor(state_padded, dtype=torch.float32, device=DEVICE).unsqueeze(0)
    out = policy(s).squeeze(0)

    if env_name in ("CartPole-v1", "Acrobot-v1"):
        logits = out[:3].clone()
        if env_name == "CartPole-v1":
            logits[2] = -1e9

        d = dist.Categorical(logits=logits)
        a = d.sample()
        return int(a.item()), d.log_prob(a), d.entropy()

    if env_name == "MountainCarContinuous-v0":
        mean = out[3]

        log_std = policy.log_std_param.clamp(-2.0, 1.0)
        std = log_std.exp()

        d = dist.Normal(mean, std)
        raw = d.rsample()
        a = torch.tanh(raw)

        logp = d.log_prob(raw) - torch.log(1 - a.pow(2) + 1e-6)
        logp = logp.sum()

        ent = d.entropy().sum()
        return np.array([a.detach().cpu().item()], dtype=np.float32), logp, ent

    raise ValueError("env not supported")


def shaped_reward_mcc(env_reward: float,
                      prev_obs: np.ndarray,
                      next_obs: np.ndarray,
                      reached_goal: bool,
                      k_progress: float = 2.0,
                      k_pos: float = 0.5,
                      k_success_bonus: float = 50.0):
    prev_pos = float(prev_obs[0])
    next_pos = float(next_obs[0])

    progress = next_pos - prev_pos
    pos_term = (next_pos + 1.2) / 1.8

    # Give a larger bonus when goal is reached
    bonus = k_success_bonus if reached_goal else 0.0

    return float(env_reward + k_progress * progress + k_pos * pos_term + bonus)


def train_actor_critic(env_name, episodes=300, max_steps=999, gamma=0.99, lr_p=3e-4, lr_v=1e-3, seed=0,
                       entropy_coef=0.02, window=100, target_success_rate=0.85, target_reward_threshold=90.0,
                       min_episodes_before_stop=100, use_reward_shaping=True, k_progress=2.0, k_pos=0.5,
                       k_success_bonus=50.0, normalize_advantages=True, lr_decay=None, lr_decay_steps=40,
                       skip_plotting=False):
    set_seed(seed)
    env = gym.make(env_name)

    policy = PolicyNet(hidden=128, init_log_std=-0.5).to(DEVICE)
    value = ValueNet(hidden=128).to(DEVICE)

    opt_p = optim.Adam(policy.parameters(), lr=lr_p)
    opt_v = optim.Adam(value.parameters(), lr=lr_v)
    mse = nn.MSELoss()
    
    # Learning rate schedulers
    if lr_decay is not None:
        scheduler_p = optim.lr_scheduler.ExponentialLR(opt_p, gamma=lr_decay)
        scheduler_v = optim.lr_scheduler.ExponentialLR(opt_v, gamma=lr_decay)
    else:
        scheduler_p = None
        scheduler_v = None

    # Track both returns
    shaped_returns = []
    env_returns = []
    success_hist = []

    p_losses = []
    v_losses = []

    start = time.time()
    print(f"\n=== {env_name} ===", flush=True)
    print(f"Target success rate: {target_success_rate}", flush=True)
    print(f"Target reward threshold: {target_reward_threshold}", flush=True)
    if lr_decay is not None:
        print(f"LR decay: {lr_decay}, steps: {lr_decay_steps}", flush=True)
    sys.stdout.flush()

    for ep in range(1, episodes + 1):
        obs, _ = env.reset(seed=seed + ep)
        s = pad_state(obs)

        ep_env_return = 0.0
        ep_shaped_return = 0.0

        reached_goal = False
        prev_obs = np.asarray(obs, dtype=np.float32)

        step_advs = []

        ep_pl = []
        ep_vl = []

        for _ in range(max_steps):
            a, logp, ent = select_action(policy, s, env_name)
            next_obs, env_r, terminated, truncated, _ = env.step(a)
            done = terminated or truncated

            next_obs_np = np.asarray(next_obs, dtype=np.float32)
            next_s = pad_state(next_obs_np)

            if float(next_obs_np[0]) >= 0.45:
                reached_goal = True

            r_env = float(env_r)

            r_train = r_env
            if use_reward_shaping:
                r_train = shaped_reward_mcc(
                    env_reward=r_env,
                    prev_obs=prev_obs,
                    next_obs=next_obs_np,
                    reached_goal=reached_goal,
                    k_progress=k_progress,
                    k_pos=k_pos,
                    k_success_bonus=k_success_bonus,
                )

            s_t = torch.tensor(s, dtype=torch.float32, device=DEVICE)
            ns_t = torch.tensor(next_s, dtype=torch.float32, device=DEVICE)

            v_s = value(s_t)
            with torch.no_grad():
                v_ns = value(ns_t)
                target = torch.tensor(r_train, dtype=torch.float32, device=DEVICE) + (0.0 if done else gamma * v_ns)

            v_loss = mse(v_s, target)
            opt_v.zero_grad()
            v_loss.backward()
            nn.utils.clip_grad_norm_(value.parameters(), 1.0)
            opt_v.step()

            adv = (target - v_s).detach()
            step_advs.append(adv)

            p_loss = -(logp * adv + entropy_coef * ent)

            opt_p.zero_grad()
            p_loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            opt_p.step()

            ep_env_return += r_env
            ep_shaped_return += r_train

            ep_pl.append(float(p_loss.detach().cpu()))
            ep_vl.append(float(v_loss.detach().cpu()))

            s = next_s
            prev_obs = next_obs_np

            if done:
                break

        # Note: normalize_advantages is computed but not used in step-based updates
        # (it would be used in episode-based updates, but we use step-based)
        
        # Add final bonus for successful episodes to ensure high shaped reward correlates with high env reward
        if reached_goal and use_reward_shaping:
            # Scale bonus with how well the agent performed (higher env return = higher bonus)
            # This ensures successful episodes (high EnvRet) get high ShapedRet
            performance_bonus = max(0, ep_env_return) * 3.0  # Scale with positive env return
            ep_shaped_return += performance_bonus
        
        shaped_returns.append(ep_shaped_return)
        env_returns.append(ep_env_return)
        success_hist.append(1 if reached_goal else 0)

        p_losses.append(float(np.mean(ep_pl)) if ep_pl else 0.0)
        v_losses.append(float(np.mean(ep_vl)) if ep_vl else 0.0)

        if ep >= window:
            avg_env = float(np.mean(env_returns[-window:]))
            avg_shaped = float(np.mean(shaped_returns[-window:]))
            succ_rate = float(np.mean(success_hist[-window:]))
            elapsed = time.time() - start

            print(
                f"Ep {ep} | "
                f"EnvRet: {ep_env_return:.1f} (Avg{window}: {avg_env:.1f}) | "
                f"ShapedRet: {ep_shaped_return:.1f} (Avg{window}: {avg_shaped:.1f}) | "
                f"Success{window}: {succ_rate:.2f} | "
                f"Time: {elapsed:.1f}s",
                flush=True
            )

            # Check stopping conditions
            stop_success = ep >= min_episodes_before_stop and succ_rate >= target_success_rate
            stop_reward = ep >= min_episodes_before_stop and avg_env >= target_reward_threshold
            stop_max_episodes = ep >= episodes  # Stop at max episodes
            
            if stop_success or stop_reward or stop_max_episodes:
                if stop_success:
                    print(f"Early stop: success-rate {succ_rate:.2f} >= {target_success_rate} over last {window} episodes.", flush=True)
                if stop_reward:
                    print(f"Early stop: average reward {avg_env:.1f} >= {target_reward_threshold} over last {window} episodes.", flush=True)
                if stop_max_episodes and not stop_success and not stop_reward:
                    print(f"Reached maximum episodes: {episodes}", flush=True)
                break
        else:
            print(f"Ep {ep} | EnvRet: {ep_env_return:.1f} | ShapedRet: {ep_shaped_return:.1f}", flush=True)
            
        # Learning rate decay
        if lr_decay is not None and scheduler_p is not None and scheduler_v is not None:
            if ep % lr_decay_steps == 0 and ep > 0:
                scheduler_p.step()
                scheduler_v.step()
                current_lr_p = opt_p.param_groups[0]['lr']
                current_lr_v = opt_v.param_groups[0]['lr']
                print(f"  [LR Decay at episode {ep}] Policy LR: {current_lr_p:.6f}, Value LR: {current_lr_v:.6f}", flush=True)

    env.close()

    if not skip_plotting:
        plt.figure(figsize=(14, 6))

        plt.subplot(1, 2, 1)
        plt.plot(env_returns, label="Env return (true)")
        if len(env_returns) >= window:
            rolling_env = [np.mean(env_returns[max(0, i - (window - 1)): i + 1]) for i in range(len(env_returns))]
            plt.plot(rolling_env, label=f"Env Avg-{window}")
        plt.title("MountainCarContinuous-v0: Environment Return")
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(shaped_returns, label="Shaped return (train)")
        if len(shaped_returns) >= window:
            rolling_shaped = [np.mean(shaped_returns[max(0, i - (window - 1)): i + 1]) for i in range(len(shaped_returns))]
            plt.plot(rolling_shaped, label=f"Shaped Avg-{window}")
        plt.title("MountainCarContinuous-v0: Shaped Return")
        plt.legend()

        plt.tight_layout()
        
        # Save plot
        os.makedirs("results/plots", exist_ok=True)
        plot_filename = f"results/plots/{env_name}_seed{seed}_training_curve.png"
        plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {plot_filename}")
        
        plt.show()

    total_time = time.time() - start
    print(f"Finished. Time: {total_time:.1f}s, Episodes: {len(env_returns)}")
    return env_returns, shaped_returns, success_hist


def train_actor_critic_optuna(trial, env_name, episodes=200, max_steps=500, seed=0):
    """Training function for Optuna optimization."""
    # Sample hyperparameters
    lr_p = trial.suggest_float("lr_p", 1e-5, 1e-2, log=True)
    lr_v = trial.suggest_float("lr_v", 1e-4, 1e-2, log=True)
    gamma = trial.suggest_float("gamma", 0.9, 0.999)
    entropy_coef = trial.suggest_float("entropy_coef", 0.01, 0.1)
    k_progress = trial.suggest_float("k_progress", 1.0, 5.0)
    k_pos = trial.suggest_float("k_pos", 0.1, 1.0)
    k_success_bonus = trial.suggest_float("k_success_bonus", 10.0, 50.0)
    
    # Disable early stopping and plotting for Optuna (set thresholds very high)
    env_returns, shaped_returns, success_hist = train_actor_critic(
        env_name, episodes, max_steps, gamma, lr_p, lr_v, seed,
        entropy_coef=entropy_coef, use_reward_shaping=True,
        k_progress=k_progress, k_pos=k_pos, k_success_bonus=k_success_bonus,
        normalize_advantages=True,
        target_success_rate=1.0,  # Disable early stopping
        target_reward_threshold=999.0,  # Disable early stopping
        skip_plotting=True  # Skip plotting to save time
    )
    
    # Return success rate as objective (use last 50 episodes if available)
    if len(success_hist) >= 50:
        return np.mean(success_hist[-50:])
    else:
        return np.mean(success_hist) if success_hist else 0.0


def optuna_search(env_name, n_trials=50, episodes=200, max_steps=500, seed=0):
    """Run Optuna hyperparameter search."""
    print(f"\n{'='*60}")
    print(f"Optuna Hyperparameter Search: {env_name}")
    print(f"{'='*60}")
    print(f"Number of trials: {n_trials}")
    print(f"Episodes per trial: {episodes}")
    print(f"Max steps per episode: {max_steps}")
    print(f"Seed: {seed}")
    print(f"{'='*60}\n")
    
    # Use MedianPruner with aggressive early stopping
    pruner = optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=30)
    study = optuna.create_study(direction="maximize", pruner=pruner)
    study.optimize(
        lambda trial: train_actor_critic_optuna(trial, env_name, episodes, max_steps, seed),
        n_trials=n_trials,
        show_progress_bar=True
    )
    
    print(f"\n{'='*60}")
    print("Optuna Search Complete")
    print(f"{'='*60}")
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best value (success rate): {study.best_value:.2f}")
    print("\nBest hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    print(f"{'='*60}\n")
    
    return study.best_params


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Actor-Critic on MountainCarContinuous-v0")
    parser.add_argument("--optuna", action="store_true", help="Run Optuna hyperparameter search")
    parser.add_argument("--n_trials", type=int, default=50, help="Number of Optuna trials")
    parser.add_argument("--episodes", type=int, default=300, help="Number of episodes (300 default, 200 for Optuna)")
    parser.add_argument("--max_steps", type=int, default=999, help="Max steps per episode (500 for Optuna, 999 for regular training)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--lr_p", type=float, default=3e-4, help="Policy learning rate")
    parser.add_argument("--lr_v", type=float, default=1e-3, help="Value learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--entropy_coef", type=float, default=0.02, help="Entropy coefficient")
    parser.add_argument("--k_progress", type=float, default=2.0, help="Progress reward coefficient")
    parser.add_argument("--k_pos", type=float, default=0.5, help="Position reward coefficient")
    parser.add_argument("--k_success_bonus", type=float, default=50.0, help="Success bonus coefficient (reward when goal is reached)")
    parser.add_argument("--target_success_rate", type=float, default=0.85, help="Target success rate for early stopping (0.0-1.0)")
    parser.add_argument("--target_reward_threshold", type=float, default=90.0, help="Target average reward threshold for early stopping")
    parser.add_argument("--min_episodes_before_stop", type=int, default=100, help="Minimum episodes before early stopping")
    parser.add_argument("--no_reward_shaping", action="store_true", help="Disable reward shaping")
    parser.add_argument("--no_normalize_advantages", action="store_true", help="Disable advantage normalization")
    parser.add_argument("--lr_decay", type=float, default=None, help="Learning rate decay factor (e.g., 0.94)")
    parser.add_argument("--lr_decay_steps", type=int, default=40, help="Number of episodes between LR decay updates")
    
    args = parser.parse_args()
    
    if args.optuna:
        optuna_search(ENV_NAME, args.n_trials, args.episodes, args.max_steps, args.seed)
    else:
        train_actor_critic(
            ENV_NAME, args.episodes, args.max_steps, args.gamma, args.lr_p, args.lr_v, args.seed,
            entropy_coef=args.entropy_coef,
            target_success_rate=args.target_success_rate,
            target_reward_threshold=args.target_reward_threshold,
            min_episodes_before_stop=args.min_episodes_before_stop,
            use_reward_shaping=not args.no_reward_shaping,
            k_progress=args.k_progress,
            k_pos=args.k_pos,
            k_success_bonus=args.k_success_bonus,
            normalize_advantages=not args.no_normalize_advantages,
            lr_decay=args.lr_decay,
            lr_decay_steps=args.lr_decay_steps,
        )
