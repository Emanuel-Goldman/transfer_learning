"""
Training script for Acrobot-v1 environment.
"""

import argparse
import os
import random
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

PADDED_STATE_DIM = 6          # max from all the games - CartPole = 4, Acrobot = 6, MountainCarCont = 2
POLICY_OUT_DIM = 5            # [3 logits for discrete, games 1-2] + [mean, log_std for continuous, game 3]

ENV_NAME = 'Acrobot-v1'


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
    def __init__(self, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(PADDED_STATE_DIM, hidden),
            nn.ReLU(),
            nn.Linear(hidden, POLICY_OUT_DIM),
        )

    def forward(self, x):
        return self.net(x)


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


def train_actor_critic_optuna(trial, env_name, episodes=1500, max_steps=500, seed=0):
    """Training function for Optuna optimization."""
    # Sample hyperparameters
    lr_p = trial.suggest_float("lr_p", 1e-5, 1e-2, log=True)
    lr_v = trial.suggest_float("lr_v", 1e-4, 1e-2, log=True)
    gamma = trial.suggest_float("gamma", 0.9, 0.999)
    hidden = trial.suggest_categorical("hidden", [64, 128, 256])
    
    set_seed(seed)
    env = gym.make(env_name)

    policy = PolicyNet(hidden=hidden).to(DEVICE)
    value = ValueNet(hidden=hidden).to(DEVICE)

    opt_p = optim.Adam(policy.parameters(), lr=lr_p)
    opt_v = optim.Adam(value.parameters(), lr=lr_v)
    mse = nn.MSELoss()

    rewards = []

    for ep in range(1, episodes + 1):
        s, _ = env.reset()
        s = pad_state(s)

        total_r = 0.0

        for _ in range(max_steps):
            a, logp = select_action(policy, s, env_name)
            ns, r, terminated, truncated, _ = env.step(a)
            done = terminated or truncated

            ns = pad_state(ns)

            s_t = torch.tensor(s, dtype=torch.float32, device=DEVICE)
            ns_t = torch.tensor(ns, dtype=torch.float32, device=DEVICE)

            v_s = value(s_t)
            with torch.no_grad():
                v_ns = value(ns_t)
                target = torch.tensor(r, dtype=torch.float32, device=DEVICE) + (0.0 if done else gamma * v_ns)

            # critic
            v_loss = mse(v_s, target)
            opt_v.zero_grad()
            v_loss.backward()
            opt_v.step()

            # actor
            adv = (target - v_s).detach()
            p_loss = -(logp * adv)

            opt_p.zero_grad()
            p_loss.backward()
            opt_p.step()

            total_r += float(r)

            s = ns
            if done:
                break

        rewards.append(total_r)
        
        # Report intermediate result for pruning
        if len(rewards) >= 100:
            avg100 = np.mean(rewards[-100:])
            trial.report(avg100, ep)
            
            # Check if threshold reached (Acrobot: -100)
            if avg100 >= -100:
                env.close()
                return avg100
            
            # Prune if needed
            if trial.should_prune():
                env.close()
                raise optuna.TrialPruned()

    env.close()
    
    # Return final average of last 100 episodes
    if len(rewards) >= 100:
        return np.mean(rewards[-100:])
    else:
        return np.mean(rewards) if rewards else 0.0


def select_action(policy, state_padded, env_name):
    s = torch.tensor(state_padded, dtype=torch.float32, device=DEVICE).unsqueeze(0)
    out = policy(s).squeeze(0)

    if env_name in ("CartPole-v1", "Acrobot-v1"):
        logits = out[:3].clone()
        if env_name == "CartPole-v1":
            logits[2] = -1e9

        d = dist.Categorical(logits=logits)
        a = d.sample()
        return int(a.item()), d.log_prob(a)

    if env_name == "MountainCarContinuous-v0":
        mean = out[3]
        log_std = out[4].clamp(-5, 2)
        std = log_std.exp()

        d = dist.Normal(mean, std)
        raw = d.rsample()
        a = torch.tanh(raw)  # [-1,1]

        logp = d.log_prob(raw) - torch.log(1 - a.pow(2) + 1e-6)
        logp = logp.sum()

        return np.array([a.detach().cpu().item()], dtype=np.float32), logp

    raise ValueError("env not supported")


def train_actor_critic(env_name, episodes=1500, max_steps=500, gamma=0.99, lr_p=5e-4, lr_v=1e-3, seed=0):
    set_seed(seed)
    env = gym.make(env_name)

    policy = PolicyNet(hidden=128).to(DEVICE)
    value = ValueNet(hidden=128).to(DEVICE)

    opt_p = optim.Adam(policy.parameters(), lr=lr_p)
    opt_v = optim.Adam(value.parameters(), lr=lr_v)
    mse = nn.MSELoss()

    rewards = []
    p_losses = []
    v_losses = []

    start = time.time()
    print(f"\n=== {env_name} ===")

    for ep in range(1, episodes + 1):
        s, _ = env.reset()
        s = pad_state(s)

        total_r = 0.0
        ep_pl = []
        ep_vl = []

        for _ in range(max_steps):
            a, logp = select_action(policy, s, env_name)
            ns, r, terminated, truncated, _ = env.step(a)
            done = terminated or truncated

            ns = pad_state(ns)

            s_t = torch.tensor(s, dtype=torch.float32, device=DEVICE)
            ns_t = torch.tensor(ns, dtype=torch.float32, device=DEVICE)

            v_s = value(s_t)
            with torch.no_grad():
                v_ns = value(ns_t)
                target = torch.tensor(r, dtype=torch.float32, device=DEVICE) + (0.0 if done else gamma * v_ns)

            # critic
            v_loss = mse(v_s, target)
            opt_v.zero_grad()
            v_loss.backward()
            opt_v.step()

            # actor
            adv = (target - v_s).detach()
            p_loss = -(logp * adv)

            opt_p.zero_grad()
            p_loss.backward()
            opt_p.step()

            total_r += float(r)
            ep_pl.append(float(p_loss.detach().cpu()))
            ep_vl.append(float(v_loss.detach().cpu()))

            s = ns
            if done:
                break

        rewards.append(total_r)
        p_losses.append(float(np.mean(ep_pl)) if ep_pl else 0.0)
        v_losses.append(float(np.mean(ep_vl)) if ep_vl else 0.0)

        if len(rewards) >= 100:
            avg100 = np.mean(rewards[-100:])
            elapsed = time.time() - start
            print(f"Episode {ep} | Reward: {total_r:.1f} | Avg100: {avg100:.1f} | Time: {elapsed:.1f}s")
            
            # Check if threshold reached (Acrobot: -100)
            if avg100 >= -100:
                print(f"\n{'='*60}")
                print(f"THRESHOLD REACHED! Avg100: {avg100:.1f} >= -100")
                print(f"Stopping training at episode {ep}")
                print(f"{'='*60}\n")
                break
        else:
            print(f"Episode {ep} | Reward: {total_r:.1f}")

    env.close()

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(rewards, label="Reward")
    if len(rewards) >= 100:
        avg_curve = [np.mean(rewards[max(0, i - 99): i + 1]) for i in range(len(rewards))]
        plt.plot(avg_curve, label="Avg-100")
    plt.title(f"Reward per Episode ({env_name})")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(p_losses, label="Policy Loss")
    plt.plot(v_losses, label="Value Loss")
    plt.title(f"Loss per Episode ({env_name})")
    plt.legend()

    plt.tight_layout()
    
    # Save plot
    os.makedirs("results/plots", exist_ok=True)
    plot_filename = f"results/plots/{env_name}_seed{seed}_training_curve.png"
    plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
    print(f"Plot saved to: {plot_filename}")
    
    plt.show()

    print(f"Finished {env_name}. Time: {time.time() - start:.1f}s, Episodes: {episodes}")
    return rewards


def optuna_search(env_name, n_trials=50, episodes=1500, max_steps=1000, seed=1):
    """Run Optuna hyperparameter search."""
    print(f"\n{'='*60}")
    print(f"Optuna Hyperparameter Search: {env_name}")
    print(f"{'='*60}")
    print(f"Number of trials: {n_trials}")
    print(f"Episodes per trial: {episodes}")
    print(f"Max steps per episode: {max_steps}")
    print(f"Seed: {seed}")
    print(f"{'='*60}\n")
    
    study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda trial: train_actor_critic_optuna(trial, env_name, episodes, max_steps, seed),
        n_trials=n_trials,
        show_progress_bar=True
    )
    
    print(f"\n{'='*60}")
    print("Optuna Search Complete")
    print(f"{'='*60}")
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best value (avg100): {study.best_value:.2f}")
    print("\nBest hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    print(f"{'='*60}\n")
    
    return study.best_params


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Actor-Critic on Acrobot-v1")
    parser.add_argument("--optuna", action="store_true", help="Run Optuna hyperparameter search")
    parser.add_argument("--n_trials", type=int, default=50, help="Number of Optuna trials")
    parser.add_argument("--episodes", type=int, default=1500, help="Number of episodes")
    parser.add_argument("--max_steps", type=int, default=1000, help="Max steps per episode")
    parser.add_argument("--seed", type=int, default=1, help="Random seed")
    parser.add_argument("--lr_p", type=float, default=5e-4, help="Policy learning rate")
    parser.add_argument("--lr_v", type=float, default=1e-3, help="Value learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    
    args = parser.parse_args()
    
    if args.optuna:
        optuna_search(ENV_NAME, args.n_trials, args.episodes, args.max_steps, args.seed)
    else:
        train_actor_critic(ENV_NAME, args.episodes, args.max_steps, args.gamma, args.lr_p, args.lr_v, args.seed)
