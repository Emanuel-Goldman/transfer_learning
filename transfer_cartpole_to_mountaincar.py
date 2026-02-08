"""
Transfer learning: Load CartPole model and train on MountainCar.
Re-initializes output layer weights before training.
"""

import os
import random
import sys
import time

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch import distributions as dist
import matplotlib.pyplot as plt

# Import from mountaincar for consistency
from mountaincar import (
    DEVICE, PADDED_STATE_DIM, POLICY_OUT_DIM,
    pad_state, PolicyNet, ValueNet, select_action, set_seed, shaped_reward_mcc
)

ENV_NAME = 'MountainCarContinuous-v0'
SOURCE_MODEL_PATH = 'models/CartPole-v1_seed0_model.pth'


def reinitialize_output_layer(network):
    """Re-initialize the weights of the output layer."""
    # For PolicyNet, re-initialize the head
    if hasattr(network, 'head'):
        nn.init.xavier_uniform_(network.head.weight)
        if network.head.bias is not None:
            nn.init.zeros_(network.head.bias)
        print(f"  Re-initialized policy output layer: {network.head.in_features} -> {network.head.out_features}")
    # For ValueNet, find the last linear layer
    else:
        for module in reversed(list(network.modules())):
            if isinstance(module, nn.Linear) and module.out_features == 1:
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
                print(f"  Re-initialized value output layer: {module.in_features} -> {module.out_features}")
                break


def train_transfer(episodes=200, max_steps=999, gamma=0.99, lr_p=3e-4, lr_v=1e-3, seed=0,
                   entropy_coef=0.02, window=100, target_reward_threshold=90.0,
                   min_episodes_before_stop=100, use_reward_shaping=True, k_progress=2.0, k_pos=0.5,
                   k_success_bonus=50.0, normalize_advantages=True, lr_decay=None, lr_decay_steps=40,
                   skip_plotting=False, k_pos_decay=0.98, k_pos_decay_steps=10):
    """Train MountainCar using a model pre-trained on CartPole."""
    
    set_seed(seed)
    env = gym.make(ENV_NAME)

    # Load CartPole model
    print(f"\nLoading CartPole model from: {SOURCE_MODEL_PATH}")
    if not os.path.exists(SOURCE_MODEL_PATH):
        raise FileNotFoundError(f"Model file not found: {SOURCE_MODEL_PATH}")
    
    checkpoint = torch.load(SOURCE_MODEL_PATH, map_location=DEVICE)
    
    # Get hidden_sizes from checkpoint or use default [64, 128]
    hidden_sizes = checkpoint.get('hidden_sizes', [64, 128])
    
    # Create MountainCar networks - need to adapt architecture
    # MountainCar uses single hidden layer (128), but we'll use CartPole's architecture
    # We'll create a custom PolicyNet that matches CartPole's structure but with MountainCar's head
    class TransferPolicyNet(nn.Module):
        def __init__(self, hidden_sizes, init_log_std=-0.5):
            super().__init__()
            layers = []
            prev_size = PADDED_STATE_DIM
            for hidden_size in hidden_sizes:
                layers.append(nn.Linear(prev_size, hidden_size))
                layers.append(nn.ReLU())
                prev_size = hidden_size
            self.body = nn.Sequential(*layers)
            self.head = nn.Linear(prev_size, POLICY_OUT_DIM)
            self.log_std_param = nn.Parameter(torch.tensor(float(init_log_std)))

        def forward(self, x):
            h = self.body(x)
            return self.head(h)
    
    class TransferValueNet(nn.Module):
        def __init__(self, hidden_sizes):
            super().__init__()
            layers = []
            prev_size = PADDED_STATE_DIM
            for hidden_size in hidden_sizes:
                layers.append(nn.Linear(prev_size, hidden_size))
                layers.append(nn.ReLU())
                prev_size = hidden_size
            layers.append(nn.Linear(prev_size, 1))
            self.net = nn.Sequential(*layers)

        def forward(self, x):
            return self.net(x).squeeze(-1)
    
    policy = TransferPolicyNet(hidden_sizes=hidden_sizes, init_log_std=-0.5).to(DEVICE)
    value = TransferValueNet(hidden_sizes=hidden_sizes).to(DEVICE)
    
    # Load pre-trained weights (only body layers, skip output layer)
    policy_state = checkpoint['policy_state_dict']
    value_state = checkpoint['value_state_dict']
    
    # Load policy body weights (CartPole uses 'net', we use 'body')
    policy_dict = policy.state_dict()
    for key, value_tensor in policy_state.items():
        if key.startswith('net.'):
            # Extract layer index from key like 'net.0.weight', 'net.1.bias', etc.
            parts = key.split('.')
            if len(parts) >= 2 and parts[1].isdigit():
                layer_idx = int(parts[1])
                # Skip the last linear layer (output layer) - it's at index len(hidden_sizes)*2
                if layer_idx < len(hidden_sizes) * 2:
                    # Map to our body structure
                    new_key = f"body.{layer_idx}.{'.'.join(parts[2:])}"
                    if new_key in policy_dict and policy_dict[new_key].shape == value_tensor.shape:
                        policy_dict[new_key] = value_tensor
    
    # Load value body weights
    value_dict = value.state_dict()
    for key, value_tensor in value_state.items():
        if key.startswith('net.'):
            parts = key.split('.')
            if len(parts) >= 2 and parts[1].isdigit():
                layer_idx = int(parts[1])
                # Skip the last linear layer (output layer)
                if layer_idx < len(hidden_sizes) * 2:
                    new_key = key  # Keep same structure
                    if new_key in value_dict and value_dict[new_key].shape == value_tensor.shape:
                        value_dict[new_key] = value_tensor
    
    policy.load_state_dict(policy_dict, strict=False)
    value.load_state_dict(value_dict, strict=False)
    print("  Loaded pre-trained weights from CartPole (body layers only)")
    
    # Re-initialize output layers
    print("Re-initializing output layers...")
    reinitialize_output_layer(policy)
    reinitialize_output_layer(value)

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
    print(f"\n=== Transfer Learning: CartPole -> {ENV_NAME} ===", flush=True)
    print(f"Target reward threshold: {target_reward_threshold} (avg{window})", flush=True)
    if lr_decay is not None:
        print(f"LR decay: {lr_decay}, steps: {lr_decay_steps}", flush=True)
    if k_pos_decay is not None:
        print(f"Position bonus decay: {k_pos_decay}, steps: {k_pos_decay_steps}", flush=True)
    sys.stdout.flush()
    
    # Track current k_pos for decay
    current_k_pos = k_pos

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
            a, logp, ent = select_action(policy, s, ENV_NAME)
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
                    k_pos=current_k_pos,
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

        # Add final bonus for successful episodes
        if reached_goal and use_reward_shaping:
            performance_bonus = max(0, ep_env_return) * 3.0
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
            stop_reward = ep >= min_episodes_before_stop and avg_env >= target_reward_threshold
            stop_max_episodes = ep >= episodes
            
            if stop_reward or stop_max_episodes:
                if stop_reward:
                    print(f"Early stop: average reward {avg_env:.1f} >= {target_reward_threshold} over last {window} episodes.", flush=True)
                if stop_max_episodes and not stop_reward:
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
        
        # Position bonus decay
        if k_pos_decay is not None:
            if ep % k_pos_decay_steps == 0 and ep > 0:
                current_k_pos *= k_pos_decay
                print(f"  [Position Bonus Decay at episode {ep}] k_pos: {current_k_pos:.4f}", flush=True)

    env.close()

    if not skip_plotting:
        plt.figure(figsize=(14, 6))

        plt.subplot(1, 2, 1)
        plt.plot(env_returns, label="Env return (true)")
        if len(env_returns) >= window:
            rolling_env = [np.mean(env_returns[max(0, i - (window - 1)): i + 1]) for i in range(len(env_returns))]
            plt.plot(rolling_env, label=f"Env Avg-{window}")
        plt.title(f"{ENV_NAME}: Environment Return (Transfer from CartPole)")
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(shaped_returns, label="Shaped return (train)")
        if len(shaped_returns) >= window:
            rolling_shaped = [np.mean(shaped_returns[max(0, i - (window - 1)): i + 1]) for i in range(len(shaped_returns))]
            plt.plot(rolling_shaped, label=f"Shaped Avg-{window}")
        plt.title(f"{ENV_NAME}: Shaped Return (Transfer from CartPole)")
        plt.legend()

        plt.tight_layout()
        
        # Save plot
        os.makedirs("results/plots", exist_ok=True)
        plot_filename = f"results/plots/{ENV_NAME}_transfer_from_cartpole_seed{seed}_training_curve.png"
        plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {plot_filename}")
        
        plt.show()

    # Save model
    os.makedirs("models", exist_ok=True)
    model_filename = f"models/{ENV_NAME}_transfer_from_cartpole_seed{seed}_model.pth"
    torch.save({
        'policy_state_dict': policy.state_dict(),
        'value_state_dict': value.state_dict(),
        'env_name': ENV_NAME,
        'seed': seed,
        'source_model': 'CartPole-v1',
        'hidden_sizes': hidden_sizes,
    }, model_filename)
    print(f"Model saved to: {model_filename}")

    total_time = time.time() - start
    print(f"Finished. Time: {total_time:.1f}s, Episodes: {len(env_returns)}")
    return env_returns, shaped_returns, success_hist


if __name__ == "__main__":
    train_transfer(
        episodes=200,
        max_steps=999,
        gamma=0.99,
        lr_p=3e-4,
        lr_v=1e-3,
        seed=0,
        entropy_coef=0.02,
        window=100,
        target_reward_threshold=90.0,
        min_episodes_before_stop=100,
        use_reward_shaping=True,
        k_progress=2.0,
        k_pos=0.5,
        k_success_bonus=50.0,
        normalize_advantages=True,
        lr_decay=None,
        lr_decay_steps=40,
        skip_plotting=False,
        k_pos_decay=0.98,
        k_pos_decay_steps=10
    )
