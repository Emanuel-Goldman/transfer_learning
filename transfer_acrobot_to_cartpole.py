"""
Transfer learning: Load Acrobot model and train on CartPole.
Re-initializes output layer weights before training.
"""

import os
import random
import time

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch import distributions as dist
import matplotlib.pyplot as plt

# Import from cartpole for consistency
from cartpole import (
    DEVICE, PADDED_STATE_DIM, POLICY_OUT_DIM,
    pad_state, PolicyNet, ValueNet, select_action, set_seed
)

ENV_NAME = 'CartPole-v1'
SOURCE_MODEL_PATH = 'models/Acrobot-v1_seed1_model.pth'


def reinitialize_output_layer(network):
    """Re-initialize the weights of the output layer."""
    # Find the last linear layer (output layer)
    for module in reversed(list(network.modules())):
        if isinstance(module, nn.Linear) and module.out_features in [POLICY_OUT_DIM, 1]:
            # Re-initialize weights and bias
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
            print(f"  Re-initialized output layer: {module.in_features} -> {module.out_features}")
            break


def train_transfer(episodes=1000, max_steps=500, gamma=0.9838, lr_p=0.001474, lr_v=0.008751, seed=0,
                   entropy_coef=0.01, lr_decay=0.94, lr_decay_steps=40,
                   normalize_advantages=True, update_mode="episode"):
    """Train CartPole using a model pre-trained on Acrobot."""
    
    set_seed(seed)
    env = gym.make(ENV_NAME)

    # Load Acrobot model
    print(f"\nLoading Acrobot model from: {SOURCE_MODEL_PATH}")
    if not os.path.exists(SOURCE_MODEL_PATH):
        raise FileNotFoundError(f"Model file not found: {SOURCE_MODEL_PATH}")
    
    checkpoint = torch.load(SOURCE_MODEL_PATH, map_location=DEVICE)
    
    # Create networks with same architecture (assuming [64, 128] from Acrobot)
    policy = PolicyNet(hidden_sizes=[64, 128]).to(DEVICE)
    value = ValueNet(hidden_sizes=[64, 128]).to(DEVICE)
    
    # Load pre-trained weights
    policy.load_state_dict(checkpoint['policy_state_dict'])
    value.load_state_dict(checkpoint['value_state_dict'])
    print("  Loaded pre-trained weights from Acrobot")
    
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

    rewards = []
    p_losses = []
    v_losses = []

    start = time.time()
    print(f"\n=== Transfer Learning: Acrobot -> {ENV_NAME} ===")
    print(f"Hidden sizes: [64, 128]")
    print(f"Update mode: {update_mode}")
    print(f"Entropy coef: {entropy_coef}")
    print(f"Normalize advantages: {normalize_advantages}")
    if lr_decay is not None:
        print(f"LR decay: {lr_decay}, steps: {lr_decay_steps}")

    for ep in range(1, episodes + 1):
        s, _ = env.reset()
        s = pad_state(s)

        total_r = 0.0
        ep_pl = []
        ep_vl = []
        
        # Episode buffer for episode-based updates
        ep_states = []
        ep_actions = []
        ep_log_probs = []
        ep_entropies = []
        ep_rewards = []
        ep_values = []
        ep_dones = []
        episode_done = False

        for _ in range(max_steps):
            a, logp, ent = select_action(policy, s, ENV_NAME)
            ns, r, terminated, truncated, _ = env.step(a)
            done = terminated or truncated

            ns = pad_state(ns)

            s_t = torch.tensor(s, dtype=torch.float32, device=DEVICE)
            ns_t = torch.tensor(ns, dtype=torch.float32, device=DEVICE)

            v_s = value(s_t)
            with torch.no_grad():
                v_ns = value(ns_t) if not done else torch.tensor(0.0, device=DEVICE)
                target = torch.tensor(r, dtype=torch.float32, device=DEVICE) + (0.0 if done else gamma * v_ns)

            if update_mode == "step":
                # Step-based update
                v_loss = mse(v_s, target)
                opt_v.zero_grad()
                v_loss.backward()
                opt_v.step()

                adv = (target - v_s).detach()
                p_loss = -(logp * adv + entropy_coef * ent)

                opt_p.zero_grad()
                p_loss.backward()
                opt_p.step()
                
                ep_pl.append(float(p_loss.detach().cpu()))
                ep_vl.append(float(v_loss.detach().cpu()))
            else:
                # Episode-based: store transitions
                ep_states.append(s_t)
                ep_actions.append(torch.tensor(a, dtype=torch.long, device=DEVICE))
                ep_log_probs.append(logp)
                ep_entropies.append(ent)
                ep_rewards.append(torch.tensor(r, dtype=torch.float32, device=DEVICE))
                ep_values.append(v_s)
                ep_dones.append(torch.tensor(done, dtype=torch.bool, device=DEVICE))

            total_r += float(r)

            s = ns
            if done:
                episode_done = True
                break

        # Episode-based update
        if update_mode == "episode" and len(ep_states) > 0:
            # Recompute all values at end of episode for consistency
            states = torch.stack(ep_states)
            actions = torch.stack(ep_actions)
            log_probs = torch.stack(ep_log_probs)
            entropies = torch.stack(ep_entropies)
            rewards_t = torch.stack(ep_rewards)
            dones = torch.stack(ep_dones)
            
            # Recompute current state values with current value network
            values = value(states)
            
            # Compute next state values
            with torch.no_grad():
                final_s = torch.tensor(s, dtype=torch.float32, device=DEVICE)
                all_states = torch.cat([states, final_s.unsqueeze(0)])
                all_values = value(all_states)
                
                next_values = all_values[1:]
                next_values = next_values * (1 - dones.float())
            
            targets = rewards_t + gamma * next_values
            advantages = (targets - values).detach()
            
            if normalize_advantages:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            # Value loss
            v_loss = mse(values, targets)
            opt_v.zero_grad()
            v_loss.backward()
            opt_v.step()
            
            # Policy loss with entropy
            p_loss = -(log_probs * advantages + entropy_coef * entropies).mean()
            opt_p.zero_grad()
            p_loss.backward()
            opt_p.step()
            
            ep_pl.append(float(p_loss.detach().cpu()))
            ep_vl.append(float(v_loss.detach().cpu()))

        # Learning rate decay
        if lr_decay is not None and scheduler_p is not None and scheduler_v is not None:
            if ep % lr_decay_steps == 0 and ep > 0:
                scheduler_p.step()
                scheduler_v.step()
                current_lr_p = opt_p.param_groups[0]['lr']
                current_lr_v = opt_v.param_groups[0]['lr']
                print(f"  [LR Decay at episode {ep}] Policy LR: {current_lr_p:.6f}, Value LR: {current_lr_v:.6f}")

        rewards.append(total_r)
        p_losses.append(float(np.mean(ep_pl)) if ep_pl else 0.0)
        v_losses.append(float(np.mean(ep_vl)) if ep_vl else 0.0)

        if len(rewards) >= 100:
            avg100 = np.mean(rewards[-100:])
            elapsed = time.time() - start
            print(f"Episode {ep} | Reward: {total_r:.1f} | Avg100: {avg100:.1f} | Time: {elapsed:.1f}s")
            
            # Check if threshold reached (CartPole: 475)
            if avg100 >= 475:
                print(f"\n{'='*60}")
                print(f"THRESHOLD REACHED! Avg100: {avg100:.1f} >= 475")
                print(f"Stopping training at episode {ep}")
                print(f"{'='*60}\n")
                break
        else:
            print(f"Episode {ep} | Reward: {total_r:.1f}")

    env.close()

    # Plotting
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(rewards, label="Reward")
    if len(rewards) >= 100:
        avg_curve = [np.mean(rewards[max(0, i - 99): i + 1]) for i in range(len(rewards))]
        plt.plot(avg_curve, label="Avg-100")
    plt.title(f"Reward per Episode ({ENV_NAME} - Transfer from Acrobot)")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(p_losses, label="Policy Loss")
    plt.plot(v_losses, label="Value Loss")
    plt.title(f"Loss per Episode ({ENV_NAME} - Transfer from Acrobot)")
    plt.legend()

    plt.tight_layout()
    
    # Save plot
    os.makedirs("results/plots", exist_ok=True)
    plot_filename = f"results/plots/{ENV_NAME}_transfer_from_acrobot_seed{seed}_training_curve.png"
    plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
    print(f"Plot saved to: {plot_filename}")
    
    plt.show()

    # Save model
    os.makedirs("models", exist_ok=True)
    model_filename = f"models/{ENV_NAME}_transfer_from_acrobot_seed{seed}_model.pth"
    torch.save({
        'policy_state_dict': policy.state_dict(),
        'value_state_dict': value.state_dict(),
        'env_name': ENV_NAME,
        'seed': seed,
        'source_model': 'Acrobot-v1',
        'hidden_sizes': [64, 128],
    }, model_filename)
    print(f"Model saved to: {model_filename}")

    print(f"Finished {ENV_NAME} (transfer from Acrobot). Time: {time.time() - start:.1f}s, Episodes: {episodes}")
    return rewards


if __name__ == "__main__":
    train_transfer(
        episodes=1000,
        max_steps=500,
        gamma=0.9838,
        lr_p=0.001474,
        lr_v=0.008751,
        seed=0,
        entropy_coef=0.01,
        lr_decay=0.94,
        lr_decay_steps=40,
        normalize_advantages=True,
        update_mode="episode"
    )
