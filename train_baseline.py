import os

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import gymnasium as gym
from collections import deque
import time

from test_script import QNetwork

# Set random seeds for complete reproducibility across runs
def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        # Randomly sample a batch of transitions from history
        state, action, reward, next_state, done = zip(
            *random.sample(self.buffer, batch_size)
        )
        return (
            np.array(state),
            np.array(action),
            np.array(reward),
            np.array(next_state),
            np.array(done)
        )

    def __len__(self):
        return len(self.buffer)

def train_baseline():
    set_seed(42)

    env = gym.make('CartPole-v1')

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # Initialize Policy network and Target network
    policy_network = QNetwork(state_dim, action_dim)
    target_network = QNetwork(state_dim, action_dim)
    target_network.load_state_dict(policy_network.state_dict())
    target_network.eval()

    # Initialize the Adam optimizer, with learning rate 1e-3
    optimizer = optim.Adam(policy_network.parameters(), lr=0.001)

    # Initialize replay buffer object with 50000 capacity
    replay = ReplayBuffer(capacity=50000)

    # Hyperparameters
    epsilon = 1.0
    epsilon_end = 0.01
    epsilon_decay = 0.999
    gamma = 0.99
    batch_size = 64
    num_episodes = 3000
    pole_lengths = np.linspace(0.4, 1.8, 30)

    # Store rewards for learning curve
    episode_rewards = []

    start_time = time.time()

    for episode in range(num_episodes):
        # Change pole length randomly
        env.unwrapped.length = np.random.choice(pole_lengths)

        state = env.reset()[0]
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        total_reward = 0
        terminated = False
        truncated = False

        while not (terminated or truncated):
            # Epsilon-greedy exploration
            if random.random() < epsilon:
                # Exploration
                action = env.action_space.sample()
            else:
                # Exploitation
                # a = argmax_a(Q(s, a))
                q_values = policy_network(state_tensor)
                action = q_values.argmax().item()

            # Retrieve next state/information
            next_state, reward, terminated, truncated, _ = env.step(action)

            # Store transition in replay buffer
            replay.add(state, action, reward, next_state, (terminated or truncated))

            state = next_state
            total_reward += reward

            # Optimization using TD, filling up the buffer a bit first
            if len(replay) >= 10000:
                states, actions, rewards, next_states, dones = replay.sample(batch_size)

                states = torch.tensor(states)
                actions = torch.tensor(actions)
                rewards = torch.tensor(rewards)
                next_states = torch.tensor(next_states)
                dones = torch.tensor(dones, dtype=torch.float32)

                # Get all Q values of the batch
                all_q_values = policy_network(states)

                # Get Q values of actually taken actions
                q_values = all_q_values[range(batch_size), actions]

                # Compute TD Target
                # y_t = r_t + gamma * max_a Q(s_(t+1), a; theta-)
                next_q_values = target_network(next_states).max(1)[0]
                # 1 - dones to not calculate future rewards if terminated, it only needs immediate reward then
                target_q_values = rewards + (gamma * next_q_values * (1 - dones))

                # Calculate Mean Squared Error (MSE) loss
                loss = nn.MSELoss()(q_values, target_q_values)

                # Stochastic Gradient Descent (SGD)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # Decay the exploration rate
        epsilon = max(epsilon * epsilon_decay, epsilon_end)

        # Update target network weights every 10 episodes, they can't stay the starting weight forever
        if episode % 10 == 0:
            target_network.load_state_dict(policy_network.state_dict())

        episode_rewards.append(total_reward)

        print(
            f'Episode {episode}, Pole Length: {env.unwrapped.length}, Total Reward: {total_reward}, Epsilon: {epsilon}'
        )

    # Save models and data
    os.makedirs("weights", exist_ok=True)
    torch.save(policy_network.state_dict(), "weights/baseline.pth")
    np.save("weights/baseline_rewards.npy", np.array(episode_rewards))
    print('Training finished for Baseline model')
    print(f"Training took {(time.time() - start_time) / 60:.2f} minutes.")

if __name__ == '__main__':
    train_baseline()
