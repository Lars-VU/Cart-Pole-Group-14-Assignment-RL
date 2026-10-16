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

SEED = 43


def set_seed(seed=SEED):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False

class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        # Amount of prioritization, 0 is uniform, 1 is full prioritization
        self.alpha = alpha
        self.buffer = []
        self.priorities = np.zeros((capacity,))
        self.position = 0

    def add(self, state, action, reward, next_state, done):
        # New transitions get maximum priority to make sure it gets sampled at least once
        max_p = self.priorities.max() if len(self.buffer) > 0 else 1.0

        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.position] = (state, action, reward, next_state, done)

        self.priorities[self.position] = max_p
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        len_buffer = len(self.buffer)
        if len_buffer == 0:
            return None

        # Calculate probabilities P(i) = p_i^alpha / sum(p_k^alpha)
        priorities = self.priorities[:len_buffer]
        probability = priorities ** self.alpha
        probabilities = probability / np.sum(probability)

        # sample indices based on probabilities
        indices = np.random.choice(len_buffer, batch_size, p=probabilities)
        samples = [self.buffer[idx] for idx in indices]

        # Compute importance sampling weights w_i = (1 / (N * P(i)))^beta
        weights = (len_buffer * probabilities[indices]) ** (-beta)
        # Normalize weights
        weights = weights / weights.max()

        states, actions, rewards, next_states, dones = zip(*samples)

        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones),
            indices,
            np.array(weights)
        )

    def update_priorities(self, indices, td_errors, epsilon_per=1e-5):
        # Update priorities p_i = |delta_i| + epsilon_per
        for idx, error in zip(indices, td_errors):
            self.priorities[idx] = abs(error) + epsilon_per

    def __len__(self):
        return len(self.buffer)

def train_per():
    set_seed()

    env = gym.make('CartPole-v1')
    env.action_space.seed(SEED)

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
    replay = PrioritizedReplayBuffer(capacity=50000, alpha=0.6)

    # Hyperparameters
    epsilon = 1.0
    epsilon_end = 0.01
    epsilon_decay = 0.999
    gamma = 0.99
    batch_size = 64
    num_episodes = 2000
    pole_lengths = np.linspace(0.4, 1.8, 30)

    # Start with lower beta, work up to 1
    beta_start = 0.4
    beta_frames = num_episodes

    # Store rewards for learning curve
    episode_rewards = []

    start_time = time.time()

    for episode in range(num_episodes):
        # Calculate current beta for weight scaling
        beta = min(1.0, beta_start + episode * (1.0 - beta_start) / beta_frames)
        print('beta:', beta)

        # Change pole length randomly
        env.unwrapped.length = np.random.choice(pole_lengths)

        state = env.reset(seed=SEED + episode)[0]
        total_reward = 0
        terminated = False
        truncated = False

        while not (terminated or truncated):
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

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
            replay.add(state, action, reward, next_state, terminated)

            state = next_state
            total_reward += reward

            # Optimization using TD, filling up the buffer a bit first
            if len(replay) >= 10000:
                states, actions, rewards, next_states, dones, indices, weights = replay.sample(batch_size, beta)

                states = torch.tensor(states)
                actions = torch.tensor(actions)
                rewards = torch.tensor(rewards)
                next_states = torch.tensor(next_states)
                dones = torch.tensor(dones, dtype=torch.float32)
                weights = torch.tensor(weights)

                # Get all Q values of the batch
                all_q_values = policy_network(states)

                # Get Q values of actually taken actions
                q_values = all_q_values[range(batch_size), actions]

                # Compute TD Target
                # y_t = r_t + gamma * max_a Q(s_(t+1), a; theta-)
                next_q_values = target_network(next_states).max(1)[0]
                # 1 - dones to not calculate future rewards if terminated, it only needs immediate reward then
                target_q_values = rewards + (gamma * next_q_values * (1 - dones))

                td_errors = (target_q_values - q_values).detach().cpu().numpy()
                replay.update_priorities(indices, td_errors)

                # Calculate Mean Squared Error (MSE) loss L(theta) = 1/B * sum(w_i * (target - Q)^2)
                loss = (weights * nn.MSELoss(reduction='none')(q_values, target_q_values)).mean()

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
    torch.save(policy_network.state_dict(), "weights/PER.pth")
    np.save("weights/PER_rewards.npy", np.array(episode_rewards))
    print('Training finished for PER model')
    print(f"Training took {(time.time() - start_time) / 60:.2f} minutes.")

if __name__ == '__main__':
    train_per()
