import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from scipy import stats
import gymnasium as gym
from collections import deque
import os
import pandas as pd
import matplotlib.pyplot as plt

SEED = 43


def set_seed(seed=SEED):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False


# This is the 'standard' neural network
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


def plot_training_curves():
    """
    Automatically loads the training reward files and plots a comparative learning curve.
    """
    models = {
        "Baseline": "weights/baseline_rewards.npy",
        "Noise Injection": "weights/noise_injection_rewards.npy",
        "PER": "weights/PER_rewards.npy"
    }

    plt.figure(figsize=(10, 5))
    window = 50  # Moving average window for smoothing out noise

    for name, path in models.items():
        if os.path.exists(path):
            rewards = np.load(path)
            # Calculate moving average for a clean trend line
            if len(rewards) >= window:
                filtered_rewards = np.convolve(rewards, np.ones(window)/window, mode='valid')
                plt.plot(filtered_rewards, label=f"{name} (Mov. Avg)")
            else:
                plt.plot(rewards, label=name, alpha=0.3)

    plt.xlabel('Episodes')
    plt.ylabel('Total Reward (Smoothed)')
    plt.title('Training Performance Comparison Across Strategies')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("weights/training_curves_comparison.png")
    plt.show()
    print("Saved comparative training curve to weights/training_curves_comparison.png")

def bar_plot(results):
    data = results[0]

    avgs = {k: v for k, v in data.items() if k.startswith('Avg_')}
    stds = {k.replace('Avg', 'Std'): data[k.replace('Avg', 'Std')] for k in avgs.keys()}

    sorted_keys = sorted(avgs.keys(), key=lambda x: float(x.split('_')[1]))
    avg_values = [avgs[k] for k in sorted_keys]
    std_values = [stds[k.replace('Avg', 'Std')] for k in sorted_keys]

    overall_avg = np.mean(np.array(avg_values))

    plt.figure(figsize=(8, 5))
    plt.bar(range(len(avg_values)), avg_values, yerr=std_values, capsize=5, alpha=0.7)
    plt.xticks(range(len(avg_values)), [k.split('_')[1] for k in sorted_keys])
    plt.xticks(rotation = 45)
    plt.xlabel('Pole length')
    plt.ylabel('Episode length')
    plt.title(f'Average score over all pole lengths = {round(overall_avg, 0)}')

    plt.savefig("bar_plot.png")
    plt.show()


# Test the agent after training
def test_pole_length(env, q_network):
    """
    This function runs your trained network on a specific pole length
    You are not allowed to change this function
    """

    wind = 25
    state = env.reset()[0]
    state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    done = False
    total_reward = 0


    while not done:

        action = q_network(state).argmax().item()
        next_state, reward, done, _, __ = env.step(action)
        next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
        state = next_state
        total_reward += reward

        if total_reward >= 500 and total_reward <= 1000:
            if total_reward % wind == 0:

                env.unwrapped.force_mag = 75

        if total_reward > 1000:
            env.unwrapped.force_mag = 25 + (0.01 * total_reward)

    return total_reward


def compute_significance(scores_baseline, scores_variant):
    t_stat, p_val = stats.ttest_ind(scores_baseline, scores_variant, equal_var=False)
    return t_stat, p_val


def evaluate_all_models():
    set_seed()
    pole_lengths = np.linspace(0.4, 1.8, 30)

    # List of models found in your weights folder
    model_files = {
        "Baseline": "baseline.pth",
        "Noise Injection": "noise_injection.pth",
        "PER": "PER.pth"
    }

    state_dim = 4
    action_dim = 2

    # Dictionary to store the 300 raw scores for each model
    raw_scores_by_model = {}

    for model_name, filename in model_files.items():
        weight_path = os.path.join("weights", filename)
        if not os.path.exists(weight_path):
            print(f"Skipping {model_name}: {weight_path} not found.")
            continue

        print(f"\n--- Evaluating Model: {model_name} ({filename}) ---")

        # Load model once
        loaded_model = QNetwork(state_dim, action_dim)
        loaded_model.load_state_dict(torch.load(weight_path, weights_only=True))
        loaded_model.eval()

        results = {}
        all_pole_scores = []
        total_score = 0

        for length_index, length in enumerate(pole_lengths):
            pole_score = []
            for x in range(10):
                env = gym.make('CartPole-v1')
                evaluation_seed = SEED + (length_index * 10) + x
                env.action_space.seed(evaluation_seed)
                env.reset(seed=evaluation_seed)
                env.unwrapped.length = length
                score = test_pole_length(env, loaded_model)
                pole_score.append(score)

            mean_score = np.mean(pole_score)
            std_score = np.std(pole_score)
            min_score = np.min(pole_score)
            max_score = np.max(pole_score)

            total_score += mean_score

            key = round(length, 2)
            results[f"Avg_{key}"] = mean_score
            results[f"Std_{key}"] = std_score
            results[f"Min_{key}"] = min_score
            results[f"Max_{key}"] = max_score

            all_pole_scores.extend(pole_score)

        results["Total_Avg"] = total_score

        # Save raw 300 scores for significance testing
        raw_scores_by_model[model_name] = all_pole_scores

        # Save individual Excel sheet per model
        df = pd.DataFrame([results])
        excel_name = f"weights/{model_name.lower().replace(' ', '_')}_results.xlsx"
        df.to_excel(excel_name, index=False)
        print(f"Saved evaluation data to {excel_name}")

        # Generate Bar Plot for this model
        avgs = {k: v for k, v in results.items() if k.startswith('Avg_')}
        stds = {k.replace('Avg', 'Std'): results[k.replace('Avg', 'Std')] for k in avgs.keys()}
        sorted_keys = sorted(avgs.keys(), key=lambda x: float(x.split('_')[1]))
        avg_values = [avgs[k] for k in sorted_keys]
        std_values = [stds[k.replace('Avg', 'Std')] for k in sorted_keys]
        overall_avg = np.mean(avg_values)

        plt.figure(figsize=(9, 5))
        plt.bar(range(len(avg_values)), avg_values, yerr=std_values, capsize=4, alpha=0.7, color='steelblue')
        plt.xticks(range(len(avg_values)), [k.split('_')[1] for k in sorted_keys], rotation=45)
        plt.xlabel('Pole Length (m)')
        plt.ylabel('Episode Length (Steps)')
        plt.title(f'{model_name} - Overall Avg Score = {round(overall_avg, 1)}')
        plt.tight_layout()

        plot_name = f"weights/{model_name.lower().replace(' ', '_')}_bar_plot.png"
        plt.savefig(plot_name)
        plt.show()
        print(f"Saved bar plot to {plot_name}")

    # Generate comparative training curves from the .npy files
    plot_training_curves()

    # Calculate the statistical significance
    if "Baseline" in raw_scores_by_model:
        baseline_scores = raw_scores_by_model["Baseline"]
        print("\n" + "="*50)
        print("STATISTICAL SIGNIFICANCE TESTING (Welch's t-test)")
        print("="*50)

        for model_name, variant_scores in raw_scores_by_model.items():
            if model_name != "Baseline":
                t_stat, p_val = compute_significance(baseline_scores, variant_scores)
                sig_text = "Statistically Significant (p < 0.05)" if p_val < 0.05 else "Not Significant (p >= 0.05)"
                print(f"Baseline vs {model_name}:")
                print(f"  t-statistic = {t_stat:.4f}")
                print(f"  p-value     = {p_val:.4e}")
                print(f"  Result      = {sig_text}\n")


def test_script():
    """
    Function that simulates the trained NN over 30 different pole lengths, 10 times per length.
    """

    pole_lengths = np.linspace(0.4, 1.8, 30)
    all_results = []

    # import here your trained neural network
    trained_nn = 'PER.pth'

    results = {}
    total_score = 0

    for length in pole_lengths:
        print(length)

        pole_score = []

        for x in range(10):
            # Uncomment to train visually or not
            env = gym.make('CartPole-v1')
            # env = gym.make('CartPole-v1', render_mode="human")
            env.unwrapped.length = length

            state_dim = env.observation_space.shape[0]
            action_dim = env.action_space.n
            loaded_model = QNetwork(state_dim, action_dim)
            loaded_model.load_state_dict(torch.load("weights/" + trained_nn, weights_only=True))

            # Switch to evaluation mode
            loaded_model.eval()  # Use for inference
            score = test_pole_length(env, loaded_model)
            pole_score.append(score)

        mean_score = np.mean(np.array(pole_score))
        std_score = np.std(np.array(pole_score))

        total_score += mean_score

        results[f"Avg_{round(length, 2)}"] = mean_score  # Store just the mean
        results[f"Std_{round(length, 2)}"] = std_score  # Store std separately

    results["Total"] = total_score  # Store just the mean
    all_results.append(results)

    bar_plot(all_results)

    # Convert list to DataFrame
    df = pd.DataFrame(all_results)
    df.to_excel("experiment_results.xlsx", index=False)


if __name__ == "__main__":
    evaluate_all_models()