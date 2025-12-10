"""
PPO Hyperparameter Experiments
================================

This file lets you run systematic experiments to understand how
different hyperparameters affect PPO's performance.

Key experiments:
1. Epsilon comparison (0.1, 0.2, 0.3) - THE MOST IMPORTANT
2. Learning rate comparison
3. GAE lambda comparison
4. Number of epochs comparison

Each experiment shows WHY the default values are chosen.
"""

import os
import sys
import torch
import numpy as np
import gymnasium as gym
from tqdm import tqdm
import json
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from scipy.ndimage import uniform_filter1d

from networks import ActorCritic
from ppo_agent import PPOAgent
from train import PPOTrainer

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 10)


class PPOExperiment:
    """
    Run controlled experiments comparing different hyperparameters
    """
    
    def __init__(
        self,
        env_name="CartPole-v1",
        base_timesteps=50000,
        num_seeds=3,
        save_dir="./experiments"
    ):
        """
        Args:
            env_name: Environment to test on
            base_timesteps: Timesteps per run
            num_seeds: Number of random seeds (for statistical reliability)
            save_dir: Where to save results
        """
        self.env_name = env_name
        self.base_timesteps = base_timesteps
        self.num_seeds = num_seeds
        self.save_dir = save_dir
        
        # Create save directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_dir = os.path.join(save_dir, f"{env_name}_{timestamp}")
        os.makedirs(self.experiment_dir, exist_ok=True)
        
        print(f"Experiment directory: {self.experiment_dir}")
    
    def run_single_experiment(self, config_name, ppo_kwargs, seed):
        """
        Run a single training run with specific hyperparameters
        
        Args:
            config_name: Name of this configuration (e.g., "epsilon_0.2")
            ppo_kwargs: Dictionary of PPO hyperparameters
            seed: Random seed
        
        Returns:
            Dictionary containing all metrics
        """
        print(f"\n{'='*60}")
        print(f"Running: {config_name} (seed={seed})")
        print(f"{'='*60}")
        
        # Set seeds
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Create environment
        env = gym.make(self.env_name)
        env.action_space.seed(seed)
        
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n
        
        # Create network and agent with specific hyperparameters
        network = ActorCritic(state_dim, action_dim)
        agent = PPOAgent(network, **ppo_kwargs)
        
        # Training metrics
        episode_rewards = []
        episode_lengths = []
        timesteps_list = []
        
        # Training loop (simplified version of PPOTrainer)
        state, _ = env.reset(seed=seed)
        current_episode_reward = 0
        current_episode_length = 0
        total_steps = 0
        
        pbar = tqdm(total=self.base_timesteps, desc=f"{config_name} (seed {seed})")
        
        steps_per_update = 2048
        
        while total_steps < self.base_timesteps:
            # Collect experience
            for _ in range(steps_per_update):
                action, log_prob, value = agent.network.get_action(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                agent.store_transition(
                    state=state,
                    action=action,
                    reward=reward,
                    value=value.item(),
                    log_prob=log_prob.item(),
                    done=done
                )
                
                current_episode_reward += reward
                current_episode_length += 1
                total_steps += 1
                
                state = next_state
                
                if done:
                    episode_rewards.append(current_episode_reward)
                    episode_lengths.append(current_episode_length)
                    timesteps_list.append(total_steps)
                    
                    current_episode_reward = 0
                    current_episode_length = 0
                    state, _ = env.reset()
                
                pbar.update(1)
                
                if total_steps >= self.base_timesteps:
                    break
            
            # Update agent
            agent.update()
        
        pbar.close()
        env.close()
        
        # Return results
        return {
            'config_name': config_name,
            'seed': seed,
            'episode_rewards': episode_rewards,
            'episode_lengths': episode_lengths,
            'timesteps': timesteps_list,
            'agent_metrics': agent.get_metrics(),
            'ppo_kwargs': ppo_kwargs
        }
    
    def experiment_epsilon(self, epsilon_values=[0.1, 0.2, 0.3]):
        """
        Experiment 1: Compare different epsilon (clipping) values
        
        This is THE KEY EXPERIMENT for your tutorial!
        
        Why test this?
        - Epsilon is PPO's main hyperparameter
        - Controls how conservative updates are
        - Too small = slow learning
        - Too large = unstable (like vanilla PG)
        
        Args:
            epsilon_values: List of epsilon values to test
        
        Returns:
            Dictionary with all results
        """
        print("\n" + "="*60)
        print("EXPERIMENT 1: EPSILON COMPARISON")
        print("="*60)
        print(f"Testing epsilon values: {epsilon_values}")
        print(f"Seeds: {list(range(self.num_seeds))}")
        print(f"Timesteps per run: {self.base_timesteps:,}")
        print()
        
        results = {}
        
        for epsilon in epsilon_values:
            config_name = f"epsilon_{epsilon}"
            results[config_name] = []
            
            # Run multiple seeds for statistical reliability
            for seed in range(self.num_seeds):
                ppo_kwargs = {
                    'epsilon': epsilon,
                    # Keep other params at default
                    'lr': 3e-4,
                    'gamma': 0.99,
                    'gae_lambda': 0.95,
                    'update_epochs': 10,
                    'batch_size': 64
                }
                
                result = self.run_single_experiment(config_name, ppo_kwargs, seed)
                results[config_name].append(result)
        
        # Save results
        self.save_experiment_results(results, "epsilon_comparison")
        
        # Plot comparison
        self.plot_epsilon_comparison(results)
        
        return results
    
    def experiment_learning_rate(self, lr_values=[1e-4, 3e-4, 1e-3]):
        """
        Experiment 2: Compare different learning rates
        
        Why test this?
        - Learning rate controls optimization speed
        - Too small = slow learning
        - Too large = instability, divergence
        
        Args:
            lr_values: List of learning rates to test
        
        Returns:
            Dictionary with all results
        """
        print("\n" + "="*60)
        print("EXPERIMENT 2: LEARNING RATE COMPARISON")
        print("="*60)
        print(f"Testing learning rates: {lr_values}")
        print()
        
        results = {}
        
        for lr in lr_values:
            config_name = f"lr_{lr}"
            results[config_name] = []
            
            for seed in range(self.num_seeds):
                ppo_kwargs = {
                    'lr': lr,
                    'epsilon': 0.2,  # Use default
                    'gamma': 0.99,
                    'gae_lambda': 0.95,
                    'update_epochs': 10,
                    'batch_size': 64
                }
                
                result = self.run_single_experiment(config_name, ppo_kwargs, seed)
                results[config_name].append(result)
        
        self.save_experiment_results(results, "learning_rate_comparison")
        self.plot_learning_rate_comparison(results)
        
        return results
    
    def experiment_update_epochs(self, epoch_values=[3, 10, 20]):
        """
        Experiment 3: Compare different numbers of update epochs
        
        Why test this?
        - More epochs = more updates per batch of data
        - But too many can lead to overfitting on old data
        - Shows tradeoff between sample efficiency and stability
        
        Args:
            epoch_values: List of epoch counts to test
        
        Returns:
            Dictionary with all results
        """
        print("\n" + "="*60)
        print("EXPERIMENT 3: UPDATE EPOCHS COMPARISON")
        print("="*60)
        print(f"Testing epoch values: {epoch_values}")
        print()
        
        results = {}
        
        for epochs in epoch_values:
            config_name = f"epochs_{epochs}"
            results[config_name] = []
            
            for seed in range(self.num_seeds):
                ppo_kwargs = {
                    'update_epochs': epochs,
                    'epsilon': 0.2,
                    'lr': 3e-4,
                    'gamma': 0.99,
                    'gae_lambda': 0.95,
                    'batch_size': 64
                }
                
                result = self.run_single_experiment(config_name, ppo_kwargs, seed)
                results[config_name].append(result)
        
        self.save_experiment_results(results, "update_epochs_comparison")
        self.plot_epochs_comparison(results)
        
        return results
    
    def save_experiment_results(self, results, experiment_name):
        """Save experiment results to JSON"""
        
        # Convert to JSON-serializable format
        def convert_to_serializable(obj):
            if isinstance(obj, (np.integer, np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            elif isinstance(obj, dict):
                return {key: convert_to_serializable(value) for key, value in obj.items()}
            return obj
        
        results = convert_to_serializable(results)
        
        filename = os.path.join(self.experiment_dir, f"{experiment_name}.json")
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n✓ Results saved to: {filename}")
    
    def plot_epsilon_comparison(self, results):
        """
        Plot comparison of different epsilon values
        
        This creates THE KEY FIGURE for your tutorial!
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('PPO Epsilon Comparison: The Key Hyperparameter', 
                     fontsize=16, fontweight='bold')
        
        # Extract data for each epsilon
        epsilon_data = {}
        
        for config_name, runs in results.items():
            epsilon = float(config_name.split('_')[1])
            
            # Collect all episode rewards from all seeds
            all_rewards = []
            max_length = 0
            
            for run in runs:
                rewards = run['episode_rewards']
                all_rewards.append(rewards)
                max_length = max(max_length, len(rewards))
            
            epsilon_data[epsilon] = all_rewards
        
        # ============================================================
        # Plot 1: Learning Curves with Confidence Intervals
        # ============================================================
        ax = axes[0, 0]
        
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(epsilon_data)))
        
        for idx, (epsilon, all_rewards) in enumerate(sorted(epsilon_data.items())):
            # Interpolate to common length for averaging
            max_len = max(len(r) for r in all_rewards)
            
            interpolated = []
            for rewards in all_rewards:
                if len(rewards) < max_len:
                    # Simple interpolation
                    indices = np.linspace(0, len(rewards)-1, max_len)
                    interp_rewards = np.interp(indices, np.arange(len(rewards)), rewards)
                    interpolated.append(interp_rewards)
                else:
                    interpolated.append(rewards[:max_len])
            
            interpolated = np.array(interpolated)
            mean_rewards = np.mean(interpolated, axis=0)
            std_rewards = np.std(interpolated, axis=0)
            
            # Smooth
            if len(mean_rewards) > 20:
                window = min(20, len(mean_rewards) // 10)
                mean_rewards = uniform_filter1d(mean_rewards, size=window)
                std_rewards = uniform_filter1d(std_rewards, size=window)
            
            episodes = np.arange(len(mean_rewards))
            
            ax.plot(episodes, mean_rewards, linewidth=2.5, 
                   color=colors[idx], label=f'ε = {epsilon}')
            ax.fill_between(episodes, 
                           mean_rewards - std_rewards,
                           mean_rewards + std_rewards,
                           alpha=0.2, color=colors[idx])
        
        ax.set_xlabel('Episode')
        ax.set_ylabel('Episode Reward')
        ax.set_title('Learning Curves (Mean ± Std across seeds)')
        ax.legend()
        ax.grid(alpha=0.3)
        
        # Add interpretation box
        ax.text(0.02, 0.98, 
               'ε = 0.1: Conservative, slower\nε = 0.2: Balanced (default)\nε = 0.3: Aggressive, may be unstable',
               transform=ax.transAxes,
               fontsize=9,
               verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # ============================================================
        # Plot 2: Final Performance (Box Plot)
        # ============================================================
        ax = axes[0, 1]
        
        final_performances = []
        labels = []
        
        for epsilon in sorted(epsilon_data.keys()):
            # Take last 50 episodes from each seed
            finals = []
            for rewards in epsilon_data[epsilon]:
                if len(rewards) >= 50:
                    finals.extend(rewards[-50:])
                else:
                    finals.extend(rewards)
            
            final_performances.append(finals)
            labels.append(f'ε={epsilon}')
        
        bp = ax.boxplot(final_performances, labels=labels, patch_artist=True)
        
        # Color boxes
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        
        ax.set_ylabel('Episode Reward')
        ax.set_title('Final Performance Distribution\n(Last 50 episodes, all seeds)')
        ax.grid(alpha=0.3, axis='y')
        
        # Add mean values as text
        for i, (finals, label) in enumerate(zip(final_performances, labels)):
            mean_val = np.mean(finals)
            ax.text(i+1, ax.get_ylim()[1]*0.95, f'{mean_val:.1f}',
                   ha='center', fontsize=10, fontweight='bold')
        
        # ============================================================
        # Plot 3: Sample Efficiency (Episodes to Threshold)
        # ============================================================
        ax = axes[1, 0]
        
        threshold = 195  # CartPole is "solved" at 195
        
        episodes_to_threshold = []
        epsilon_labels = []
        
        for epsilon in sorted(epsilon_data.keys()):
            eps_to_thresh = []
            
            for rewards in epsilon_data[epsilon]:
                # Find first episode where running average exceeds threshold
                window = 10
                if len(rewards) >= window:
                    running_avg = uniform_filter1d(rewards, size=window)
                    solved_idx = np.where(running_avg >= threshold)[0]
                    
                    if len(solved_idx) > 0:
                        eps_to_thresh.append(solved_idx[0])
                    else:
                        eps_to_thresh.append(len(rewards))  # Didn't solve
            
            if eps_to_thresh:
                episodes_to_threshold.append(eps_to_thresh)
                epsilon_labels.append(f'ε={epsilon}')
        
        if episodes_to_threshold:
            bp = ax.boxplot(episodes_to_threshold, labels=epsilon_labels, patch_artist=True)
            
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
            
            ax.set_ylabel('Episodes to Solve')
            ax.set_title(f'Sample Efficiency\n(Episodes to reach {threshold} reward)')
            ax.grid(alpha=0.3, axis='y')
        
        # ============================================================
        # Plot 4: Stability (Coefficient of Variation)
        # ============================================================
        ax = axes[1, 1]
        
        stability_scores = []
        epsilon_vals = []
        
        for epsilon in sorted(epsilon_data.keys()):
            # Calculate CV (std/mean) for last 50 episodes
            cvs = []
            for rewards in epsilon_data[epsilon]:
                if len(rewards) >= 50:
                    last_50 = rewards[-50:]
                    cv = np.std(last_50) / (np.mean(last_50) + 1e-8)
                    cvs.append(cv)
            
            if cvs:
                stability_scores.append(np.mean(cvs))
                epsilon_vals.append(epsilon)
        
        bars = ax.bar(range(len(epsilon_vals)), stability_scores, color=colors)
        ax.set_xticks(range(len(epsilon_vals)))
        ax.set_xticklabels([f'ε={e}' for e in epsilon_vals])
        ax.set_ylabel('Coefficient of Variation (lower = more stable)')
        ax.set_title('Training Stability\n(CV of last 50 episodes)')
        ax.grid(alpha=0.3, axis='y')
        
        # Add value labels on bars
        for i, (bar, val) in enumerate(zip(bars, stability_scores)):
            ax.text(bar.get_x() + bar.get_width()/2, val + 0.01,
                   f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        # Save
        filename = os.path.join(self.experiment_dir, "epsilon_comparison.png")
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"✓ Saved plot: {filename}")
        
        return fig
    
    def plot_learning_rate_comparison(self, results):
        """Plot learning rate comparison"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Learning Rate Comparison', fontsize=16, fontweight='bold')
        
        # Similar structure to epsilon comparison
        # Extract and plot learning curves for each LR
        
        lr_data = {}
        for config_name, runs in results.items():
            lr = float(config_name.split('_')[1])
            all_rewards = [run['episode_rewards'] for run in runs]
            lr_data[lr] = all_rewards
        
        ax = axes[0, 0]
        colors = plt.cm.plasma(np.linspace(0.2, 0.8, len(lr_data)))
        
        for idx, (lr, all_rewards) in enumerate(sorted(lr_data.items())):
            max_len = max(len(r) for r in all_rewards)
            interpolated = []
            for rewards in all_rewards:
                if len(rewards) < max_len:
                    indices = np.linspace(0, len(rewards)-1, max_len)
                    interp_rewards = np.interp(indices, np.arange(len(rewards)), rewards)
                    interpolated.append(interp_rewards)
                else:
                    interpolated.append(rewards[:max_len])
            
            interpolated = np.array(interpolated)
            mean_rewards = np.mean(interpolated, axis=0)
            std_rewards = np.std(interpolated, axis=0)
            
            if len(mean_rewards) > 20:
                window = min(20, len(mean_rewards) // 10)
                mean_rewards = uniform_filter1d(mean_rewards, size=window)
            
            episodes = np.arange(len(mean_rewards))
            ax.plot(episodes, mean_rewards, linewidth=2.5, 
                   color=colors[idx], label=f'LR = {lr}')
        
        ax.set_xlabel('Episode')
        ax.set_ylabel('Episode Reward')
        ax.set_title('Learning Curves for Different Learning Rates')
        ax.legend()
        ax.grid(alpha=0.3)
        
        # Add other subplots...
        axes[0, 1].text(0.5, 0.5, 'Final Performance\n(Coming soon)', 
                       ha='center', va='center', transform=axes[0, 1].transAxes)
        axes[1, 0].text(0.5, 0.5, 'Convergence Speed\n(Coming soon)', 
                       ha='center', va='center', transform=axes[1, 0].transAxes)
        axes[1, 1].text(0.5, 0.5, 'Stability\n(Coming soon)', 
                       ha='center', va='center', transform=axes[1, 1].transAxes)
        
        plt.tight_layout()
        
        filename = os.path.join(self.experiment_dir, "learning_rate_comparison.png")
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"✓ Saved plot: {filename}")
        
        return fig
    
    def plot_epochs_comparison(self, results):
        """Plot update epochs comparison"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Update Epochs Comparison', fontsize=16, fontweight='bold')
        
        # Similar to above...
        epochs_data = {}
        for config_name, runs in results.items():
            epochs = int(config_name.split('_')[1])
            all_rewards = [run['episode_rewards'] for run in runs]
            epochs_data[epochs] = all_rewards
        
        ax = axes[0, 0]
        colors = plt.cm.cool(np.linspace(0.2, 0.8, len(epochs_data)))
        
        for idx, (epochs, all_rewards) in enumerate(sorted(epochs_data.items())):
            max_len = max(len(r) for r in all_rewards)
            interpolated = []
            for rewards in all_rewards:
                if len(rewards) < max_len:
                    indices = np.linspace(0, len(rewards)-1, max_len)
                    interp_rewards = np.interp(indices, np.arange(len(rewards)), rewards)
                    interpolated.append(interp_rewards)
                else:
                    interpolated.append(rewards[:max_len])
            
            interpolated = np.array(interpolated)
            mean_rewards = np.mean(interpolated, axis=0)
            
            if len(mean_rewards) > 20:
                window = min(20, len(mean_rewards) // 10)
                mean_rewards = uniform_filter1d(mean_rewards, size=window)
            
            episodes_arr = np.arange(len(mean_rewards))
            ax.plot(episodes_arr, mean_rewards, linewidth=2.5, 
                   color=colors[idx], label=f'{epochs} epochs')
        
        ax.set_xlabel('Episode')
        ax.set_ylabel('Episode Reward')
        ax.set_title('Learning Curves for Different Update Epochs')
        ax.legend()
        ax.grid(alpha=0.3)
        
        axes[0, 1].text(0.5, 0.5, 'Sample Efficiency\n(Coming soon)', 
                       ha='center', va='center', transform=axes[0, 1].transAxes)
        axes[1, 0].text(0.5, 0.5, 'Wall Clock Time\n(Coming soon)', 
                       ha='center', va='center', transform=axes[1, 0].transAxes)
        axes[1, 1].text(0.5, 0.5, 'Overfitting Risk\n(Coming soon)', 
                       ha='center', va='center', transform=axes[1, 1].transAxes)
        
        plt.tight_layout()
        
        filename = os.path.join(self.experiment_dir, "update_epochs_comparison.png")
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"✓ Saved plot: {filename}")
        
        return fig


def main():
    """
    Run all experiments
    """
    print("\n" + "="*60)
    print("PPO HYPERPARAMETER EXPERIMENTS")
    print("="*60)
    print("\nThis will run systematic experiments to compare:")
    print("  1. Epsilon values (clipping parameter)")
    print("  2. Learning rates")
    print("  3. Update epochs")
    print("\nEach experiment runs multiple seeds for reliability.")
    print(f"\nEstimated time: 30-60 minutes")
    
    response = input("\nContinue? (yes/no): ").strip().lower()
    if response not in ['yes', 'y']:
        print("Cancelled.")
        return
    
    # Create experiment runner
    experiment = PPOExperiment(
        env_name="CartPole-v1",
        base_timesteps=50000,  # Faster for experiments
        num_seeds=3  # Statistical reliability
    )
    
    # Run experiments
    print("\n" + "="*60)
    print("STARTING EXPERIMENTS")
    print("="*60)
    
    # Experiment 1: Epsilon (MOST IMPORTANT!)
    print("\n🔬 Running Epsilon Experiment...")
    epsilon_results = experiment.experiment_epsilon(epsilon_values=[0.1, 0.2, 0.3])
    
    # Experiment 2: Learning Rate
    print("\n🔬 Running Learning Rate Experiment...")
    lr_results = experiment.experiment_learning_rate(lr_values=[1e-4, 3e-4, 1e-3])
    
    # Experiment 3: Update Epochs
    print("\n🔬 Running Update Epochs Experiment...")
    epochs_results = experiment.experiment_update_epochs(epoch_values=[3, 10, 20])
    
    print("\n" + "="*60)
    print("ALL EXPERIMENTS COMPLETE!")
    print("="*60)
    print(f"\nResults saved to: {experiment.experiment_dir}")
    print("\nGenerated plots:")
    print("  • epsilon_comparison.png (KEY FIGURE!)")
    print("  • learning_rate_comparison.png")
    print("  • update_epochs_comparison.png")
    print("\nUse these in your tutorial to show:")
    print("  - Why ε=0.2 is the default")
    print("  - How different hyperparameters affect performance")
    print("  - The importance of proper tuning")


if __name__ == "__main__":
    main()