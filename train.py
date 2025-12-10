"""
Training Loop for PPO
=====================

This file handles:
- Environment interaction
- Data collection
- Agent training
- Logging and checkpointing
"""

import gymnasium as gym
import torch
import numpy as np
from tqdm import tqdm
import json
import os
from datetime import datetime

from networks import ActorCritic
from ppo_agent import PPOAgent


class PPOTrainer:
    """
    Handles the complete training process
    """
    
    def __init__(
        self,
        env_name="CartPole-v1",
        total_timesteps=100000,
        steps_per_update=2048,
        seed=42,
        save_dir="./results"
    ):
        """
        Args:
            env_name: Gymnasium environment name
            total_timesteps: Total number of environment steps
            steps_per_update: How many steps to collect before updating
            seed: Random seed for reproducibility
            save_dir: Directory to save results
        """
        self.env_name = env_name
        self.total_timesteps = total_timesteps
        self.steps_per_update = steps_per_update
        self.seed = seed
        
        # Create save directory with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.save_dir = os.path.join(save_dir, f"{env_name}_{timestamp}")
        os.makedirs(self.save_dir, exist_ok=True)
        
        # Set random seeds for reproducibility
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Create environment
        self.env = gym.make(env_name)
        self.env.action_space.seed(seed)
        
        # Get environment dimensions
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.n
        
        print(f"Environment: {env_name}")
        print(f"State dimension: {self.state_dim}")
        print(f"Action dimension: {self.action_dim}")
        
        # Create network and agent
        self.network = ActorCritic(self.state_dim, self.action_dim)
        self.agent = PPOAgent(self.network)
        
        # Training metrics
        self.episode_rewards = []
        self.episode_lengths = []
        self.timesteps = []
        
        # Current episode tracking
        self.current_episode_reward = 0
        self.current_episode_length = 0
    
    def train(self):
        """
        Main training loop
        
        Process:
        1. Collect experience for steps_per_update steps
        2. Update agent using PPO
        3. Log metrics
        4. Repeat until total_timesteps reached
        """
        state, _ = self.env.reset(seed=self.seed)
        
        # Progress bar
        pbar = tqdm(total=self.total_timesteps, desc="Training PPO")
        
        total_steps = 0
        update_count = 0
        
        while total_steps < self.total_timesteps:
            # ============================================================
            # PHASE 1: Collect Experience
            # ============================================================
            for _ in range(self.steps_per_update):
                # Get action from policy
                action, log_prob, value = self.agent.network.get_action(state)
                
                # Take step in environment
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                
                # Store transition
                self.agent.store_transition(
                    state=state,
                    action=action,
                    reward=reward,
                    value=value.item(),
                    log_prob=log_prob.item(),
                    done=done
                )
                
                # Update episode tracking
                self.current_episode_reward += reward
                self.current_episode_length += 1
                total_steps += 1
                
                # Update state
                state = next_state
                
                # Handle episode end
                if done:
                    # Log episode metrics
                    self.episode_rewards.append(self.current_episode_reward)
                    self.episode_lengths.append(self.current_episode_length)
                    self.timesteps.append(total_steps)
                    
                    # Update progress bar
                    pbar.set_postfix({
                        'reward': f'{self.current_episode_reward:.1f}',
                        'length': self.current_episode_length,
                        'episodes': len(self.episode_rewards)
                    })
                    
                    # Reset episode tracking
                    self.current_episode_reward = 0
                    self.current_episode_length = 0
                    
                    # Reset environment
                    state, _ = self.env.reset()
                
                # Update progress bar
                pbar.update(1)
                
                # Stop if we've reached total_timesteps
                if total_steps >= self.total_timesteps:
                    break
            
            # ============================================================
            # PHASE 2: Update Agent
            # ============================================================
            self.agent.update()
            update_count += 1
            
            # Log every 10 updates
            if update_count % 10 == 0:
                self.log_progress(update_count)
        
        pbar.close()
        
        # Save final results
        self.save_results()
        
        print(f"\n✓ Training completed!")
        print(f"Results saved to: {self.save_dir}")
    
    def log_progress(self, update_count):
        """Log training progress"""
        if len(self.episode_rewards) == 0:
            return
        
        # Compute statistics
        recent_rewards = self.episode_rewards[-100:]  # Last 100 episodes
        mean_reward = np.mean(recent_rewards)
        std_reward = np.std(recent_rewards)
        
        # Get agent metrics
        metrics = self.agent.get_metrics()
        
        log_str = f"\nUpdate {update_count}:"
        log_str += f"\n  Mean Reward (100 ep): {mean_reward:.2f} ± {std_reward:.2f}"
        log_str += f"\n  Episodes: {len(self.episode_rewards)}"
        
        if len(metrics['policy_loss']) > 0:
            log_str += f"\n  Policy Loss: {metrics['policy_loss'][-1]:.4f}"
            log_str += f"\n  Value Loss: {metrics['value_loss'][-1]:.4f}"
            log_str += f"\n  Clipped Fraction: {metrics['clipped_fraction'][-1]:.3f}"
            log_str += f"\n  KL Divergence: {metrics['kl_divergence'][-1]:.4f}"
        
        print(log_str)
    
    def save_results(self):
        """Save training results and model"""
        # Save model
        model_path = os.path.join(self.save_dir, "model.pt")
        torch.save(self.network.state_dict(), model_path)
        
        # Save metrics
        metrics = {
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'timesteps': self.timesteps,
            'agent_metrics': self.agent.get_metrics()
        }
        
        metrics_path = os.path.join(self.save_dir, "metrics.json")
        
        # Convert numpy types to Python types for JSON serialization
        def convert_to_serializable(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            elif isinstance(obj, dict):
                return {key: convert_to_serializable(value) for key, value in obj.items()}
            return obj
        
        metrics = convert_to_serializable(metrics)
        
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Save config
        config = {
            'env_name': self.env_name,
            'total_timesteps': self.total_timesteps,
            'steps_per_update': self.steps_per_update,
            'seed': self.seed,
            'state_dim': self.state_dim,
            'action_dim': self.action_dim
        }
        
        config_path = os.path.join(self.save_dir, "config.json")
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"Saved model to: {model_path}")
        print(f"Saved metrics to: {metrics_path}")
        print(f"Saved config to: {config_path}")
    
    def evaluate(self, num_episodes=10, render=False):
        """
        Evaluate trained policy
        
        Args:
            num_episodes: Number of episodes to evaluate
            render: Whether to render the environment
        """
        eval_rewards = []
        
        for episode in range(num_episodes):
            state, _ = self.env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                if render:
                    self.env.render()
                
                # Get action (deterministic for evaluation)
                action, _, _ = self.agent.network.get_action(state, deterministic=True)
                
                # Take step
                state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                
                episode_reward += reward
            
            eval_rewards.append(episode_reward)
            print(f"Episode {episode + 1}: {episode_reward:.1f}")
        
        mean_reward = np.mean(eval_rewards)
        std_reward = np.std(eval_rewards)
        
        print(f"\nEvaluation over {num_episodes} episodes:")
        print(f"Mean Reward: {mean_reward:.2f} ± {std_reward:.2f}")
        
        return eval_rewards


def main():
    """
    Main function to run training
    """
    # You can try different environments:
    # - CartPole-v1: Simple, fast training (good for testing)
    # - LunarLander-v2: More complex, better for demonstration
    # - Acrobot-v1: Another classic control task
    
    trainer = PPOTrainer(
        env_name="CartPole-v1",
        total_timesteps=100000,
        steps_per_update=2048,
        seed=42
    )
    
    # Train
    trainer.train()
    
    # Evaluate
    print("\n" + "="*50)
    print("Evaluating trained policy...")
    print("="*50)
    trainer.evaluate(num_episodes=10, render=False)


if __name__ == "__main__":
    main()