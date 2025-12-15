"""
Training Loop for PPO Glucose Control
======================================

Handles the complete training process:
- Environment interaction
- Experience collection
- Agent updates
- Logging and checkpointing
- Progress tracking

Based on standard PPO training practices with adaptations
for the glucose control domain.
"""

import os
import json
import numpy as np
import torch
from datetime import datetime
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple

from environment import GlucoseInsulinEnv
from networks import ContinuousActorCritic
from ppo_agent import PPOAgent


class PPOTrainer:
    """
    Trainer class for PPO glucose control.
    
    Manages the training loop, logging, and checkpointing.
    Tracks both RL metrics (rewards, losses) and clinical
    metrics (time in range, hypoglycaemia events).
    """
    
    def __init__(
        self,
        env_config: Optional[Dict] = None,
        agent_config: Optional[Dict] = None,
        total_timesteps: int = 100000,
        steps_per_update: int = 2048,
        eval_frequency: int = 10,
        eval_episodes: int = 5,
        save_frequency: int = 20,
        seed: int = 42,
        save_dir: str = "./results",
        experiment_name: Optional[str] = None
    ):
        """
        Initialise the trainer.
        
        Args:
            env_config: Configuration for environment
            agent_config: Configuration for PPO agent
            total_timesteps: Total training timesteps
            steps_per_update: Steps to collect before each update
            eval_frequency: Evaluate every N updates
            eval_episodes: Number of episodes for evaluation
            save_frequency: Save model every N updates
            seed: Random seed for reproducibility
            save_dir: Directory to save results
            experiment_name: Name for this experiment
        """
        # Set seeds for reproducibility
        self.seed = seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Default configurations
        self.env_config = env_config or {
            'max_insulin_dose': 5.0,
            'episode_length_hours': 24.0,
            'sample_time_minutes': 5.0,
            'target_glucose_min': 70.0,
            'target_glucose_max': 180.0,
            'patient_variability': True,
            'meal_variability': True
        }
        
        self.agent_config = agent_config or {
            'hidden_dim': 64,
            'lr': 3e-4,
            'gamma': 0.99,
            'gae_lambda': 0.95,
            'epsilon': 0.2,
            'value_coef': 0.5,
            'entropy_coef': 0.01,
            'max_grad_norm': 0.5,
            'update_epochs': 10,
            'batch_size': 64
        }
        
        # Training parameters
        self.total_timesteps = total_timesteps
        self.steps_per_update = steps_per_update
        self.eval_frequency = eval_frequency
        self.eval_episodes = eval_episodes
        self.save_frequency = save_frequency
        
        # Create save directory
        if experiment_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            experiment_name = f"glucose_ppo_{timestamp}"
        
        self.save_dir = os.path.join(save_dir, experiment_name)
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(os.path.join(self.save_dir, "models"), exist_ok=True)
        os.makedirs(os.path.join(self.save_dir, "figures"), exist_ok=True)
        
        # Create environment
        self.env = GlucoseInsulinEnv(**self.env_config, seed=seed)
        
        # Get dimensions
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]
        
        # Create agent
        self.agent = PPOAgent(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            action_low=0.0,
            action_high=self.env_config['max_insulin_dose'],
            **self.agent_config
        )
        
        # Training metrics
        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int] = []
        self.episode_timesteps: List[int] = []
        
        # Clinical metrics per episode
        self.clinical_metrics: List[Dict] = []
        
        # Evaluation metrics
        self.eval_rewards: List[float] = []
        self.eval_timesteps: List[int] = []
        self.eval_clinical: List[Dict] = []
        
        # Current episode tracking
        self.current_episode_reward = 0.0
        self.current_episode_length = 0
        
        # Print configuration
        print("=" * 60)
        print("PPO Glucose Control Trainer")
        print("=" * 60)
        print(f"\nEnvironment:")
        print(f"  State dimension: {self.state_dim}")
        print(f"  Action dimension: {self.action_dim}")
        print(f"  Max insulin dose: {self.env_config['max_insulin_dose']} U/hr")
        print(f"  Episode length: {self.env_config['episode_length_hours']} hours")
        print(f"  Sample time: {self.env_config['sample_time_minutes']} minutes")
        print(f"\nAgent:")
        print(f"  Hidden dim: {self.agent_config['hidden_dim']}")
        print(f"  Learning rate: {self.agent_config['lr']}")
        print(f"  Epsilon (clip): {self.agent_config['epsilon']}")
        print(f"  Gamma: {self.agent_config['gamma']}")
        print(f"  GAE lambda: {self.agent_config['gae_lambda']}")
        print(f"\nTraining:")
        print(f"  Total timesteps: {total_timesteps:,}")
        print(f"  Steps per update: {steps_per_update}")
        print(f"  Seed: {seed}")
        print(f"\nSave directory: {self.save_dir}")
        print("=" * 60)
    
    def train(self) -> Dict:
        """
        Run the full training loop.
        
        Returns:
            Dictionary containing all training metrics
        """
        print("\nStarting training...")
        
        # Initialise environment
        state, info = self.env.reset(seed=self.seed)
        
        # Progress bar
        pbar = tqdm(total=self.total_timesteps, desc="Training PPO")
        
        total_steps = 0
        update_count = 0
        episode_count = 0
        
        while total_steps < self.total_timesteps:
            # ============================================================
            # PHASE 1: Collect experience
            # ============================================================
            for step in range(self.steps_per_update):
                # Get action from policy
                action, log_prob, value = self.agent.get_action(state)
                
                # Take step in environment
                next_state, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                
                # Store transition
                self.agent.store_transition(
                    state=state,
                    action=action,
                    reward=reward,
                    value=value,
                    log_prob=log_prob,
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
                    # Get clinical metrics
                    clinical = self.env.get_episode_stats()
                    
                    # Store episode metrics
                    self.episode_rewards.append(self.current_episode_reward)
                    self.episode_lengths.append(self.current_episode_length)
                    self.episode_timesteps.append(total_steps)
                    self.clinical_metrics.append(clinical)
                    
                    episode_count += 1
                    
                    # Update progress bar
                    pbar.set_postfix({
                        'reward': f'{self.current_episode_reward:.1f}',
                        'TIR': f'{clinical["time_in_range"]:.1f}%',
                        'episodes': episode_count
                    })
                    
                    # Reset episode tracking
                    self.current_episode_reward = 0.0
                    self.current_episode_length = 0
                    
                    # Reset environment
                    state, info = self.env.reset()
                
                # Update progress bar
                pbar.update(1)
                
                # Check if we've reached total timesteps
                if total_steps >= self.total_timesteps:
                    break
            
            # ============================================================
            # PHASE 2: Update agent
            # ============================================================
            # Get value estimate for final state (for GAE)
            if not done:
                next_value = self.agent.network.get_value(state)
            else:
                next_value = 0.0
            
            # Perform PPO update
            update_metrics = self.agent.update(next_value=next_value, next_done=done)
            update_count += 1
            
            # ============================================================
            # PHASE 3: Logging and evaluation
            # ============================================================
            # Log progress every 10 updates
            if update_count % 10 == 0:
                self._log_progress(update_count, total_steps, update_metrics)
            
            # Evaluate agent
            if update_count % self.eval_frequency == 0:
                eval_results = self.evaluate(num_episodes=self.eval_episodes)
                self.eval_rewards.append(eval_results['mean_reward'])
                self.eval_timesteps.append(total_steps)
                self.eval_clinical.append(eval_results)
                
                print(f"\n  Evaluation (update {update_count}):")
                print(f"    Mean reward: {eval_results['mean_reward']:.1f} ± {eval_results['std_reward']:.1f}")
                print(f"    Time in Range: {eval_results['mean_tir']:.1f}%")
                print(f"    Time Below Range: {eval_results['mean_tbr']:.1f}%")
            
            # Save checkpoint
            if update_count % self.save_frequency == 0:
                self._save_checkpoint(update_count, total_steps)
        
        pbar.close()
        
        # Final save
        self._save_results()
        
        print("\n" + "=" * 60)
        print("Training complete!")
        print("=" * 60)
        print(f"Total timesteps: {total_steps:,}")
        print(f"Total episodes: {episode_count}")
        print(f"Total updates: {update_count}")
        print(f"\nResults saved to: {self.save_dir}")
        
        return self._get_training_results()
    
    def evaluate(
        self,
        num_episodes: int = 10,
        deterministic: bool = True,
        render: bool = False
    ) -> Dict:
        """
        Evaluate the current policy.
        
        Args:
            num_episodes: Number of evaluation episodes
            deterministic: Use deterministic policy (no sampling)
            render: Whether to render episodes
        
        Returns:
            Dictionary of evaluation metrics
        """
        eval_rewards = []
        eval_clinical = []
        
        # Create separate evaluation environment
        eval_env = GlucoseInsulinEnv(**self.env_config, seed=self.seed + 1000)
        
        for episode in range(num_episodes):
            state, _ = eval_env.reset(seed=self.seed + 1000 + episode)
            episode_reward = 0.0
            done = False
            
            while not done:
                if render:
                    eval_env.render()
                
                # Get action (deterministic for evaluation)
                action, _, _ = self.agent.get_action(state, deterministic=deterministic)
                
                # Take step
                state, reward, terminated, truncated, info = eval_env.step(action)
                done = terminated or truncated
                episode_reward += reward
            
            eval_rewards.append(episode_reward)
            eval_clinical.append(eval_env.get_episode_stats())
        
        # Compute statistics
        results = {
            'mean_reward': np.mean(eval_rewards),
            'std_reward': np.std(eval_rewards),
            'min_reward': np.min(eval_rewards),
            'max_reward': np.max(eval_rewards),
            'mean_tir': np.mean([c['time_in_range'] for c in eval_clinical]),
            'mean_tbr': np.mean([c['time_below_range'] for c in eval_clinical]),
            'mean_tar': np.mean([c['time_above_range'] for c in eval_clinical]),
            'mean_glucose': np.mean([c['mean_glucose'] for c in eval_clinical]),
            'mean_cv': np.mean([c['glucose_cv'] for c in eval_clinical]),
            'mean_insulin': np.mean([c['total_insulin'] for c in eval_clinical])
        }
        
        return results
    
    def _log_progress(
        self,
        update_count: int,
        total_steps: int,
        update_metrics: Dict
    ) -> None:
        """Log training progress."""
        if len(self.episode_rewards) == 0:
            return
        
        # Get recent episode statistics
        recent_rewards = self.episode_rewards[-100:]
        recent_clinical = self.clinical_metrics[-100:]
        
        mean_reward = np.mean(recent_rewards)
        std_reward = np.std(recent_rewards)
        mean_tir = np.mean([c['time_in_range'] for c in recent_clinical])
        
        print(f"\nUpdate {update_count} | Steps: {total_steps:,}")
        print(f"  Episodes: {len(self.episode_rewards)}")
        print(f"  Mean reward (100 ep): {mean_reward:.1f} ± {std_reward:.1f}")
        print(f"  Mean TIR (100 ep): {mean_tir:.1f}%")
        
        if update_metrics:
            print(f"  Policy loss: {update_metrics['policy_loss']:.4f}")
            print(f"  Value loss: {update_metrics['value_loss']:.4f}")
            print(f"  Clipped fraction: {update_metrics['clipped_fraction']:.3f}")
            print(f"  KL divergence: {update_metrics['kl_divergence']:.4f}")
    
    def _save_checkpoint(self, update_count: int, total_steps: int) -> None:
        """Save training checkpoint."""
        checkpoint_path = os.path.join(
            self.save_dir, "models", f"checkpoint_{update_count}.pt"
        )
        self.agent.save(checkpoint_path)
        print(f"\n  Saved checkpoint: {checkpoint_path}")
    
    def _save_results(self) -> None:
        """Save all training results."""
        # Save final model
        model_path = os.path.join(self.save_dir, "models", "final_model.pt")
        self.agent.save(model_path)
        
        # Prepare metrics for JSON serialisation
        def convert_to_serialisable(obj):
            if isinstance(obj, (np.integer, np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, list):
                return [convert_to_serialisable(item) for item in obj]
            elif isinstance(obj, dict):
                return {key: convert_to_serialisable(value) for key, value in obj.items()}
            return obj
        
        # Save training metrics
        metrics = {
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'episode_timesteps': self.episode_timesteps,
            'clinical_metrics': self.clinical_metrics,
            'eval_rewards': self.eval_rewards,
            'eval_timesteps': self.eval_timesteps,
            'eval_clinical': self.eval_clinical,
            'agent_metrics': self.agent.get_metrics()
        }
        
        metrics = convert_to_serialisable(metrics)
        
        metrics_path = os.path.join(self.save_dir, "metrics.json")
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Save configuration
        config = {
            'env_config': self.env_config,
            'agent_config': self.agent_config,
            'total_timesteps': self.total_timesteps,
            'steps_per_update': self.steps_per_update,
            'seed': self.seed,
            'state_dim': self.state_dim,
            'action_dim': self.action_dim
        }
        
        config = convert_to_serialisable(config)
        
        config_path = os.path.join(self.save_dir, "config.json")
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"\nSaved model to: {model_path}")
        print(f"Saved metrics to: {metrics_path}")
        print(f"Saved config to: {config_path}")
    
    def _get_training_results(self) -> Dict:
        """Get summary of training results."""
        if len(self.episode_rewards) == 0:
            return {}
        
        # Final evaluation
        final_eval = self.evaluate(num_episodes=10)
        
        return {
            'total_episodes': len(self.episode_rewards),
            'final_mean_reward': np.mean(self.episode_rewards[-100:]),
            'final_std_reward': np.std(self.episode_rewards[-100:]),
            'best_reward': np.max(self.episode_rewards),
            'final_eval': final_eval,
            'save_dir': self.save_dir
        }


def train_ppo_glucose(
    total_timesteps: int = 100000,
    seed: int = 42,
    experiment_name: Optional[str] = None
) -> Dict:
    """
    Convenience function to train PPO on glucose control.
    
    Args:
        total_timesteps: Total training timesteps
        seed: Random seed
        experiment_name: Name for experiment
    
    Returns:
        Training results dictionary
    """
    trainer = PPOTrainer(
        total_timesteps=total_timesteps,
        seed=seed,
        experiment_name=experiment_name
    )
    
    results = trainer.train()
    return results


def main():
    """Main training script."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train PPO for glucose control")
    parser.add_argument(
        "--timesteps", type=int, default=100000,
        help="Total training timesteps"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--name", type=str, default=None,
        help="Experiment name"
    )
    parser.add_argument(
        "--steps-per-update", type=int, default=2048,
        help="Steps to collect before each update"
    )
    parser.add_argument(
        "--eval-frequency", type=int, default=10,
        help="Evaluate every N updates"
    )
    
    args = parser.parse_args()
    
    # Create trainer
    trainer = PPOTrainer(
        total_timesteps=args.timesteps,
        steps_per_update=args.steps_per_update,
        eval_frequency=args.eval_frequency,
        seed=args.seed,
        experiment_name=args.name
    )
    
    # Train
    results = trainer.train()
    
    # Print final results
    print("\n" + "=" * 60)
    print("Final Results")
    print("=" * 60)
    print(f"Total episodes: {results['total_episodes']}")
    print(f"Final mean reward: {results['final_mean_reward']:.1f} ± {results['final_std_reward']:.1f}")
    print(f"Best episode reward: {results['best_reward']:.1f}")
    print(f"\nFinal Evaluation:")
    print(f"  Mean reward: {results['final_eval']['mean_reward']:.1f}")
    print(f"  Time in Range: {results['final_eval']['mean_tir']:.1f}%")
    print(f"  Time Below Range: {results['final_eval']['mean_tbr']:.1f}%")
    print(f"  Mean glucose: {results['final_eval']['mean_glucose']:.1f} mg/dL")


if __name__ == "__main__":
    main()