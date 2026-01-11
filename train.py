"""
Training Loop for PPO Glucose Control
======================================

STABILITY-FIXED VERSION with conservative hyperparameters.

Author: [Your Name]
Date: December 2024
"""

import os
import json
import numpy as np
import torch
from datetime import datetime
from tqdm import tqdm
from typing import Dict, List, Optional

from environment import GlucoseInsulinEnv
from ppo_agent import PPOAgent


class PPOTrainer:
    """Trainer class for PPO glucose control - STABILITY FIXED."""
    
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
        self.seed = seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        self.env_config = env_config or {
            'max_insulin_dose': 5.0,
            'episode_length_hours': 24.0,
            'sample_time_minutes': 5.0,
            'target_glucose_min': 70.0,
            'target_glucose_max': 180.0,
            'patient_variability': True,
            'meal_variability': True
        }
        
        # STABILITY FIX: Conservative hyperparameters
        self.agent_config = agent_config or {
            'hidden_dim': 64,
            'lr': 1e-4,            # Reduced from 3e-4
            'gamma': 0.99,
            'gae_lambda': 0.95,
            'epsilon': 0.1,        # Reduced from 0.2
            'value_coef': 0.5,
            'entropy_coef': 0.01,
            'max_grad_norm': 0.5,
            'update_epochs': 4,    # Reduced from 10
            'batch_size': 64
        }
        
        self.total_timesteps = total_timesteps
        self.steps_per_update = steps_per_update
        self.eval_frequency = eval_frequency
        self.eval_episodes = eval_episodes
        self.save_frequency = save_frequency
        
        if experiment_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            experiment_name = f"glucose_ppo_{timestamp}"
        
        self.save_dir = os.path.join(save_dir, experiment_name)
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(os.path.join(self.save_dir, "models"), exist_ok=True)
        os.makedirs(os.path.join(self.save_dir, "figures"), exist_ok=True)
        
        self.env = GlucoseInsulinEnv(**self.env_config, seed=seed)
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]
        
        self.agent = PPOAgent(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            action_low=0.0,
            action_high=self.env_config['max_insulin_dose'],
            **self.agent_config
        )
        
        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int] = []
        self.episode_timesteps: List[int] = []
        self.clinical_metrics: List[Dict] = []
        self.eval_rewards: List[float] = []
        self.eval_timesteps: List[int] = []
        self.eval_clinical: List[Dict] = []
        self.current_episode_reward = 0.0
        self.current_episode_length = 0
        
        print("=" * 60)
        print("PPO Glucose Control Trainer (STABILITY FIXED)")
        print("=" * 60)
        print(f"\nEnvironment:")
        print(f"  State dimension: {self.state_dim}")
        print(f"  Action dimension: {self.action_dim}")
        print(f"  Max insulin dose: {self.env_config['max_insulin_dose']} U/hr")
        print(f"\nAgent (CONSERVATIVE):")
        print(f"  Learning rate: {self.agent_config['lr']}")
        print(f"  Epsilon (clip): {self.agent_config['epsilon']}")
        print(f"  Update epochs: {self.agent_config['update_epochs']}")
        print(f"\nTraining:")
        print(f"  Total timesteps: {total_timesteps:,}")
        print(f"  Seed: {seed}")
        print(f"\nSave directory: {self.save_dir}")
        print("=" * 60)
    
    def train(self) -> Dict:
        print("\nStarting training...")
        
        state, info = self.env.reset(seed=self.seed)
        pbar = tqdm(total=self.total_timesteps, desc="Training PPO")
        
        total_steps = 0
        update_count = 0
        episode_count = 0
        
        while total_steps < self.total_timesteps:
            for step in range(self.steps_per_update):
                action, log_prob, value = self.agent.get_action(state)
                
                if np.isnan(action).any():
                    action = np.array([1.0])
                
                next_state, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                
                self.agent.store_transition(state, action, reward, value, log_prob, done)
                
                self.current_episode_reward += reward
                self.current_episode_length += 1
                total_steps += 1
                state = next_state
                
                if done:
                    clinical = self.env.get_episode_stats()
                    self.episode_rewards.append(self.current_episode_reward)
                    self.episode_lengths.append(self.current_episode_length)
                    self.episode_timesteps.append(total_steps)
                    self.clinical_metrics.append(clinical)
                    episode_count += 1
                    
                    pbar.set_postfix({
                        'reward': f'{self.current_episode_reward:.1f}',
                        'TIR': f'{clinical["time_in_range"]:.1f}%',
                        'ep': episode_count
                    })
                    
                    self.current_episode_reward = 0.0
                    self.current_episode_length = 0
                    state, info = self.env.reset()
                
                pbar.update(1)
                if total_steps >= self.total_timesteps:
                    break
            
            if not done:
                next_value = self.agent.network.get_value(state)
            else:
                next_value = 0.0
            
            update_metrics = self.agent.update(next_value=next_value, next_done=done)
            update_count += 1
            
            if update_count % 10 == 0:
                self._log_progress(update_count, total_steps, update_metrics)
            
            if update_count % self.eval_frequency == 0:
                eval_results = self.evaluate(num_episodes=self.eval_episodes)
                self.eval_rewards.append(eval_results['mean_reward'])
                self.eval_timesteps.append(total_steps)
                self.eval_clinical.append(eval_results)
                print(f"\n  Evaluation (update {update_count}):")
                print(f"    Mean reward: {eval_results['mean_reward']:.1f}")
                print(f"    TIR: {eval_results['mean_tir']:.1f}% | TBR: {eval_results['mean_tbr']:.1f}%")
            
            if update_count % self.save_frequency == 0:
                self._save_checkpoint(update_count)
        
        pbar.close()
        self._save_results()
        
        print("\n" + "=" * 60)
        print("Training complete!")
        print(f"Results saved to: {self.save_dir}")
        print("=" * 60)
        
        return self._get_training_results()
    
    def evaluate(self, num_episodes: int = 10, deterministic: bool = True) -> Dict:
        eval_rewards = []
        eval_clinical = []
        eval_env = GlucoseInsulinEnv(**self.env_config, seed=self.seed + 1000)
        
        for episode in range(num_episodes):
            state, _ = eval_env.reset(seed=self.seed + 1000 + episode)
            episode_reward = 0.0
            done = False
            
            while not done:
                action, _, _ = self.agent.get_action(state, deterministic=deterministic)
                state, reward, terminated, truncated, _ = eval_env.step(action)
                done = terminated or truncated
                episode_reward += reward
            
            eval_rewards.append(episode_reward)
            eval_clinical.append(eval_env.get_episode_stats())
        
        return {
            'mean_reward': np.mean(eval_rewards),
            'std_reward': np.std(eval_rewards),
            'mean_tir': np.mean([c['time_in_range'] for c in eval_clinical]),
            'mean_tbr': np.mean([c['time_below_range'] for c in eval_clinical]),
            'mean_tar': np.mean([c['time_above_range'] for c in eval_clinical]),
            'mean_glucose': np.mean([c['mean_glucose'] for c in eval_clinical]),
            'mean_cv': np.mean([c['glucose_cv'] for c in eval_clinical]),
            'mean_insulin': np.mean([c['total_insulin'] for c in eval_clinical])
        }
    
    def _log_progress(self, update_count: int, total_steps: int, update_metrics: Dict) -> None:
        if len(self.episode_rewards) == 0:
            return
        
        recent = min(100, len(self.episode_rewards))
        mean_reward = np.mean(self.episode_rewards[-recent:])
        mean_tir = np.mean([c['time_in_range'] for c in self.clinical_metrics[-recent:]])
        
        print(f"\nUpdate {update_count} | Steps: {total_steps:,}")
        print(f"  Mean reward ({recent} ep): {mean_reward:.1f}")
        print(f"  Mean TIR: {mean_tir:.1f}%")
        
        if update_metrics:
            kl = update_metrics.get('kl_divergence', 0)
            kl_str = f"{kl:.4f}" if not np.isinf(kl) else "inf ⚠️"
            print(f"  Policy loss: {update_metrics.get('policy_loss', 0):.4f}")
            print(f"  KL divergence: {kl_str}")
    
    def _save_checkpoint(self, update_count: int) -> None:
        path = os.path.join(self.save_dir, "models", f"checkpoint_{update_count}.pt")
        self.agent.save(path)
        print(f"\n  Saved: {path}")
    
    def _save_results(self) -> None:
        self.agent.save(os.path.join(self.save_dir, "models", "final_model.pt"))
        
        def convert(obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            elif isinstance(obj, (np.floating,)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, list):
                return [convert(i) for i in obj]
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            return obj
        
        metrics = convert({
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'episode_timesteps': self.episode_timesteps,
            'clinical_metrics': self.clinical_metrics,
            'eval_rewards': self.eval_rewards,
            'eval_timesteps': self.eval_timesteps,
            'eval_clinical': self.eval_clinical,
            'agent_metrics': self.agent.get_metrics()
        })
        
        with open(os.path.join(self.save_dir, "metrics.json"), 'w') as f:
            json.dump(metrics, f, indent=2)
        
        config = convert({
            'env_config': self.env_config,
            'agent_config': self.agent_config,
            'total_timesteps': self.total_timesteps,
            'seed': self.seed
        })
        
        with open(os.path.join(self.save_dir, "config.json"), 'w') as f:
            json.dump(config, f, indent=2)
    
    def _get_training_results(self) -> Dict:
        if len(self.episode_rewards) == 0:
            return {}
        
        final_eval = self.evaluate(num_episodes=10)
        recent = min(100, len(self.episode_rewards))
        
        return {
            'total_episodes': len(self.episode_rewards),
            'final_mean_reward': np.mean(self.episode_rewards[-recent:]),
            'final_std_reward': np.std(self.episode_rewards[-recent:]),
            'best_reward': np.max(self.episode_rewards),
            'final_eval': final_eval,
            'save_dir': self.save_dir
        }


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Train PPO for glucose control")
    parser.add_argument("--timesteps", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--steps-per-update", type=int, default=2048)
    parser.add_argument("--eval-frequency", type=int, default=10)
    
    args = parser.parse_args()
    
    trainer = PPOTrainer(
        total_timesteps=args.timesteps,
        steps_per_update=args.steps_per_update,
        eval_frequency=args.eval_frequency,
        seed=args.seed,
        experiment_name=args.name
    )
    
    results = trainer.train()
    
    print("\n" + "=" * 60)
    print("Final Results")
    print("=" * 60)
    print(f"Episodes: {results['total_episodes']}")
    print(f"Final reward: {results['final_mean_reward']:.1f}")
    print(f"Final TIR: {results['final_eval']['mean_tir']:.1f}%")
    print(f"Final TBR: {results['final_eval']['mean_tbr']:.1f}%")


if __name__ == "__main__":
    main()