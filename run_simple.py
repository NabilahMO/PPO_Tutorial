"""
Quick Start Script for PPO Tutorial
====================================

This script runs everything you need in one go:
1. Train PPO on CartPole
2. Generate all visualizations
3. Save everything to a results folder

Usage:
    python run_simple.py
"""

import os
import sys
import torch
import numpy as np
import gymnasium as gym
from datetime import datetime

# Import your modules
from networks import ActorCritic
from ppo_agent import PPOAgent
from train import PPOTrainer
from visualize import PPOVisualizer


def quick_train(
    env_name="CartPole-v1",
    total_timesteps=50000,  # Lower for quick testing
    seed=42,
    show_progress=True
):
    """
    Quick training function with sensible defaults
    
    Args:
        env_name: Environment to train on
        total_timesteps: How long to train (50k is quick, 100k is better)
        seed: Random seed
        show_progress: Whether to print progress
    """
    print("="*60)
    print("PPO QUICK START")
    print("="*60)
    print(f"\nEnvironment: {env_name}")
    print(f"Total timesteps: {total_timesteps:,}")
    print(f"Random seed: {seed}")
    print("\nThis will take a few minutes...\n")
    
    # Create trainer
    trainer = PPOTrainer(
        env_name=env_name,
        total_timesteps=total_timesteps,
        steps_per_update=2048,
        seed=seed
    )
    
    # Train
    print("Starting training...")
    trainer.train()
    
    # Evaluate
    print("\n" + "="*60)
    print("EVALUATION")
    print("="*60)
    eval_rewards = trainer.evaluate(num_episodes=10, render=False)
    
    print(f"\nMean Reward: {np.mean(eval_rewards):.2f} ± {np.std(eval_rewards):.2f}")
    
    return trainer.save_dir


def quick_visualize(results_dir):
    """
    Generate all visualizations
    
    Args:
        results_dir: Directory containing training results
    """
    print("\n" + "="*60)
    print("GENERATING VISUALIZATIONS")
    print("="*60)
    
    # Create visualizer
    viz = PPOVisualizer(results_dir)
    
    # Generate all plots
    viz.plot_all(save=True)
    
    print("\n✓ All visualizations saved!")
    print(f"\nCheck this folder for PNG files:")
    print(f"  {results_dir}")


def main():
    """
    Main function - runs everything!
    """
    # Check if results directory already exists
    if len(sys.argv) > 1:
        # User provided a results directory - just visualize
        results_dir = sys.argv[1]
        print(f"Visualizing existing results from: {results_dir}")
        quick_visualize(results_dir)
        return
    
    # Otherwise, train from scratch
    print("\n" + "="*60)
    print("WELCOME TO PPO TRAINING!")
    print("="*60)
    print("\nThis script will:")
    print("  1. Train a PPO agent on CartPole")
    print("  2. Evaluate the trained agent")
    print("  3. Generate all visualizations")
    print("\nEstimated time: 5-10 minutes")
    
    # Ask user for confirmation
    response = input("\nContinue? (yes/no): ").strip().lower()
    if response not in ['yes', 'y']:
        print("Cancelled.")
        return
    
    # Choose environment
    print("\nAvailable environments:")
    print("  1. CartPole-v1 (Easy, fast - recommended for testing)")
    print("  2. LunarLander-v2 (Harder, slower - better for showcase)")
    print("  3. Acrobot-v1 (Medium difficulty)")
    
    env_choice = input("\nChoose environment (1/2/3) [default=1]: ").strip()
    
    env_map = {
        '1': ('CartPole-v1', 50000),
        '2': ('LunarLander-v2', 100000),
        '3': ('Acrobot-v1', 100000),
        '': ('CartPole-v1', 50000)  # Default
    }
    
    env_name, timesteps = env_map.get(env_choice, ('CartPole-v1', 50000))
    
    # Train
    results_dir = quick_train(
        env_name=env_name,
        total_timesteps=timesteps,
        seed=42
    )
    
    # Visualize
    quick_visualize(results_dir)
    
    print("\n" + "="*60)
    print("ALL DONE!")
    print("="*60)
    print(f"\nYour results are in: {results_dir}")
    print("\nFiles created:")
    print("  • model.pt - Trained model weights")
    print("  • metrics.json - All training metrics")
    print("  • 01_learning_curves.png")
    print("  • 02_training_metrics.png")
    print("  • 03_clipping_behavior.png")
    print("  • 04_figure1_clipped_objective.png")
    print("  • ... and more!")
    print("\nYou can now use these figures in your tutorial!")


if __name__ == "__main__":
    main()