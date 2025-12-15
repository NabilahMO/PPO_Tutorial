"""
Main Entry Point for PPO Glucose Control
=========================================

Provides a unified command-line interface for:
- Training PPO agents
- Evaluating trained agents
- Running experiments
- Generating visualisations

Usage examples:
    python run.py train --timesteps 100000
    python run.py evaluate --model results/model.pt
    python run.py experiment --type epsilon
    python run.py visualise --results-dir results/
"""

import os
import argparse
import sys
from datetime import datetime


def train_command(args):
    """Run training."""
    from train import PPOTrainer
    
    print("\n" + "=" * 60)
    print("PPO GLUCOSE CONTROL - TRAINING")
    print("=" * 60)
    
    trainer = PPOTrainer(
        total_timesteps=args.timesteps,
        steps_per_update=args.steps_per_update,
        eval_frequency=args.eval_frequency,
        seed=args.seed,
        experiment_name=args.name,
        save_dir=args.save_dir
    )
    
    results = trainer.train()
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Results saved to: {results['save_dir']}")
    
    return results


def evaluate_command(args):
    """Run evaluation."""
    from evaluate import run_baseline_comparison, load_and_evaluate_agent
    
    print("\n" + "=" * 60)
    print("PPO GLUCOSE CONTROL - EVALUATION")
    print("=" * 60)
    
    if args.model:
        # Evaluate specific model
        from evaluate import evaluate_ppo_agent
        from ppo_agent import PPOAgent
        from environment import GlucoseInsulinEnv
        
        env_config = {
            'max_insulin_dose': 5.0,
            'episode_length_hours': 24.0,
            'sample_time_minutes': 5.0,
            'patient_variability': True,
            'meal_variability': True
        }
        
        result = load_and_evaluate_agent(
            model_path=args.model,
            env_config=env_config,
            num_episodes=args.episodes,
            seed=args.seed
        )
        
        print("\nEvaluation Results:")
        print(f"  Mean reward: {result.mean_reward:.1f} ± {result.std_reward:.1f}")
        print(f"  Time in Range: {result.mean_tir:.1f}%")
        print(f"  Time Below Range: {result.mean_tbr:.1f}%")
        print(f"  Time Above Range: {result.mean_tar:.1f}%")
    else:
        # Run baseline comparison
        results = run_baseline_comparison(
            model_path=args.model,
            num_episodes=args.episodes,
            seed=args.seed,
            save_dir=args.save_dir
        )
    
    return results


def experiment_command(args):
    """Run experiments."""
    from experiments import PPOExperiment
    
    print("\n" + "=" * 60)
    print(f"PPO GLUCOSE CONTROL - {args.type.upper()} EXPERIMENT")
    print("=" * 60)
    
    experiment = PPOExperiment(
        base_save_dir=args.save_dir,
        default_timesteps=args.timesteps
    )
    
    seeds = [int(s) for s in args.seeds.split(',')]
    
    if args.type == 'epsilon':
        epsilons = [float(e) for e in args.values.split(',')]
        results = experiment.run_epsilon_experiment(
            epsilon_values=epsilons,
            seeds=seeds
        )
    elif args.type == 'reward':
        results = experiment.run_reward_weight_experiment(
            seeds=seeds
        )
    elif args.type == 'baseline':
        results = experiment.run_baseline_comparison(
            model_path=args.model,
            train_if_no_model=True
        )
    else:
        print(f"Unknown experiment type: {args.type}")
        return None
    
    return results


def visualise_command(args):
    """Generate visualisations."""
    from visualise import generate_all_visualisations, PPOVisualiser
    
    print("\n" + "=" * 60)
    print("PPO GLUCOSE CONTROL - VISUALISATION")
    print("=" * 60)
    
    if args.results_dir:
        # Generate from results directory
        paths = generate_all_visualisations(
            results_dir=args.results_dir,
            output_dir=args.output_dir
        )
        print(f"\nGenerated {len(paths)} figures")
    else:
        # Generate example/educational figures
        vis = PPOVisualiser(save_dir=args.output_dir or './figures')
        
        # Generate clipped objective explanation
        vis.plot_clipped_objective_explanation(
            epsilon=args.epsilon,
            title="PPO Clipped Objective (Educational)",
            filename="ppo_clipped_objective_explanation.png"
        )
        
        print("\nGenerated educational visualisation")
    
    return None


def quick_start_command(args):
    """Quick start: train, evaluate, and visualise."""
    print("\n" + "=" * 60)
    print("PPO GLUCOSE CONTROL - QUICK START")
    print("=" * 60)
    print("This will train a PPO agent and generate all visualisations.")
    print(f"Timesteps: {args.timesteps:,}")
    print(f"Seed: {args.seed}")
    print("=" * 60)
    
    # Train
    from train import PPOTrainer
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"quickstart_{timestamp}"
    
    trainer = PPOTrainer(
        total_timesteps=args.timesteps,
        seed=args.seed,
        experiment_name=experiment_name
    )
    
    results = trainer.train()
    
    # Generate visualisations
    from visualise import generate_all_visualisations
    
    generate_all_visualisations(results['save_dir'])
    
    print("\n" + "=" * 60)
    print("QUICK START COMPLETE!")
    print("=" * 60)
    print(f"\nResults: {results['save_dir']}")
    print(f"Final reward: {results['final_mean_reward']:.1f}")
    print(f"Time in Range: {results['final_eval']['mean_tir']:.1f}%")
    
    return results


def demo_command(args):
    """Run a quick demo of the environment."""
    from environment import GlucoseInsulinEnv
    import numpy as np
    
    print("\n" + "=" * 60)
    print("PPO GLUCOSE CONTROL - ENVIRONMENT DEMO")
    print("=" * 60)
    
    env = GlucoseInsulinEnv(
        patient_variability=True,
        meal_variability=True
    )
    
    state, info = env.reset(seed=args.seed)
    
    print(f"\nPatient:")
    print(f"  Weight: {info['patient_weight']:.1f} kg")
    print(f"  Insulin sensitivity: {info['patient_insulin_sensitivity']:.2f}")
    
    print(f"\nMeal schedule:")
    for meal in info['meal_schedule']:
        print(f"  {meal['time']:.1f}h: {meal['cho']:.0f}g CHO")
    
    print(f"\nRunning 24-hour simulation with fixed basal insulin...")
    print("-" * 60)
    
    total_reward = 0
    done = False
    step = 0
    
    while not done:
        # Fixed basal rate
        action = np.array([1.0])
        
        state, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated
        
        # Print every 2 hours
        if step % 24 == 0:
            env.render()
        
        step += 1
    
    print("-" * 60)
    
    stats = env.get_episode_stats()
    print(f"\nEpisode Statistics:")
    print(f"  Total reward: {stats['total_reward']:.1f}")
    print(f"  Time in Range: {stats['time_in_range']:.1f}%")
    print(f"  Time Below Range: {stats['time_below_range']:.1f}%")
    print(f"  Time Above Range: {stats['time_above_range']:.1f}%")
    print(f"  Mean glucose: {stats['mean_glucose']:.1f} mg/dL")
    print(f"  Glucose CV: {stats['glucose_cv']:.1f}%")
    print(f"  Total insulin: {stats['total_insulin']:.1f} U")
    
    print("\n✓ Demo complete!")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="PPO for Glucose Control in Type 1 Diabetes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Quick start (train + visualise):
    python run.py quickstart --timesteps 50000

  Train a new agent:
    python run.py train --timesteps 100000 --seed 42

  Evaluate a trained agent:
    python run.py evaluate --model results/models/final_model.pt

  Run epsilon comparison experiment:
    python run.py experiment --type epsilon --values 0.1,0.2,0.3

  Generate visualisations:
    python run.py visualise --results-dir results/experiment_name/

  Run environment demo:
    python run.py demo
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # ---- Quick Start ----
    quickstart_parser = subparsers.add_parser(
        'quickstart', help='Quick start: train and visualise'
    )
    quickstart_parser.add_argument(
        '--timesteps', type=int, default=50000,
        help='Training timesteps'
    )
    quickstart_parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed'
    )
    
    # ---- Train ----
    train_parser = subparsers.add_parser('train', help='Train a PPO agent')
    train_parser.add_argument(
        '--timesteps', type=int, default=100000,
        help='Total training timesteps'
    )
    train_parser.add_argument(
        '--steps-per-update', type=int, default=2048,
        help='Steps to collect before each PPO update'
    )
    train_parser.add_argument(
        '--eval-frequency', type=int, default=10,
        help='Evaluate every N updates'
    )
    train_parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed'
    )
    train_parser.add_argument(
        '--name', type=str, default=None,
        help='Experiment name'
    )
    train_parser.add_argument(
        '--save-dir', type=str, default='./results',
        help='Directory to save results'
    )
    
    # ---- Evaluate ----
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate agents')
    eval_parser.add_argument(
        '--model', type=str, default=None,
        help='Path to trained model'
    )
    eval_parser.add_argument(
        '--episodes', type=int, default=10,
        help='Number of evaluation episodes'
    )
    eval_parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed'
    )
    eval_parser.add_argument(
        '--save-dir', type=str, default='./results/evaluation',
        help='Directory to save results'
    )
    
    # ---- Experiment ----
    exp_parser = subparsers.add_parser('experiment', help='Run experiments')
    exp_parser.add_argument(
        '--type', type=str, required=True,
        choices=['epsilon', 'reward', 'baseline'],
        help='Type of experiment'
    )
    exp_parser.add_argument(
        '--timesteps', type=int, default=50000,
        help='Training timesteps per run'
    )
    exp_parser.add_argument(
        '--seeds', type=str, default='42,123',
        help='Comma-separated random seeds'
    )
    exp_parser.add_argument(
        '--values', type=str, default='0.1,0.2,0.3',
        help='Comma-separated values to test (for epsilon experiment)'
    )
    exp_parser.add_argument(
        '--model', type=str, default=None,
        help='Path to pre-trained model (for baseline comparison)'
    )
    exp_parser.add_argument(
        '--save-dir', type=str, default='./experiments',
        help='Directory to save results'
    )
    
    # ---- Visualise ----
    vis_parser = subparsers.add_parser('visualise', help='Generate visualisations')
    vis_parser.add_argument(
        '--results-dir', type=str, default=None,
        help='Directory containing training results'
    )
    vis_parser.add_argument(
        '--output-dir', type=str, default=None,
        help='Directory to save figures'
    )
    vis_parser.add_argument(
        '--epsilon', type=float, default=0.2,
        help='Epsilon value for educational plot'
    )
    
    # ---- Demo ----
    demo_parser = subparsers.add_parser('demo', help='Run environment demo')
    demo_parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed'
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        sys.exit(1)
    
    # Run command
    if args.command == 'quickstart':
        quick_start_command(args)
    elif args.command == 'train':
        train_command(args)
    elif args.command == 'evaluate':
        evaluate_command(args)
    elif args.command == 'experiment':
        experiment_command(args)
    elif args.command == 'visualise':
        visualise_command(args)
    elif args.command == 'demo':
        demo_command(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()