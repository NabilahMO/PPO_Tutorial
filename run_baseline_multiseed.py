"""
Run Baseline Comparison with Multiple Seeds
============================================

Trains PPO agents across multiple seeds and compares against baselines.

Usage:
    python run_baseline_multiseed.py --timesteps 100000 --seeds 42,123,456,789,101
"""

import os
import json
import argparse
import numpy as np
from datetime import datetime

from environment import GlucoseInsulinEnv
from ppo_agent import PPOAgent
from train import PPOTrainer
from evaluate import (
    evaluate_ppo_agent,
    evaluate_controller,
    FixedBasalController,
    BasalBolusController,
    RuleBasedController,
    PIDController
)


def run_baseline_comparison_multiseed(
    timesteps: int = 100000,
    seeds: list = [42, 123, 456, 789, 101],
    save_dir: str = None
):
    """
    Run baseline comparison with multiple seeds for PPO.
    """
    if save_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = f"./experiments/baseline_multiseed_{timestamp}"
    
    os.makedirs(save_dir, exist_ok=True)
    
    print("\n" + "=" * 70)
    print("BASELINE COMPARISON (MULTI-SEED)")
    print("=" * 70)
    print(f"Seeds: {seeds}")
    print(f"Timesteps: {timesteps}")
    print(f"Save dir: {save_dir}")
    print("=" * 70)
    
    # Default configs
    env_config = {
        'max_insulin_dose': 5.0,
        'episode_length_hours': 24.0,
        'sample_time_minutes': 5.0,
        'target_glucose_min': 70.0,
        'target_glucose_max': 180.0,
        'patient_variability': True,
        'meal_variability': True
    }
    
    # =========================================================================
    # 1. Train and evaluate PPO across all seeds
    # =========================================================================
    print("\n" + "-" * 50)
    print("Training PPO agents across seeds...")
    print("-" * 50)
    
    ppo_results = []
    
    for seed in seeds:
        print(f"\n>>> Seed {seed}")
        
        # Train
        trainer = PPOTrainer(
            total_timesteps=timesteps,
            seed=seed,
            experiment_name=f"baseline_ppo_seed{seed}",
            save_dir=save_dir
        )
        
        result = trainer.train()
        
        # Get final evaluation metrics
        final_eval = result.get('final_eval', {})
        ppo_results.append({
            'seed': seed,
            'reward': final_eval.get('mean_reward', 0),
            'tir': final_eval.get('mean_tir', 0),
            'tbr': final_eval.get('mean_tbr', 0),
            'tar': final_eval.get('mean_tar', 0)
        })
        
        print(f"    TIR: {final_eval.get('mean_tir', 0):.1f}%")
    
    # Aggregate PPO results
    ppo_mean_reward = np.mean([r['reward'] for r in ppo_results])
    ppo_std_reward = np.std([r['reward'] for r in ppo_results])
    ppo_mean_tir = np.mean([r['tir'] for r in ppo_results])
    ppo_std_tir = np.std([r['tir'] for r in ppo_results])
    ppo_mean_tbr = np.mean([r['tbr'] for r in ppo_results])
    ppo_std_tbr = np.std([r['tbr'] for r in ppo_results])
    
    # =========================================================================
    # 2. Evaluate baseline controllers (same seeds for fair comparison)
    # =========================================================================
    print("\n" + "-" * 50)
    print("Evaluating baseline controllers...")
    print("-" * 50)
    
    baselines = [
        FixedBasalController(basal_rate=1.0),
        BasalBolusController(basal_rate=0.8, insulin_to_carb_ratio=10.0),
        RuleBasedController(basal_rate=1.0, target_glucose=120.0),
        PIDController(basal_rate=1.0, target_glucose=120.0)
    ]
    
    baseline_results = {}
    
    for controller in baselines:
        print(f"\n>>> {controller.name}")
        
        controller_metrics = []
        
        for seed in seeds:
            result = evaluate_controller(
                controller=controller,
                env_config=env_config,
                num_episodes=10,
                seed=seed
            )
            
            summary = result.summary()
            controller_metrics.append({
                'seed': seed,
                'reward': summary['mean_reward'],
                'tir': summary['mean_tir'],
                'tbr': summary['mean_tbr'],
                'tar': summary['mean_tar']
            })
        
        # Aggregate
        baseline_results[controller.name] = {
            'mean_reward': np.mean([m['reward'] for m in controller_metrics]),
            'std_reward': np.std([m['reward'] for m in controller_metrics]),
            'mean_tir': np.mean([m['tir'] for m in controller_metrics]),
            'std_tir': np.std([m['tir'] for m in controller_metrics]),
            'mean_tbr': np.mean([m['tbr'] for m in controller_metrics]),
            'std_tbr': np.std([m['tbr'] for m in controller_metrics]),
            'per_seed': controller_metrics
        }
        
        print(f"    TIR: {baseline_results[controller.name]['mean_tir']:.1f}% ± {baseline_results[controller.name]['std_tir']:.1f}%")
    
    # =========================================================================
    # 3. Compile final comparison
    # =========================================================================
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print(f"{'Controller':<30} {'Reward':>12} {'TIR (%)':>15} {'TBR (%)':>15}")
    print("-" * 80)
    
    # PPO first
    print(f"{'PPO Agent':<30} {ppo_mean_reward:>6.1f} ± {ppo_std_reward:<4.1f} {ppo_mean_tir:>6.1f} ± {ppo_std_tir:<4.1f}% {ppo_mean_tbr:>6.1f} ± {ppo_std_tbr:<4.1f}%")
    
    # Baselines
    for name, data in baseline_results.items():
        print(f"{name:<30} {data['mean_reward']:>6.1f} ± {data['std_reward']:<4.1f} {data['mean_tir']:>6.1f} ± {data['std_tir']:<4.1f}% {data['mean_tbr']:>6.1f} ± {data['std_tbr']:<4.1f}%")
    
    print("=" * 80)
    
    # =========================================================================
    # 4. Save results
    # =========================================================================
    all_results = {
        'ppo': {
            'mean_reward': ppo_mean_reward,
            'std_reward': ppo_std_reward,
            'mean_tir': ppo_mean_tir,
            'std_tir': ppo_std_tir,
            'mean_tbr': ppo_mean_tbr,
            'std_tbr': ppo_std_tbr,
            'per_seed': ppo_results
        },
        'baselines': baseline_results,
        'config': {
            'timesteps': timesteps,
            'seeds': seeds
        }
    }
    
    results_path = os.path.join(save_dir, "baseline_multiseed_results.json")
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nResults saved to: {results_path}")
    
    # =========================================================================
    # 5. Generate comparison figure
    # =========================================================================
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Prepare data
        names = ['PPO'] + list(baseline_results.keys())
        tir_means = [ppo_mean_tir] + [baseline_results[n]['mean_tir'] for n in baseline_results]
        tir_stds = [ppo_std_tir] + [baseline_results[n]['std_tir'] for n in baseline_results]
        tbr_means = [ppo_mean_tbr] + [baseline_results[n]['mean_tbr'] for n in baseline_results]
        tbr_stds = [ppo_std_tbr] + [baseline_results[n]['std_tbr'] for n in baseline_results]
        
        colors = ['#28A745'] + ['#6c757d'] * len(baseline_results)
        
        x = np.arange(len(names))
        
        # TIR plot
        ax1 = axes[0]
        bars1 = ax1.bar(x, tir_means, yerr=tir_stds, color=colors, 
                        edgecolor='black', alpha=0.85, capsize=5)
        ax1.axhline(y=70, color='red', linestyle='--', linewidth=2, label='Target (>70%)')
        ax1.set_xticks(x)
        ax1.set_xticklabels(names, rotation=20, ha='right')
        ax1.set_ylabel('Time in Range (%)')
        ax1.set_title('Efficacy: Time in Range (70-180 mg/dL)')
        ax1.set_ylim([0, 110])
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, mean, std in zip(bars1, tir_means, tir_stds):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 2,
                    f'{mean:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # TBR plot
        ax2 = axes[1]
        bars2 = ax2.bar(x, tbr_means, yerr=tbr_stds, color=colors,
                        edgecolor='black', alpha=0.85, capsize=5)
        ax2.axhline(y=4, color='red', linestyle='--', linewidth=2, label='Target (<4%)')
        ax2.set_xticks(x)
        ax2.set_xticklabels(names, rotation=20, ha='right')
        ax2.set_ylabel('Time Below Range (%)')
        ax2.set_title('Safety: Hypoglycaemia (<70 mg/dL)')
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, mean, std in zip(bars2, tbr_means, tbr_stds):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.5,
                    f'{mean:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        plt.suptitle('PPO vs Baseline Controllers (5 Seeds)', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        fig_path = os.path.join(save_dir, "baseline_multiseed_comparison.png")
        plt.savefig(fig_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"Figure saved to: {fig_path}")
        
    except Exception as e:
        print(f"Could not generate figure: {e}")
    
    return all_results


def main():
    parser = argparse.ArgumentParser(description='Run baseline comparison with multiple seeds')
    parser.add_argument('--timesteps', type=int, default=100000,
                       help='Training timesteps per seed')
    parser.add_argument('--seeds', type=str, default='42,123,456,789,101',
                       help='Comma-separated random seeds')
    parser.add_argument('--save-dir', type=str, default=None,
                       help='Directory to save results')
    
    args = parser.parse_args()
    
    seeds = [int(s) for s in args.seeds.split(',')]
    
    run_baseline_comparison_multiseed(
        timesteps=args.timesteps,
        seeds=seeds,
        save_dir=args.save_dir
    )


if __name__ == "__main__":
    main()
