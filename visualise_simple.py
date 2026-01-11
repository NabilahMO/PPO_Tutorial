"""
Simple PPO Visualisations for Medium Article
=============================================

Generates exactly 3 publication-ready figures:
1. Learning Curves (reward + TIR over training)
2. PPO Clipping Explanation (positive/negative advantage)
3. 24-Hour Glucose Profile (glucose + insulin trace)

Usage:
    python visualise_simple.py --results-dir ./results/stability_test/
    
Or generate sample figures:
    python visualise_simple.py --demo
"""

import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d

# Style settings
plt.style.use('seaborn-v0_8-whitegrid')
FIGSIZE = (12, 6)
DPI = 300


def plot_learning_curves(results_dir: str, output_path: str = "learning_curves.png"):
    """
    Generate learning curves showing reward and TIR over training.
    
    Args:
        results_dir: Path to results directory containing metrics.json
        output_path: Where to save the figure
    """
    # Load metrics
    metrics_path = os.path.join(results_dir, "metrics.json")
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
    
    episode_rewards = np.array(metrics['episode_rewards'])
    episodes = np.arange(len(episode_rewards))
    
    # Smooth rewards
    window = min(20, len(episode_rewards) // 5) if len(episode_rewards) > 20 else 1
    rewards_smooth = uniform_filter1d(episode_rewards, size=window, mode='nearest')
    
    # Create figure
    fig, ax1 = plt.subplots(figsize=FIGSIZE)
    
    # Plot rewards (left axis)
    color_reward = '#2E86AB'
    ax1.plot(episodes, episode_rewards, alpha=0.2, color=color_reward)
    ax1.plot(episodes, rewards_smooth, color=color_reward, linewidth=2.5, label='Episode Reward')
    ax1.set_xlabel('Episode', fontsize=12)
    ax1.set_ylabel('Episode Reward', color=color_reward, fontsize=12)
    ax1.tick_params(axis='y', labelcolor=color_reward)
    
    # Plot TIR (right axis) if available
    if 'clinical_metrics' in metrics and len(metrics['clinical_metrics']) > 0:
        ax2 = ax1.twinx()
        color_tir = '#28A745'
        
        tir = np.array([c.get('time_in_range', 0) for c in metrics['clinical_metrics']])
        tir_smooth = uniform_filter1d(tir, size=window, mode='nearest') if len(tir) > window else tir
        
        ax2.plot(episodes[:len(tir)], tir, alpha=0.2, color=color_tir)
        ax2.plot(episodes[:len(tir)], tir_smooth, color=color_tir, linewidth=2.5, label='Time in Range (%)')
        
        ax2.set_ylabel('Time in Range (%)', color=color_tir, fontsize=12)
        ax2.tick_params(axis='y', labelcolor=color_tir)
        ax2.set_ylim([0, 105])
        
        # Combined legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='lower right', fontsize=11)
    else:
        ax1.legend(loc='lower right', fontsize=11)
    
    ax1.set_title('PPO Training Progress', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(output_path, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"✓ Saved: {output_path}")


def plot_clipping_objective(epsilon: float = 0.1, output_path: str = "clipping_objective.png"):
    """
    Generate PPO clipping explanation figure.
    
    Shows how clipping works for positive and negative advantages.
    
    Args:
        epsilon: Clipping parameter (default 0.1 for your conservative setting)
        output_path: Where to save the figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    r = np.linspace(0.0, 2.0, 1000)
    r_clipped = np.clip(r, 1 - epsilon, 1 + epsilon)
    
    # ===== LEFT: Positive Advantage =====
    ax1 = axes[0]
    A_pos = 1.0
    
    unclipped = r * A_pos
    clipped = r_clipped * A_pos
    ppo = np.minimum(unclipped, clipped)
    
    ax1.plot(r, unclipped, 'b--', linewidth=2, label='Unclipped: r·Â', alpha=0.7)
    ax1.plot(r, clipped, 'r--', linewidth=2, label='Clipped: clip(r)·Â', alpha=0.7)
    ax1.plot(r, ppo, 'g-', linewidth=3, label='PPO objective: min(...)')
    
    # Vertical lines at clip boundaries
    ax1.axvline(x=1-epsilon, color='gray', linestyle=':', alpha=0.6, linewidth=1.5)
    ax1.axvline(x=1+epsilon, color='gray', linestyle=':', alpha=0.6, linewidth=1.5)
    ax1.axvline(x=1.0, color='black', linestyle='-', alpha=0.3, linewidth=1)
    
    # Shading
    ax1.fill_between(r, 0, ppo, where=(ppo > 0), alpha=0.1, color='green')
    
    # Annotations
    ax1.annotate(f'1-ε\n({1-epsilon})', xy=(1-epsilon, -0.25), fontsize=10, ha='center')
    ax1.annotate(f'1+ε\n({1+epsilon})', xy=(1+epsilon, -0.25), fontsize=10, ha='center')
    ax1.annotate('Clipping caps\nthe benefit', xy=(1.5, 0.9), fontsize=10, ha='center',
                bbox=dict(boxstyle='round', facecolor='#ffffcc', alpha=0.8))
    
    ax1.set_xlabel('Probability Ratio r(θ) = π_new / π_old', fontsize=11)
    ax1.set_ylabel('Objective Value', fontsize=11)
    ax1.set_title('Positive Advantage (Â > 0)\n"Good action — want to do it MORE"', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=10)
    ax1.set_xlim([0, 2])
    ax1.set_ylim([-0.5, 2.2])
    ax1.grid(True, alpha=0.3)
    
    # ===== RIGHT: Negative Advantage =====
    ax2 = axes[1]
    A_neg = -1.0
    
    unclipped = r * A_neg
    clipped = r_clipped * A_neg
    ppo = np.minimum(unclipped, clipped)
    
    ax2.plot(r, unclipped, 'b--', linewidth=2, label='Unclipped: r·Â', alpha=0.7)
    ax2.plot(r, clipped, 'r--', linewidth=2, label='Clipped: clip(r)·Â', alpha=0.7)
    ax2.plot(r, ppo, 'g-', linewidth=3, label='PPO objective: min(...)')
    
    ax2.axvline(x=1-epsilon, color='gray', linestyle=':', alpha=0.6, linewidth=1.5)
    ax2.axvline(x=1+epsilon, color='gray', linestyle=':', alpha=0.6, linewidth=1.5)
    ax2.axvline(x=1.0, color='black', linestyle='-', alpha=0.3, linewidth=1)
    
    ax2.fill_between(r, ppo, 0, where=(ppo < 0), alpha=0.1, color='green')
    
    ax2.annotate(f'1-ε\n({1-epsilon})', xy=(1-epsilon, 0.25), fontsize=10, ha='center')
    ax2.annotate(f'1+ε\n({1+epsilon})', xy=(1+epsilon, 0.25), fontsize=10, ha='center')
    ax2.annotate('Clipping limits\nthe penalty', xy=(0.5, -0.9), fontsize=10, ha='center',
                bbox=dict(boxstyle='round', facecolor='#ffffcc', alpha=0.8))
    
    ax2.set_xlabel('Probability Ratio r(θ) = π_new / π_old', fontsize=11)
    ax2.set_ylabel('Objective Value', fontsize=11)
    ax2.set_title('Negative Advantage (Â < 0)\n"Bad action — want to do it LESS"', fontsize=12, fontweight='bold')
    ax2.legend(loc='lower left', fontsize=10)
    ax2.set_xlim([0, 2])
    ax2.set_ylim([-2.2, 0.5])
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle(f'PPO Clipped Objective (ε = {epsilon})', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(output_path, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"✓ Saved: {output_path}")


def plot_glucose_profile(
    glucose_trace: list = None,
    insulin_trace: list = None,
    results_dir: str = None,
    output_path: str = "glucose_profile.png"
):
    """
    Generate 24-hour glucose profile with insulin delivery.
    
    Args:
        glucose_trace: List of glucose values (or load from results_dir)
        insulin_trace: List of insulin values
        results_dir: Path to load traces from evaluation
        output_path: Where to save the figure
    """
    # Try to load from results if not provided
    if glucose_trace is None and results_dir is not None:
        eval_path = os.path.join(results_dir, "evaluation.json")
        if os.path.exists(eval_path):
            with open(eval_path, 'r') as f:
                eval_data = json.load(f)
            if 'episodes' in eval_data and len(eval_data['episodes']) > 0:
                ep = eval_data['episodes'][0]
                glucose_trace = ep.get('glucose_trace', [])
                insulin_trace = ep.get('insulin_trace', [])
    
    # Generate demo data if still no data
    if glucose_trace is None or len(glucose_trace) == 0:
        print("  Generating demo glucose profile...")
        np.random.seed(42)
        n_steps = 288  # 24 hours at 5-min intervals
        
        # Simulate realistic glucose with meals
        glucose_trace = []
        glucose = 120.0
        
        meal_times = [7*12, 12*12, 18*12]  # 7am, 12pm, 6pm in 5-min steps
        
        for t in range(n_steps):
            # Meal effects
            for meal_t in meal_times:
                if meal_t <= t < meal_t + 36:  # 3 hours effect
                    glucose += np.random.uniform(0.5, 2.0)
                elif meal_t + 36 <= t < meal_t + 72:
                    glucose -= np.random.uniform(0.3, 1.0)
            
            # Random drift
            glucose += np.random.normal(0, 1)
            glucose = np.clip(glucose, 70, 200)
            glucose_trace.append(glucose)
        
        # Simulated insulin
        insulin_trace = np.clip(np.random.uniform(0.5, 2.5, n_steps-1), 0, 5).tolist()
    
    # Convert to arrays
    glucose = np.array(glucose_trace)
    n_samples = len(glucose)
    time_hours = np.arange(n_samples) * 5 / 60  # 5-min intervals to hours
    
    # Create figure
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), height_ratios=[2, 1])
    
    # ===== TOP: Glucose =====
    ax1 = axes[0]
    
    # Background zones
    ax1.axhspan(70, 180, color='#90EE90', alpha=0.3, label='Target (70-180 mg/dL)')
    ax1.axhspan(0, 70, color='#FFB3B3', alpha=0.3, label='Hypoglycaemia (<70)')
    ax1.axhspan(180, 400, color='#FFE4B3', alpha=0.3, label='Hyperglycaemia (>180)')
    
    # Glucose line
    ax1.plot(time_hours, glucose, color='#2E86AB', linewidth=2.5, label='Blood Glucose')
    
    # Target lines
    ax1.axhline(y=70, color='#dc3545', linestyle='--', alpha=0.6, linewidth=1.5)
    ax1.axhline(y=180, color='#fd7e14', linestyle='--', alpha=0.6, linewidth=1.5)
    
    # Calculate metrics
    tir = np.mean((glucose >= 70) & (glucose <= 180)) * 100
    tbr = np.mean(glucose < 70) * 100
    tar = np.mean(glucose > 180) * 100
    
    # Stats box
    stats_text = f'TIR: {tir:.1f}%\nTBR: {tbr:.1f}%\nTAR: {tar:.1f}%'
    ax1.text(0.02, 0.97, stats_text, transform=ax1.transAxes, fontsize=11,
            verticalalignment='top', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'))
    
    ax1.set_ylabel('Blood Glucose (mg/dL)', fontsize=12)
    ax1.set_title('24-Hour Glucose Profile', fontsize=14, fontweight='bold')
    ax1.set_ylim([40, 300])
    ax1.set_xlim([0, 24])
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticklabels([])  # Hide x labels for top plot
    
    # ===== BOTTOM: Insulin =====
    ax2 = axes[1]
    
    if insulin_trace is not None and len(insulin_trace) > 0:
        insulin = np.array(insulin_trace)
        time_insulin = time_hours[1:len(insulin)+1] if len(insulin) < len(time_hours) else time_hours[:len(insulin)]
        
        ax2.fill_between(time_insulin, 0, insulin, color='#A23B72', alpha=0.4)
        ax2.plot(time_insulin, insulin, color='#A23B72', linewidth=2, label='Insulin Delivery')
        
        # Total insulin
        total_insulin = np.sum(insulin) * 5 / 60  # Convert to units
        ax2.text(0.02, 0.1, f'Total: {total_insulin:.1f} U', transform=ax2.transAxes,
                fontsize=11, ha='left', fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
        
        ax2.set_ylim([0, max(insulin) * 1.3 + 0.1])
        ax2.legend(loc='upper right', fontsize=10, framealpha=0.9)
    
    ax2.set_xlabel('Time (hours)', fontsize=12)
    ax2.set_ylabel('Insulin (U/hr)', fontsize=12)
    ax2.set_xlim([0, 24])
    ax2.grid(True, alpha=0.3)
    
    # X-axis labels for time of day
    ax2.set_xticks([0, 4, 8, 12, 16, 20, 24])
    ax2.set_xticklabels(['00:00', '04:00', '08:00', '12:00', '16:00', '20:00', '24:00'])
    
    plt.tight_layout()
    fig.savefig(output_path, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"✓ Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Generate PPO visualisations for Medium article')
    parser.add_argument('--results-dir', type=str, default=None,
                       help='Path to training results directory')
    parser.add_argument('--output-dir', type=str, default='./figures',
                       help='Where to save figures')
    parser.add_argument('--epsilon', type=float, default=0.1,
                       help='Clipping parameter for objective plot')
    parser.add_argument('--demo', action='store_true',
                       help='Generate demo figures without real data')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("\n" + "="*50)
    print("Generating PPO Visualisations for Medium Article")
    print("="*50 + "\n")
    
    # 1. Clipping Objective (always generate - doesn't need data)
    print("1. PPO Clipping Objective...")
    plot_clipping_objective(
        epsilon=args.epsilon,
        output_path=os.path.join(args.output_dir, "clipping_objective.png")
    )
    
    # 2. Learning Curves
    print("\n2. Learning Curves...")
    if args.results_dir and os.path.exists(os.path.join(args.results_dir, "metrics.json")):
        plot_learning_curves(
            results_dir=args.results_dir,
            output_path=os.path.join(args.output_dir, "learning_curves.png")
        )
    elif args.demo:
        # Generate demo learning curves
        print("  Generating demo learning curves...")
        np.random.seed(42)
        n_episodes = 100
        
        # Simulated improving rewards
        base = np.linspace(-50, 40, n_episodes)
        noise = np.random.normal(0, 10, n_episodes)
        demo_rewards = base + noise
        
        # Simulated TIR improvement
        tir_base = np.linspace(30, 95, n_episodes)
        tir_noise = np.random.normal(0, 5, n_episodes)
        demo_tir = np.clip(tir_base + tir_noise, 0, 100)
        
        demo_metrics = {
            'episode_rewards': demo_rewards.tolist(),
            'clinical_metrics': [{'time_in_range': t} for t in demo_tir]
        }
        
        # Save temp metrics
        temp_dir = os.path.join(args.output_dir, 'temp')
        os.makedirs(temp_dir, exist_ok=True)
        with open(os.path.join(temp_dir, 'metrics.json'), 'w') as f:
            json.dump(demo_metrics, f)
        
        plot_learning_curves(
            results_dir=temp_dir,
            output_path=os.path.join(args.output_dir, "learning_curves.png")
        )
        
        # Cleanup
        import shutil
        shutil.rmtree(temp_dir)
    else:
        print("  ⚠ No results directory provided. Use --demo for sample figures.")
    
    # 3. Glucose Profile
    print("\n3. 24-Hour Glucose Profile...")
    plot_glucose_profile(
        results_dir=args.results_dir,
        output_path=os.path.join(args.output_dir, "glucose_profile.png")
    )
    
    print("\n" + "="*50)
    print(f"✓ All figures saved to: {args.output_dir}")
    print("="*50 + "\n")
    
    print("Files generated:")
    print("  1. clipping_objective.png  → Section 2 (Clipped Surrogate Objective)")
    print("  2. learning_curves.png     → Section 6 (Training the Agent)")
    print("  3. glucose_profile.png     → Section 6 (Training the Agent)")


if __name__ == "__main__":
    main()