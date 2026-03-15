"""
Generate Publication Figures for PPO Glucose Control
=====================================================

Generates figures for MICCAI submission:
1. Model Performance (5 seeds learning curves + summary)
2. Baseline Comparison (PPO vs PID vs Fixed vs Rule-based)
3. Epsilon Comparison (0.1 vs 0.2 vs 0.3)
4. Reward Weights Comparison

Usage:
    python generate_figures.py --results-dir ./results --experiments-dir ./experiments --output-dir ./figures
"""

import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d
from glob import glob

# Style settings
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 14

DPI = 300
COLORS = {
    'primary': '#2E86AB',
    'secondary': '#A23B72', 
    'success': '#28A745',
    'warning': '#F18F01',
    'danger': '#C73E1D',
    'purple': '#7B2CBF',
    'teal': '#2A9D8F'
}


def load_json(filepath):
    """Load JSON file safely."""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"  Warning: Could not load {filepath}: {e}")
        return None


def find_results_dirs(base_dir, pattern="perf_seed*"):
    """Find all matching result directories."""
    dirs = glob(os.path.join(base_dir, pattern))
    return sorted(dirs)


# =============================================================================
# FIGURE 1: Model Performance (5 Seeds)
# =============================================================================
def plot_model_performance(results_dir, output_dir):
    """
    Generate model performance figure showing:
    - Learning curves for all 5 seeds
    - Final performance summary bar chart
    """
    print("\n1. Generating Model Performance Figure...")
    
    # Find all seed directories
    seed_dirs = find_results_dirs(results_dir, "perf_seed*")
    
    if not seed_dirs:
        print("  ⚠ No perf_seed* directories found. Skipping.")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # ===== LEFT: Learning Curves =====
    ax1 = axes[0]
    
    all_rewards = []
    all_tir = []
    seed_labels = []
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(seed_dirs)))
    
    for seed_dir, color in zip(seed_dirs, colors):
        seed_name = os.path.basename(seed_dir)
        seed_num = seed_name.replace('perf_seed', '')
        seed_labels.append(seed_num)
        
        metrics = load_json(os.path.join(seed_dir, "metrics.json"))
        if metrics is None:
            continue
        
        rewards = np.array(metrics.get('episode_rewards', []))
        all_rewards.append(rewards)
        
        # Smooth and plot
        if len(rewards) > 10:
            window = min(20, len(rewards) // 5)
            rewards_smooth = uniform_filter1d(rewards, size=window, mode='nearest')
        else:
            rewards_smooth = rewards
        
        episodes = np.arange(len(rewards_smooth))
        ax1.plot(episodes, rewards_smooth, color=color, linewidth=2, 
                label=f'Seed {seed_num}', alpha=0.8)
        
        # Get TIR
        clinical = metrics.get('clinical_metrics', [])
        if clinical:
            final_tir = clinical[-1].get('time_in_range', 0)
            all_tir.append(final_tir)
    
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Episode Reward')
    ax1.set_title('Learning Curves Across Seeds')
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)
    
    # ===== RIGHT: Final Performance Summary =====
    ax2 = axes[1]
    
    if all_tir:
        x = np.arange(len(seed_labels))
        bars = ax2.bar(x, all_tir, color=COLORS['success'], alpha=0.8, edgecolor='black')
        
        # Add value labels
        for bar, val in zip(bars, all_tir):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{val:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # Target line
        ax2.axhline(y=70, color=COLORS['danger'], linestyle='--', 
                   linewidth=2, label='Clinical Target (70%)')
        
        ax2.set_xticks(x)
        ax2.set_xticklabels([f'Seed {s}' for s in seed_labels])
        ax2.set_ylabel('Time in Range (%)')
        ax2.set_title('Final Performance by Seed')
        ax2.set_ylim([0, 110])
        ax2.legend(loc='lower right')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Summary stats
        mean_tir = np.mean(all_tir)
        std_tir = np.std(all_tir)
        ax2.text(0.98, 0.02, f'Mean: {mean_tir:.1f}% ± {std_tir:.1f}%',
                transform=ax2.transAxes, ha='right', va='bottom',
                fontsize=11, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    plt.suptitle('PPO Model Performance (5 Random Seeds)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'figure1_model_performance.png')
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✓ Saved: {output_path}")


# =============================================================================
# FIGURE 2: Baseline Comparison
# =============================================================================
def plot_baseline_comparison(experiments_dir, output_dir):
    """
    Generate baseline comparison figure showing:
    - TIR comparison (PPO vs baselines)
    - Safety comparison (TBR)
    """
    print("\n2. Generating Baseline Comparison Figure...")
    
    # Find baseline comparison results
    baseline_dirs = glob(os.path.join(experiments_dir, "baseline_comparison*"))
    
    if not baseline_dirs:
        print("  ⚠ No baseline_comparison* directories found. Skipping.")
        return
    
    # Use most recent
    baseline_dir = sorted(baseline_dirs)[-1]
    
    # Try to load comparison.json or results from the directory
    comparison_path = os.path.join(baseline_dir, "comparison.json")
    results_data = load_json(comparison_path)
    
    if results_data is None:
        # Try alternate paths
        for filename in ['baseline_comparison.json', 'results.json']:
            alt_path = os.path.join(baseline_dir, filename)
            results_data = load_json(alt_path)
            if results_data:
                break
    
    if results_data is None:
        print(f"  ⚠ No results found in {baseline_dir}. Skipping.")
        return
    
    # Extract data
    if 'summaries' in results_data:
        summaries = results_data['summaries']
    elif isinstance(results_data, list):
        summaries = results_data
    else:
        summaries = results_data.get('results', [])
    
    if not summaries:
        print("  ⚠ No summary data found. Skipping.")
        return
    
    names = [s.get('name', f'Method {i}') for i, s in enumerate(summaries)]
    tir_values = [s.get('mean_tir', 0) for s in summaries]
    tbr_values = [s.get('mean_tbr', 0) for s in summaries]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Colors: PPO in green, others in progressively lighter blues
    n = len(names)
    colors = [COLORS['success']] + [plt.cm.Blues(0.3 + 0.5*i/(n-1)) for i in range(n-1)]
    
    # ===== LEFT: Time in Range =====
    ax1 = axes[0]
    x = np.arange(len(names))
    bars1 = ax1.bar(x, tir_values, color=colors, edgecolor='black', alpha=0.85)
    
    ax1.axhline(y=70, color=COLORS['danger'], linestyle='--', linewidth=2, label='Target (>70%)')
    
    for bar, val in zip(bars1, tir_values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=20, ha='right')
    ax1.set_ylabel('Time in Range (%)')
    ax1.set_title('Efficacy: Time in Range (70-180 mg/dL)')
    ax1.set_ylim([0, 110])
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # ===== RIGHT: Time Below Range (Safety) =====
    ax2 = axes[1]
    bars2 = ax2.bar(x, tbr_values, color=colors, edgecolor='black', alpha=0.85)
    
    ax2.axhline(y=4, color=COLORS['danger'], linestyle='--', linewidth=2, label='Target (<4%)')
    
    for bar, val in zip(bars2, tbr_values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=20, ha='right')
    ax2.set_ylabel('Time Below Range (%)')
    ax2.set_title('Safety: Hypoglycaemia (<70 mg/dL)')
    ax2.set_ylim([0, max(tbr_values) * 1.3 + 1 if tbr_values else 10])
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('PPO vs Baseline Controllers', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'figure2_baseline_comparison.png')
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✓ Saved: {output_path}")


# =============================================================================
# FIGURE 3: Epsilon Comparison
# =============================================================================
def plot_epsilon_comparison(experiments_dir, output_dir):
    """
    Generate epsilon comparison figure showing:
    - Learning curves for each epsilon
    - Final TIR comparison
    - KL divergence stability
    """
    print("\n3. Generating Epsilon Comparison Figure...")
    
    # Find epsilon experiment results
    epsilon_dirs = glob(os.path.join(experiments_dir, "epsilon_experiment*"))
    
    if not epsilon_dirs:
        print("  ⚠ No epsilon_experiment* directories found. Skipping.")
        return
    
    # Use most recent
    epsilon_dir = sorted(epsilon_dirs)[-1]
    
    # Load results
    results_path = os.path.join(epsilon_dir, "epsilon_comparison.json")
    results_data = load_json(results_path)
    
    if results_data is None:
        print(f"  ⚠ No epsilon_comparison.json found in {epsilon_dir}. Skipping.")
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    epsilons = sorted([float(k) for k in results_data.keys()])
    colors = [COLORS['success'], COLORS['warning'], COLORS['danger']]
    
    # ===== LEFT: Learning Curves =====
    ax1 = axes[0]
    
    for eps, color in zip(epsilons, colors):
        data = results_data[str(eps)]
        rewards = np.array(data.get('episode_rewards', []))
        
        if len(rewards) > 10:
            window = min(20, len(rewards) // 5)
            rewards_smooth = uniform_filter1d(rewards, size=window, mode='nearest')
        else:
            rewards_smooth = rewards
        
        episodes = np.arange(len(rewards_smooth))
        ax1.plot(episodes, rewards_smooth, color=color, linewidth=2.5,
                label=f'ε = {eps}', alpha=0.9)
    
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Episode Reward')
    ax1.set_title('Learning Curves')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # ===== MIDDLE: Final TIR Comparison =====
    ax2 = axes[1]
    
    tir_values = []
    tir_stds = []
    for eps in epsilons:
        data = results_data[str(eps)]
        final_eval = data.get('final_eval', {})
        tir_values.append(final_eval.get('mean_tir', 0))
        tir_stds.append(final_eval.get('std_tir', 0))
    
    x = np.arange(len(epsilons))
    bars = ax2.bar(x, tir_values, yerr=tir_stds, color=colors, 
                   edgecolor='black', alpha=0.85, capsize=5)
    
    ax2.axhline(y=70, color='black', linestyle='--', linewidth=1.5, label='Target (70%)')
    
    for bar, val, std in zip(bars, tir_values, tir_stds):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 1,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'ε = {e}' for e in epsilons])
    ax2.set_ylabel('Time in Range (%)')
    ax2.set_title('Final Performance')
    ax2.set_ylim([0, 110])
    ax2.legend(loc='lower right')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # ===== RIGHT: KL Divergence (Stability) =====
    ax3 = axes[2]
    
    for eps, color in zip(epsilons, colors):
        data = results_data[str(eps)]
        agent_metrics = data.get('agent_metrics', {})
        kl = np.array(agent_metrics.get('kl_divergence', []))
        
        if len(kl) > 0:
            updates = np.arange(len(kl))
            ax3.plot(updates, kl, color=color, linewidth=2, label=f'ε = {eps}', alpha=0.8)
    
    ax3.set_xlabel('Update')
    ax3.set_ylabel('KL Divergence')
    ax3.set_title('Training Stability')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.suptitle('Effect of Clipping Parameter ε', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'figure3_epsilon_comparison.png')
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✓ Saved: {output_path}")


# =============================================================================
# FIGURE 4: Reward Weights Comparison
# =============================================================================
def plot_reward_comparison(experiments_dir, output_dir):
    """
    Generate reward weights comparison figure.
    """
    print("\n4. Generating Reward Weights Comparison Figure...")
    
    # Find reward experiment results
    reward_dirs = glob(os.path.join(experiments_dir, "reward_weight*"))
    
    if not reward_dirs:
        print("  ⚠ No reward_weight* directories found. Skipping.")
        return
    
    # Use most recent
    reward_dir = sorted(reward_dirs)[-1]
    
    # Load results
    results_path = os.path.join(reward_dir, "reward_weight_comparison.json")
    results_data = load_json(results_path)
    
    if results_data is None:
        print(f"  ⚠ No reward_weight_comparison.json found in {reward_dir}. Skipping.")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    config_names = list(results_data.keys())
    colors = plt.cm.Set2(np.linspace(0, 0.8, len(config_names)))
    
    # ===== LEFT: Learning Curves =====
    ax1 = axes[0]
    
    for name, color in zip(config_names, colors):
        data = results_data[name]
        rewards = np.array(data.get('episode_rewards', []))
        
        if len(rewards) > 10:
            window = min(20, len(rewards) // 5)
            rewards_smooth = uniform_filter1d(rewards, size=window, mode='nearest')
        else:
            rewards_smooth = rewards
        
        episodes = np.arange(len(rewards_smooth))
        ax1.plot(episodes, rewards_smooth, color=color, linewidth=2, label=name, alpha=0.8)
    
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Episode Reward')
    ax1.set_title('Learning Curves')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # ===== RIGHT: TIR vs TBR Trade-off =====
    ax2 = axes[1]
    
    tir_values = []
    tbr_values = []
    for name in config_names:
        data = results_data[name]
        final_eval = data.get('final_eval', {})
        tir_values.append(final_eval.get('mean_tir', 0))
        tbr_values.append(final_eval.get('mean_tbr', 0))
    
    x = np.arange(len(config_names))
    width = 0.35
    
    bars1 = ax2.bar(x - width/2, tir_values, width, label='TIR (%)', 
                    color=COLORS['success'], edgecolor='black', alpha=0.85)
    bars2 = ax2.bar(x + width/2, tbr_values, width, label='TBR (%)', 
                    color=COLORS['danger'], edgecolor='black', alpha=0.85)
    
    # Add value labels
    for bar, val in zip(bars1, tir_values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val:.1f}', ha='center', va='bottom', fontsize=9)
    for bar, val in zip(bars2, tbr_values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                f'{val:.1f}', ha='center', va='bottom', fontsize=9)
    
    ax2.set_xticks(x)
    ax2.set_xticklabels(config_names, rotation=20, ha='right')
    ax2.set_ylabel('Percentage (%)')
    ax2.set_title('Efficacy vs Safety Trade-off')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Effect of Reward Function Design', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'figure4_reward_comparison.png')
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✓ Saved: {output_path}")


# =============================================================================
# FIGURE 5: PPO Clipping Explanation (Educational)
# =============================================================================
def plot_clipping_explanation(output_dir, epsilon=0.1):
    """
    Generate PPO clipping explanation figure for the paper.
    """
    print("\n5. Generating PPO Clipping Explanation Figure...")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
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
    ax1.plot(r, ppo, 'g-', linewidth=3, label='PPO: min(...)')
    
    ax1.axvline(x=1-epsilon, color='gray', linestyle=':', alpha=0.6)
    ax1.axvline(x=1+epsilon, color='gray', linestyle=':', alpha=0.6)
    ax1.fill_between(r, 0, ppo, where=(ppo > 0), alpha=0.1, color='green')
    
    ax1.annotate(f'1-ε={1-epsilon}', xy=(1-epsilon, -0.15), fontsize=10, ha='center')
    ax1.annotate(f'1+ε={1+epsilon}', xy=(1+epsilon, -0.15), fontsize=10, ha='center')
    
    ax1.set_xlabel('Probability Ratio r(θ)')
    ax1.set_ylabel('Objective')
    ax1.set_title('Positive Advantage (Â > 0)\n"Good action — clipping caps the bonus"')
    ax1.legend(loc='upper left')
    ax1.set_xlim([0, 2])
    ax1.set_ylim([-0.3, 2.2])
    ax1.grid(True, alpha=0.3)
    
    # ===== RIGHT: Negative Advantage =====
    ax2 = axes[1]
    A_neg = -1.0
    
    unclipped = r * A_neg
    clipped = r_clipped * A_neg
    ppo = np.minimum(unclipped, clipped)
    
    ax2.plot(r, unclipped, 'b--', linewidth=2, label='Unclipped: r·Â', alpha=0.7)
    ax2.plot(r, clipped, 'r--', linewidth=2, label='Clipped: clip(r)·Â', alpha=0.7)
    ax2.plot(r, ppo, 'g-', linewidth=3, label='PPO: min(...)')
    
    ax2.axvline(x=1-epsilon, color='gray', linestyle=':', alpha=0.6)
    ax2.axvline(x=1+epsilon, color='gray', linestyle=':', alpha=0.6)
    ax2.fill_between(r, ppo, 0, where=(ppo < 0), alpha=0.1, color='green')
    
    ax2.annotate(f'1-ε={1-epsilon}', xy=(1-epsilon, 0.15), fontsize=10, ha='center')
    ax2.annotate(f'1+ε={1+epsilon}', xy=(1+epsilon, 0.15), fontsize=10, ha='center')
    
    ax2.set_xlabel('Probability Ratio r(θ)')
    ax2.set_ylabel('Objective')
    ax2.set_title('Negative Advantage (Â < 0)\n"Bad action — clipping limits the penalty"')
    ax2.legend(loc='lower left')
    ax2.set_xlim([0, 2])
    ax2.set_ylim([-2.2, 0.3])
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle(f'PPO Clipped Surrogate Objective (ε = {epsilon})', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'figure5_clipping_explanation.png')
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✓ Saved: {output_path}")


# =============================================================================
# MAIN
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description='Generate publication figures for PPO glucose control')
    parser.add_argument('--results-dir', type=str, default='./results',
                       help='Directory containing training results (perf_seed* folders)')
    parser.add_argument('--experiments-dir', type=str, default='./experiments',
                       help='Directory containing experiment results')
    parser.add_argument('--output-dir', type=str, default='./figures',
                       help='Where to save figures')
    parser.add_argument('--epsilon', type=float, default=0.1,
                       help='Epsilon value for clipping explanation figure')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("\n" + "="*60)
    print("GENERATING PUBLICATION FIGURES")
    print("="*60)
    print(f"Results dir:     {args.results_dir}")
    print(f"Experiments dir: {args.experiments_dir}")
    print(f"Output dir:      {args.output_dir}")
    print("="*60)
    
    # Generate all figures
    plot_model_performance(args.results_dir, args.output_dir)
    plot_baseline_comparison(args.experiments_dir, args.output_dir)
    plot_epsilon_comparison(args.experiments_dir, args.output_dir)
    plot_reward_comparison(args.experiments_dir, args.output_dir)
    plot_clipping_explanation(args.output_dir, epsilon=args.epsilon)
    
    print("\n" + "="*60)
    print("COMPLETE!")
    print("="*60)
    print(f"\nFigures saved to: {args.output_dir}/")
    print("""
Files generated:
  figure1_model_performance.png   → Results section
  figure2_baseline_comparison.png → Results section  
  figure3_epsilon_comparison.png  → Results section
  figure4_reward_comparison.png   → Results section
  figure5_clipping_explanation.png → Methods section
""")


if __name__ == "__main__":
    main()
