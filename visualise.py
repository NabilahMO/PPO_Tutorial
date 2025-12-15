"""
Visualisation Module for PPO Glucose Control
=============================================

Simplified, publication-ready visualisations with:
- Maximum 4 subplots per figure
- Clear legends on all plots
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
from scipy.ndimage import uniform_filter1d

# Set style for publication-quality figures
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Default figure settings
FIGURE_DPI = 300
DEFAULT_FIGSIZE = (12, 6)
SMALL_FIGSIZE = (10, 5)

# Colour scheme
COLOURS = {
    'reward': '#2E86AB',       # Blue
    'tir': '#28A745',          # Green
    'policy_loss': '#2E86AB',  # Blue
    'value_loss': '#A23B72',   # Magenta
    'entropy': '#28A745',      # Green
    'explained_var': '#F18F01', # Orange
    'clipped': '#2E86AB',      # Blue
    'kl': '#F18F01',           # Orange
    'target_range': '#90EE90', # Light green
    'hypo': '#FFB3B3',         # Light red
    'hyper': '#FFE4B3',        # Light orange
    'insulin': '#A23B72'       # Magenta
}


class PPOVisualiser:
    """
    Simplified visualisation class for PPO glucose control.
    
    Design principles:
    - Maximum 4 subplots per figure
    - Every plot has a legend
    - No redundant information
    - Clean, publication-ready style
    """
    
    def __init__(
        self,
        save_dir: str = "./figures",
        figsize: Tuple[int, int] = DEFAULT_FIGSIZE,
        dpi: int = FIGURE_DPI
    ):
        """
        Initialise the visualiser.
        
        Args:
            save_dir: Directory to save figures
            figsize: Default figure size
            dpi: Resolution for saved figures
        """
        self.save_dir = save_dir
        self.figsize = figsize
        self.dpi = dpi
        
        os.makedirs(save_dir, exist_ok=True)
    
    def _save_figure(self, fig: plt.Figure, filename: str) -> str:
        """Save figure and return path."""
        filepath = os.path.join(self.save_dir, filename)
        fig.savefig(filepath, dpi=self.dpi, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        print(f"  Saved: {filepath}")
        return filepath
    
    # ================================================================
    # FIGURE 1: LEARNING CURVE (Single Plot with Dual Axis)
    # ================================================================
    
    def plot_learning_curve(
        self,
        episode_rewards: List[float],
        clinical_metrics: Optional[List[Dict]] = None,
        window: int = 20,
        title: str = "PPO Training Progress",
        filename: str = "01_learning_curve.png"
    ) -> str:
        """
        Plot learning curve with reward and TIR on dual axes.
        
        Single, clear plot showing:
        - Episode rewards (left axis, blue)
        - Time in Range % (right axis, green)
        - Smoothed trends for both
        
        Args:
            episode_rewards: List of episode rewards
            clinical_metrics: List of clinical metric dicts
            window: Smoothing window size
            title: Figure title
            filename: Output filename
        
        Returns:
            Path to saved figure
        """
        fig, ax1 = plt.subplots(figsize=SMALL_FIGSIZE)
        
        episodes = np.arange(len(episode_rewards))
        rewards = np.array(episode_rewards)
        
        # Smooth rewards
        window = min(window, len(rewards) // 3) if len(rewards) > 10 else 1
        if window > 1:
            rewards_smooth = uniform_filter1d(rewards, size=window, mode='nearest')
        else:
            rewards_smooth = rewards
        
        # Plot rewards on left axis
        ax1.plot(episodes, rewards, alpha=0.2, color=COLOURS['reward'])
        line1, = ax1.plot(episodes, rewards_smooth, color=COLOURS['reward'], 
                         linewidth=2, label='Episode Reward')
        ax1.set_xlabel('Episode', fontsize=11)
        ax1.set_ylabel('Episode Reward', color=COLOURS['reward'], fontsize=11)
        ax1.tick_params(axis='y', labelcolor=COLOURS['reward'])
        
        # Plot TIR on right axis if available
        if clinical_metrics is not None and len(clinical_metrics) > 0:
            ax2 = ax1.twinx()
            
            tir = np.array([c.get('time_in_range', 0) for c in clinical_metrics])
            
            if window > 1 and len(tir) > window:
                tir_smooth = uniform_filter1d(tir, size=window, mode='nearest')
            else:
                tir_smooth = tir
            
            ax2.plot(episodes[:len(tir)], tir, alpha=0.2, color=COLOURS['tir'])
            line2, = ax2.plot(episodes[:len(tir)], tir_smooth, color=COLOURS['tir'], 
                             linewidth=2, label='Time in Range (%)')
            
            # Add clinical target line
            ax2.axhline(y=70, color=COLOURS['tir'], linestyle='--', 
                       alpha=0.5, linewidth=1)
            ax2.text(len(episodes) * 0.02, 72, 'Clinical target (70%)', 
                    fontsize=9, color=COLOURS['tir'], alpha=0.7)
            
            ax2.set_ylabel('Time in Range (%)', color=COLOURS['tir'], fontsize=11)
            ax2.tick_params(axis='y', labelcolor=COLOURS['tir'])
            ax2.set_ylim([0, 100])
            
            # Combined legend
            lines = [line1, line2]
            labels = [l.get_label() for l in lines]
            ax1.legend(lines, labels, loc='lower left', fontsize=10)
        else:
            ax1.legend(loc='lower right', fontsize=10)
        
        ax1.set_title(title, fontsize=13, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return self._save_figure(fig, filename)
    
    # ================================================================
    # FIGURE 2: TRAINING METRICS (2x2 Grid)
    # ================================================================
    
    def plot_training_metrics(
        self,
        metrics: Dict[str, List[float]],
        title: str = "PPO Training Metrics",
        filename: str = "02_training_metrics.png"
    ) -> str:
        """
        Plot PPO training metrics in a clean 2x2 grid.
        
        Shows:
        - Policy loss (should decrease)
        - Value loss (should decrease)
        - Policy entropy (should decrease as policy becomes confident)
        - Explained variance (should increase towards 1.0)
        
        Args:
            metrics: Dictionary with metric names as keys
            title: Figure title
            filename: Output filename
        
        Returns:
            Path to saved figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle(title, fontsize=14, fontweight='bold')
        
        # Define what to plot
        plot_configs = [
            ('policy_loss', 'Policy Loss', COLOURS['policy_loss'], 
             'Should decrease and stabilise'),
            ('value_loss', 'Value Loss', COLOURS['value_loss'],
             'Should decrease over training'),
            ('entropy', 'Policy Entropy', COLOURS['entropy'],
             'Decreases as policy becomes confident'),
            ('explained_variance', 'Explained Variance', COLOURS['explained_var'],
             'Higher is better (1.0 = perfect)')
        ]
        
        for idx, (metric_key, label, colour, description) in enumerate(plot_configs):
            ax = axes[idx // 2, idx % 2]
            
            if metric_key in metrics and len(metrics[metric_key]) > 0:
                data = np.array(metrics[metric_key])
                updates = np.arange(len(data))
                
                # Plot raw and smoothed
                ax.plot(updates, data, alpha=0.3, color=colour)
                
                if len(data) > 5:
                    window = max(3, len(data) // 10)
                    smoothed = uniform_filter1d(data, size=window, mode='nearest')
                    ax.plot(updates, smoothed, color=colour, linewidth=2, label=label)
                else:
                    ax.plot(updates, data, color=colour, linewidth=2, label=label)
                
                # Add reference lines for explained variance
                if metric_key == 'explained_variance':
                    ax.axhline(y=1.0, color='green', linestyle='--', 
                              alpha=0.7, label='Ideal (1.0)')
                    ax.axhline(y=0.0, color='red', linestyle='--', 
                              alpha=0.7, label='Random (0.0)')
                    ax.set_ylim([-0.5, 1.1])
                
                ax.legend(loc='best', fontsize=9)
            else:
                ax.text(0.5, 0.5, f'No {label} data', 
                       ha='center', va='center', transform=ax.transAxes,
                       fontsize=11, color='gray')
            
            ax.set_xlabel('Update', fontsize=10)
            ax.set_ylabel(label, fontsize=10)
            ax.set_title(f'{label}\n({description})', fontsize=10)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return self._save_figure(fig, filename)
    
    # ================================================================
    # FIGURE 3: PPO DIAGNOSTICS (1x2 Grid - Clipping + KL)
    # ================================================================
    
    def plot_ppo_diagnostics(
        self,
        metrics: Dict[str, List[float]],
        title: str = "PPO Clipping Diagnostics",
        filename: str = "03_ppo_diagnostics.png"
    ) -> str:
        """
        Plot PPO-specific diagnostics: clipping and KL divergence.
        
        Shows:
        - Clipped fraction: % of ratios clipped (left)
        - KL divergence: policy change magnitude (right)
        
        Args:
            metrics: Dictionary with 'clipped_fraction' and 'kl_divergence'
            title: Figure title
            filename: Output filename
        
        Returns:
            Path to saved figure
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(title, fontsize=14, fontweight='bold')
        
        # ---- Left: Clipped Fraction ----
        ax1 = axes[0]
        if 'clipped_fraction' in metrics and len(metrics['clipped_fraction']) > 0:
            clipped = np.array(metrics['clipped_fraction']) * 100
            updates = np.arange(len(clipped))
            
            ax1.plot(updates, clipped, alpha=0.3, color=COLOURS['clipped'])
            
            if len(clipped) > 5:
                window = max(3, len(clipped) // 10)
                smoothed = uniform_filter1d(clipped, size=window, mode='nearest')
                ax1.plot(updates, smoothed, color=COLOURS['clipped'], 
                        linewidth=2, label='Clipped fraction')
            else:
                ax1.plot(updates, clipped, color=COLOURS['clipped'], 
                        linewidth=2, label='Clipped fraction')
            
            ax1.set_xlabel('Update', fontsize=11)
            ax1.set_ylabel('Clipped Fraction (%)', fontsize=11)
            ax1.set_title('Fraction of Ratios Clipped\n(High = PPO actively constraining)', 
                         fontsize=11)
            ax1.legend(loc='best', fontsize=10)
            ax1.set_ylim([0, 100])
        else:
            ax1.text(0.5, 0.5, 'No clipping data', 
                    ha='center', va='center', transform=ax1.transAxes)
        ax1.grid(True, alpha=0.3)
        
        # ---- Right: KL Divergence ----
        ax2 = axes[1]
        if 'kl_divergence' in metrics and len(metrics['kl_divergence']) > 0:
            kl = np.array(metrics['kl_divergence'])
            updates = np.arange(len(kl))
            
            ax2.plot(updates, kl, alpha=0.3, color=COLOURS['kl'])
            
            if len(kl) > 5:
                window = max(3, len(kl) // 10)
                smoothed = uniform_filter1d(kl, size=window, mode='nearest')
                ax2.plot(updates, smoothed, color=COLOURS['kl'], 
                        linewidth=2, label='KL divergence')
            else:
                ax2.plot(updates, kl, color=COLOURS['kl'], 
                        linewidth=2, label='KL divergence')
            
            # TRPO target reference
            ax2.axhline(y=0.01, color='green', linestyle='--', 
                       alpha=0.7, label='TRPO target (δ=0.01)')
            
            ax2.set_xlabel('Update', fontsize=11)
            ax2.set_ylabel('KL Divergence', fontsize=11)
            ax2.set_title('Policy Change (KL Divergence)\n(Lower = more conservative updates)', 
                         fontsize=11)
            ax2.legend(loc='best', fontsize=10)
            ax2.set_yscale('log')
        else:
            ax2.text(0.5, 0.5, 'No KL divergence data', 
                    ha='center', va='center', transform=ax2.transAxes)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return self._save_figure(fig, filename)
    
    # ================================================================
    # FIGURE 4: CLIPPED OBJECTIVE EXPLANATION (Keep as is)
    # ================================================================
    
    def plot_clipped_objective(
        self,
        epsilon: float = 0.2,
        title: str = "PPO Clipped Objective",
        filename: str = "04_clipped_objective.png"
    ) -> str:
        """
        Educational visualisation of PPO clipped objective.
        
        Recreates Figure 1 from PPO paper showing how clipping
        works for positive and negative advantages.
        
        Args:
            epsilon: Clipping parameter
            title: Figure title
            filename: Output filename
        
        Returns:
            Path to saved figure
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle(title, fontsize=14, fontweight='bold')
        
        r = np.linspace(0.0, 2.0, 1000)
        
        # ---- Left: Positive Advantage ----
        ax1 = axes[0]
        A_pos = 1.0
        
        unclipped = r * A_pos
        r_clipped = np.clip(r, 1 - epsilon, 1 + epsilon)
        clipped = r_clipped * A_pos
        ppo = np.minimum(unclipped, clipped)
        
        ax1.plot(r, unclipped, 'b--', linewidth=2, label='Unclipped: r·A', alpha=0.7)
        ax1.plot(r, clipped, 'r--', linewidth=2, label='Clipped: clip(r)·A', alpha=0.7)
        ax1.plot(r, ppo, 'g-', linewidth=3, label='PPO: min(...)')
        
        ax1.axvline(x=1-epsilon, color='gray', linestyle=':', alpha=0.5)
        ax1.axvline(x=1+epsilon, color='gray', linestyle=':', alpha=0.5)
        ax1.axvline(x=1.0, color='black', linestyle='-', alpha=0.3)
        
        ax1.fill_between(r, 0, ppo, where=(ppo > 0), alpha=0.15, color='green')
        
        ax1.annotate(f'1-ε = {1-epsilon}', xy=(1-epsilon, -0.3), fontsize=10, ha='center')
        ax1.annotate(f'1+ε = {1+epsilon}', xy=(1+epsilon, -0.3), fontsize=10, ha='center')
        ax1.annotate('Clipping prevents\nover-optimisation', 
                    xy=(1.6, 1.0), fontsize=10, ha='center',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        ax1.set_xlabel('Probability Ratio r(θ)', fontsize=12)
        ax1.set_ylabel('Objective L(θ)', fontsize=12)
        ax1.set_title('Positive Advantage (A > 0)\n"Good action - increase probability"', 
                     fontsize=11)
        ax1.legend(loc='upper left', fontsize=10)
        ax1.set_xlim([0, 2])
        ax1.set_ylim([-0.5, 2.5])
        ax1.grid(True, alpha=0.3)
        
        # ---- Right: Negative Advantage ----
        ax2 = axes[1]
        A_neg = -1.0
        
        unclipped = r * A_neg
        clipped = r_clipped * A_neg
        ppo = np.minimum(unclipped, clipped)
        
        ax2.plot(r, unclipped, 'b--', linewidth=2, label='Unclipped: r·A', alpha=0.7)
        ax2.plot(r, clipped, 'r--', linewidth=2, label='Clipped: clip(r)·A', alpha=0.7)
        ax2.plot(r, ppo, 'g-', linewidth=3, label='PPO: min(...)')
        
        ax2.axvline(x=1-epsilon, color='gray', linestyle=':', alpha=0.5)
        ax2.axvline(x=1+epsilon, color='gray', linestyle=':', alpha=0.5)
        ax2.axvline(x=1.0, color='black', linestyle='-', alpha=0.3)
        
        ax2.fill_between(r, ppo, 0, where=(ppo < 0), alpha=0.15, color='green')
        
        ax2.annotate(f'1-ε = {1-epsilon}', xy=(1-epsilon, 0.3), fontsize=10, ha='center')
        ax2.annotate(f'1+ε = {1+epsilon}', xy=(1+epsilon, 0.3), fontsize=10, ha='center')
        ax2.annotate('Clipping prevents\ncomplete elimination', 
                    xy=(0.4, -1.0), fontsize=10, ha='center',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        ax2.set_xlabel('Probability Ratio r(θ)', fontsize=12)
        ax2.set_ylabel('Objective L(θ)', fontsize=12)
        ax2.set_title('Negative Advantage (A < 0)\n"Bad action - decrease probability"', 
                     fontsize=11)
        ax2.legend(loc='lower left', fontsize=10)
        ax2.set_xlim([0, 2])
        ax2.set_ylim([-2.5, 0.5])
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return self._save_figure(fig, filename)
    
    # ================================================================
    # FIGURE 5: GLUCOSE PROFILE (1x2 Grid)
    # ================================================================
    
    def plot_glucose_profile(
        self,
        glucose_trace: List[float],
        insulin_trace: Optional[List[float]] = None,
        meal_times: Optional[List[float]] = None,
        sample_time_minutes: float = 5.0,
        title: str = "24-Hour Glucose Profile",
        filename: str = "05_glucose_profile.png"
    ) -> str:
        """
        Plot glucose and insulin profiles for a single episode.
        
        Args:
            glucose_trace: Blood glucose values
            insulin_trace: Insulin doses
            meal_times: Meal times in hours
            sample_time_minutes: Time between samples
            title: Figure title
            filename: Output filename
        
        Returns:
            Path to saved figure
        """
        n_samples = len(glucose_trace)
        time_hours = np.arange(n_samples) * sample_time_minutes / 60.0
        glucose = np.array(glucose_trace)
        
        if insulin_trace is not None:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            ax1, ax2 = axes
        else:
            fig, ax1 = plt.subplots(figsize=(10, 5))
        
        fig.suptitle(title, fontsize=14, fontweight='bold')
        
        # ---- Glucose Plot ----
        # Background zones
        ax1.axhspan(70, 180, color=COLOURS['target_range'], alpha=0.3, 
                   label='Target range (70-180)')
        ax1.axhspan(0, 70, color=COLOURS['hypo'], alpha=0.3, 
                   label='Hypoglycaemia (<70)')
        ax1.axhspan(180, 400, color=COLOURS['hyper'], alpha=0.3, 
                   label='Hyperglycaemia (>180)')
        
        # Glucose trace
        ax1.plot(time_hours, glucose, color='#2E86AB', linewidth=2, 
                label='Blood glucose')
        
        # Target lines
        ax1.axhline(y=70, color='red', linestyle='--', alpha=0.5, linewidth=1)
        ax1.axhline(y=180, color='orange', linestyle='--', alpha=0.5, linewidth=1)
        
        # Meal markers
        if meal_times is not None:
            for i, meal_time in enumerate(meal_times):
                ax1.axvline(x=meal_time, color='purple', linestyle='--', alpha=0.5)
                if i == 0:
                    ax1.annotate('Meal', xy=(meal_time, 320), fontsize=9, 
                               color='purple', ha='center')
        
        # Calculate and display TIR
        tir = np.mean((glucose >= 70) & (glucose <= 180)) * 100
        tbr = np.mean(glucose < 70) * 100
        tar = np.mean(glucose > 180) * 100
        
        stats_text = f'TIR: {tir:.1f}%\nTBR: {tbr:.1f}%\nTAR: {tar:.1f}%'
        ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax1.set_xlabel('Time (hours)', fontsize=11)
        ax1.set_ylabel('Blood Glucose (mg/dL)', fontsize=11)
        ax1.set_title('Glucose Levels', fontsize=11)
        ax1.set_ylim([40, 350])
        ax1.set_xlim([0, max(time_hours)])
        ax1.legend(loc='upper right', fontsize=9)
        ax1.grid(True, alpha=0.3)
        
        # ---- Insulin Plot ----
        if insulin_trace is not None:
            insulin = np.array(insulin_trace)
            time_insulin = time_hours[1:len(insulin)+1]
            
            ax2.fill_between(time_insulin, 0, insulin, 
                           color=COLOURS['insulin'], alpha=0.4)
            ax2.plot(time_insulin, insulin, color=COLOURS['insulin'], 
                    linewidth=2, label='Insulin delivery')
            
            # Total insulin
            total = np.sum(insulin) * sample_time_minutes / 60.0
            ax2.text(0.98, 0.95, f'Total: {total:.1f} U',
                    transform=ax2.transAxes, fontsize=10,
                    verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            ax2.set_xlabel('Time (hours)', fontsize=11)
            ax2.set_ylabel('Insulin Rate (U/hr)', fontsize=11)
            ax2.set_title('Insulin Delivery', fontsize=11)
            ax2.set_ylim([0, max(insulin) * 1.2 + 0.1])
            ax2.set_xlim([0, max(time_hours)])
            ax2.legend(loc='upper right', fontsize=10)
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return self._save_figure(fig, filename)
    
    # ================================================================
    # FIGURE 6: EPSILON COMPARISON
    # ================================================================
    
    def plot_epsilon_comparison(
        self,
        experiment_results: Dict,
        title: str = "Effect of Clipping Parameter ε",
        filename: str = "06_epsilon_comparison.png"
    ) -> str:
        """
        Compare different epsilon values in a clean 2x2 grid.
        
        Shows:
        - Learning curves (top left)
        - Final reward comparison (top right)
        - Time in Range comparison (bottom left)
        - Training stability (bottom right)
        
        Args:
            experiment_results: Dict with epsilon values as keys
            title: Figure title
            filename: Output filename
        
        Returns:
            Path to saved figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(title, fontsize=14, fontweight='bold')
        
        epsilons = sorted(experiment_results.keys())
        colours = plt.cm.viridis(np.linspace(0.2, 0.8, len(epsilons)))
        
        # ---- Plot 1: Learning Curves ----
        ax1 = axes[0, 0]
        for eps, colour in zip(epsilons, colours):
            data = experiment_results[eps]
            if 'episode_rewards' in data:
                rewards = np.array(data['episode_rewards'])
                episodes = np.arange(len(rewards))
                
                window = min(30, len(rewards) // 5) if len(rewards) > 10 else 1
                if window > 1:
                    smoothed = uniform_filter1d(rewards, size=window, mode='nearest')
                else:
                    smoothed = rewards
                
                ax1.plot(episodes, smoothed, color=colour, linewidth=2, 
                        label=f'ε = {eps}')
        
        ax1.set_xlabel('Episode', fontsize=11)
        ax1.set_ylabel('Episode Reward', fontsize=11)
        ax1.set_title('Learning Curves', fontsize=11)
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # ---- Plot 2: Final Reward ----
        ax2 = axes[0, 1]
        final_rewards = []
        for eps in epsilons:
            data = experiment_results[eps]
            if 'final_eval' in data:
                final_rewards.append(data['final_eval'].get('mean_reward', 0))
            elif 'episode_rewards' in data:
                rewards = data['episode_rewards']
                final_rewards.append(np.mean(rewards[-20:]) if len(rewards) >= 20 else np.mean(rewards))
            else:
                final_rewards.append(0)
        
        bars = ax2.bar(range(len(epsilons)), final_rewards, color=colours)
        ax2.set_xticks(range(len(epsilons)))
        ax2.set_xticklabels([f'ε = {e}' for e in epsilons], fontsize=10)
        ax2.set_ylabel('Final Mean Reward', fontsize=11)
        ax2.set_title('Final Performance', fontsize=11)
        ax2.grid(True, alpha=0.3, axis='y')
        
        for bar, val in zip(bars, final_rewards):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f'{val:.0f}', ha='center', va='bottom', fontsize=10)
        
        # ---- Plot 3: Time in Range ----
        ax3 = axes[1, 0]
        tir_values = []
        for eps in epsilons:
            data = experiment_results[eps]
            if 'final_eval' in data:
                tir_values.append(data['final_eval'].get('mean_tir', 0))
            else:
                tir_values.append(0)
        
        bars = ax3.bar(range(len(epsilons)), tir_values, color=colours)
        ax3.axhline(y=70, color='green', linestyle='--', alpha=0.7, 
                   label='Clinical target (70%)')
        ax3.set_xticks(range(len(epsilons)))
        ax3.set_xticklabels([f'ε = {e}' for e in epsilons], fontsize=10)
        ax3.set_ylabel('Time in Range (%)', fontsize=11)
        ax3.set_title('Clinical Efficacy', fontsize=11)
        ax3.legend(fontsize=10)
        ax3.grid(True, alpha=0.3, axis='y')
        
        for bar, val in zip(bars, tir_values):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f'{val:.1f}', ha='center', va='bottom', fontsize=10)
        
        # ---- Plot 4: Stability ----
        ax4 = axes[1, 1]
        stability_values = []
        for eps in epsilons:
            data = experiment_results[eps]
            if 'episode_rewards' in data:
                rewards = np.array(data['episode_rewards'])
                # CV of final third
                n = len(rewards)
                final_third = rewards[2*n//3:]
                if len(final_third) > 0 and np.mean(np.abs(final_third)) > 0:
                    cv = np.std(final_third) / (np.abs(np.mean(final_third)) + 1e-8)
                else:
                    cv = 0
                stability_values.append(cv)
            else:
                stability_values.append(0)
        
        bars = ax4.bar(range(len(epsilons)), stability_values, color=colours)
        ax4.set_xticks(range(len(epsilons)))
        ax4.set_xticklabels([f'ε = {e}' for e in epsilons], fontsize=10)
        ax4.set_ylabel('Coefficient of Variation', fontsize=11)
        ax4.set_title('Training Stability (lower = more stable)', fontsize=11)
        ax4.grid(True, alpha=0.3, axis='y')
        
        for bar, val in zip(bars, stability_values):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f'{val:.2f}', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        return self._save_figure(fig, filename)
    
    # ================================================================
    # FIGURE 7: BASELINE COMPARISON
    # ================================================================
    
    def plot_baseline_comparison(
        self,
        results: List[Dict],
        title: str = "PPO vs Baseline Controllers",
        filename: str = "07_baseline_comparison.png"
    ) -> str:
        """
        Compare PPO against baseline controllers.
        
        Shows reward, TIR, and TBR side by side.
        
        Args:
            results: List of result dictionaries
            title: Figure title
            filename: Output filename
        
        Returns:
            Path to saved figure
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(title, fontsize=14, fontweight='bold')
        
        names = [r.get('name', f'Controller {i}') for i, r in enumerate(results)]
        colours = plt.cm.Set2(np.linspace(0, 1, len(names)))
        
        # Highlight PPO
        edge_colours = ['red' if 'PPO' in n else 'none' for n in names]
        linewidths = [3 if 'PPO' in n else 0 for n in names]
        
        metrics = [
            ('mean_reward', 'Mean Reward', None),
            ('mean_tir', 'Time in Range (%)', 70),
            ('mean_tbr', 'Time Below Range (%)', 4)
        ]
        
        for ax, (key, label, target) in zip(axes, metrics):
            values = [r.get(key, 0) for r in results]
            
            bars = ax.bar(range(len(names)), values, color=colours,
                         edgecolor=edge_colours, linewidth=linewidths)
            
            if target is not None:
                colour = 'green' if key == 'mean_tir' else 'red'
                ax.axhline(y=target, color=colour, linestyle='--', alpha=0.7,
                          label=f'Target ({target})')
                ax.legend(fontsize=9)
            
            ax.set_xticks(range(len(names)))
            ax.set_xticklabels(names, rotation=45, ha='right', fontsize=9)
            ax.set_ylabel(label, fontsize=11)
            ax.set_title(label, fontsize=11)
            ax.grid(True, alpha=0.3, axis='y')
            
            for bar, val in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                       f'{val:.1f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        return self._save_figure(fig, filename)


def generate_all_visualisations(
    results_dir: str,
    output_dir: Optional[str] = None
) -> List[str]:
    """
    Generate all visualisations from saved training results.
    
    Args:
        results_dir: Directory containing training results
        output_dir: Directory to save figures
    
    Returns:
        List of generated figure paths
    """
    if output_dir is None:
        output_dir = os.path.join(results_dir, "figures")
    
    metrics_path = os.path.join(results_dir, "metrics.json")
    if not os.path.exists(metrics_path):
        print(f"No metrics found at {metrics_path}")
        return []
    
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
    
    vis = PPOVisualiser(save_dir=output_dir)
    generated = []
    
    print(f"\nGenerating visualisations from {results_dir}...")
    
    # 1. Learning curve
    if 'episode_rewards' in metrics:
        path = vis.plot_learning_curve(
            episode_rewards=metrics['episode_rewards'],
            clinical_metrics=metrics.get('clinical_metrics')
        )
        generated.append(path)
    
    # 2. Training metrics
    if 'agent_metrics' in metrics:
        path = vis.plot_training_metrics(metrics['agent_metrics'])
        generated.append(path)
        
        # 3. PPO diagnostics
        path = vis.plot_ppo_diagnostics(metrics['agent_metrics'])
        generated.append(path)
    
    # 4. Clipped objective explanation
    path = vis.plot_clipped_objective()
    generated.append(path)
    
    print(f"\nGenerated {len(generated)} figures in {output_dir}")
    return generated


def main():
    """Test visualisation functions."""
    print("Generating test visualisations...")
    
    vis = PPOVisualiser(save_dir="./test_figures")
    
    # Generate clipped objective
    vis.plot_clipped_objective()
    
    print("\n✓ Test complete!")


if __name__ == "__main__":
    main()