"""
Visualisation Module for PPO Glucose Control
=============================================

Comprehensive plotting functions for:
- Training progress (learning curves, losses)
- PPO-specific metrics (clipping behaviour, ratios)
- Clinical metrics (glucose profiles, time in range)
- Controller comparisons
- Publication-ready figures

All plots follow consistent styling and are designed
to be informative for both RL and clinical audiences.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Union
from scipy.ndimage import uniform_filter1d

# Set style for publication-quality figures
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Default figure settings
FIGURE_DPI = 300
FIGURE_FORMAT = 'png'
DEFAULT_FIGSIZE = (10, 6)
SMALL_FIGSIZE = (8, 5)
LARGE_FIGSIZE = (14, 10)

# Colour scheme
COLOURS = {
    'primary': '#2E86AB',      # Blue
    'secondary': '#A23B72',    # Magenta
    'success': '#28A745',      # Green
    'warning': '#F18F01',      # Orange
    'danger': '#C73E1D',       # Red
    'neutral': '#6C757D',      # Grey
    'target_range': '#90EE90', # Light green for glucose target
    'hypo': '#FFB3B3',         # Light red for hypoglycaemia
    'hyper': '#FFE4B3'         # Light orange for hyperglycaemia
}


class PPOVisualiser:
    """
    Visualisation class for PPO glucose control experiments.
    
    Provides methods for plotting:
    - Training curves and metrics
    - PPO-specific diagnostics
    - Clinical glucose profiles
    - Controller comparisons
    """
    
    def __init__(
        self,
        save_dir: str = "./figures",
        figsize: Tuple[int, int] = DEFAULT_FIGSIZE,
        dpi: int = FIGURE_DPI,
        style: str = 'seaborn-v0_8-whitegrid'
    ):
        """
        Initialise the visualiser.
        
        Args:
            save_dir: Directory to save figures
            figsize: Default figure size
            dpi: Resolution for saved figures
            style: Matplotlib style
        """
        self.save_dir = save_dir
        self.figsize = figsize
        self.dpi = dpi
        
        os.makedirs(save_dir, exist_ok=True)
        plt.style.use(style)
    
    def _save_figure(self, fig: plt.Figure, filename: str) -> str:
        """Save figure and return path."""
        filepath = os.path.join(self.save_dir, filename)
        fig.savefig(filepath, dpi=self.dpi, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        print(f"  Saved: {filepath}")
        return filepath
    
    # ================================================================
    # TRAINING CURVES
    # ================================================================
    
    def plot_learning_curves(
        self,
        episode_rewards: List[float],
        episode_timesteps: Optional[List[int]] = None,
        clinical_metrics: Optional[List[Dict]] = None,
        window: int = 50,
        title: str = "PPO Learning Curves",
        filename: str = "01_learning_curves.png"
    ) -> str:
        """
        Plot learning curves showing training progress.
        
        Creates a 2x2 figure with:
        - Episode rewards over time
        - Episode rewards vs timesteps
        - Time in Range progression (if clinical metrics provided)
        - Reward distribution across training
        
        Args:
            episode_rewards: List of episode rewards
            episode_timesteps: Timestep at end of each episode
            clinical_metrics: List of clinical metric dicts
            window: Smoothing window size
            title: Figure title
            filename: Output filename
        
        Returns:
            Path to saved figure
        """
        fig, axes = plt.subplots(2, 2, figsize=LARGE_FIGSIZE)
        fig.suptitle(title, fontsize=14, fontweight='bold')
        
        episodes = np.arange(len(episode_rewards))
        rewards = np.array(episode_rewards)
        
        # Smooth rewards
        if len(rewards) > window:
            smoothed = uniform_filter1d(rewards, size=window, mode='nearest')
        else:
            smoothed = rewards
        
        # ---- Plot 1: Rewards over episodes ----
        ax1 = axes[0, 0]
        ax1.plot(episodes, rewards, alpha=0.3, color=COLOURS['primary'], label='Raw')
        ax1.plot(episodes, smoothed, color=COLOURS['primary'], linewidth=2, label=f'Smoothed (w={window})')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Episode Reward')
        ax1.set_title('Episode Rewards')
        ax1.legend(loc='lower right')
        ax1.grid(True, alpha=0.3)
        
        # ---- Plot 2: Rewards over timesteps ----
        ax2 = axes[0, 1]
        if episode_timesteps is not None:
            ax2.plot(episode_timesteps, rewards, alpha=0.3, color=COLOURS['secondary'])
            if len(rewards) > window:
                smoothed_ts = uniform_filter1d(rewards, size=window, mode='nearest')
                ax2.plot(episode_timesteps, smoothed_ts, color=COLOURS['secondary'], linewidth=2)
            ax2.set_xlabel('Timesteps')
        else:
            ax2.plot(episodes, smoothed, color=COLOURS['secondary'], linewidth=2)
            ax2.set_xlabel('Episode')
        ax2.set_ylabel('Episode Reward')
        ax2.set_title('Rewards vs Training Progress')
        ax2.grid(True, alpha=0.3)
        
        # ---- Plot 3: Time in Range progression ----
        ax3 = axes[1, 0]
        if clinical_metrics is not None and len(clinical_metrics) > 0:
            tir = [c.get('time_in_range', 0) for c in clinical_metrics]
            tir = np.array(tir)
            
            ax3.plot(episodes[:len(tir)], tir, alpha=0.3, color=COLOURS['success'])
            if len(tir) > window:
                tir_smoothed = uniform_filter1d(tir, size=window, mode='nearest')
                ax3.plot(episodes[:len(tir)], tir_smoothed, color=COLOURS['success'], linewidth=2)
            
            ax3.axhline(y=70, color=COLOURS['danger'], linestyle='--', alpha=0.7, label='Clinical target (70%)')
            ax3.set_ylim([0, 100])
            ax3.set_xlabel('Episode')
            ax3.set_ylabel('Time in Range (%)')
            ax3.set_title('Clinical Metric: Time in Range (70-180 mg/dL)')
            ax3.legend(loc='lower right')
        else:
            ax3.text(0.5, 0.5, 'No clinical metrics available', 
                    ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('Clinical Metrics')
        ax3.grid(True, alpha=0.3)
        
        # ---- Plot 4: Reward distribution ----
        ax4 = axes[1, 1]
        
        # Split into thirds
        n = len(rewards)
        if n >= 30:
            third = n // 3
            early = rewards[:third]
            mid = rewards[third:2*third]
            late = rewards[2*third:]
            
            parts = [early, mid, late]
            labels = ['Early\n(first third)', 'Middle\n(second third)', 'Late\n(final third)']
            positions = [1, 2, 3]
            
            bp = ax4.boxplot(parts, positions=positions, patch_artist=True,
                           labels=labels, widths=0.6)
            
            colours_box = [COLOURS['warning'], COLOURS['primary'], COLOURS['success']]
            for patch, colour in zip(bp['boxes'], colours_box):
                patch.set_facecolor(colour)
                patch.set_alpha(0.6)
        else:
            ax4.boxplot([rewards], patch_artist=True)
            ax4.set_xticklabels(['All episodes'])
        
        ax4.set_ylabel('Episode Reward')
        ax4.set_title('Reward Distribution Across Training')
        ax4.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        return self._save_figure(fig, filename)
    
    def plot_training_metrics(
        self,
        metrics: Dict[str, List[float]],
        title: str = "PPO Training Metrics",
        filename: str = "02_training_metrics.png"
    ) -> str:
        """
        Plot PPO training metrics (losses, entropy, etc.).
        
        Args:
            metrics: Dictionary with metric names as keys
            title: Figure title
            filename: Output filename
        
        Returns:
            Path to saved figure
        """
        # Select metrics to plot
        metric_names = ['policy_loss', 'value_loss', 'entropy', 'explained_variance']
        available = [m for m in metric_names if m in metrics and len(metrics[m]) > 0]
        
        if len(available) == 0:
            print("  No training metrics available to plot")
            return ""
        
        n_plots = len(available)
        n_cols = 2
        n_rows = (n_plots + 1) // 2
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4 * n_rows))
        fig.suptitle(title, fontsize=14, fontweight='bold')
        
        if n_plots == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        titles = {
            'policy_loss': 'Policy Loss',
            'value_loss': 'Value Loss',
            'entropy': 'Policy Entropy',
            'explained_variance': 'Explained Variance',
            'total_loss': 'Total Loss',
            'kl_divergence': 'KL Divergence'
        }
        
        colours_list = [COLOURS['primary'], COLOURS['secondary'], 
                       COLOURS['success'], COLOURS['warning']]
        
        for idx, metric_name in enumerate(available):
            ax = axes[idx]
            data = np.array(metrics[metric_name])
            updates = np.arange(len(data))
            
            colour = colours_list[idx % len(colours_list)]
            
            # Plot with smoothing
            ax.plot(updates, data, alpha=0.3, color=colour)
            if len(data) > 10:
                smoothed = uniform_filter1d(data, size=min(10, len(data)//5), mode='nearest')
                ax.plot(updates, smoothed, color=colour, linewidth=2)
            
            ax.set_xlabel('Update')
            ax.set_ylabel(metric_name.replace('_', ' ').title())
            ax.set_title(titles.get(metric_name, metric_name))
            ax.grid(True, alpha=0.3)
            
            # Add reference lines
            if metric_name == 'explained_variance':
                ax.axhline(y=1.0, color=COLOURS['success'], linestyle='--', 
                          alpha=0.7, label='Ideal (1.0)')
                ax.axhline(y=0.0, color=COLOURS['danger'], linestyle='--', 
                          alpha=0.7, label='Random (0.0)')
                ax.legend(loc='lower right', fontsize=8)
                ax.set_ylim([-0.5, 1.1])
        
        # Hide unused axes
        for idx in range(len(available), len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        return self._save_figure(fig, filename)
    
    # ================================================================
    # PPO-SPECIFIC VISUALISATIONS
    # ================================================================
    
    def plot_clipping_behaviour(
        self,
        metrics: Dict[str, List[float]],
        title: str = "PPO Clipping Behaviour",
        filename: str = "03_clipping_behaviour.png"
    ) -> str:
        """
        Visualise PPO clipping mechanism behaviour.
        
        Shows:
        - Clipped fraction over training
        - Ratio statistics
        - KL divergence
        - Relationship between clipping and KL
        
        Args:
            metrics: Dictionary with 'clipped_fraction', 'ratio_mean', etc.
            title: Figure title
            filename: Output filename
        
        Returns:
            Path to saved figure
        """
        fig, axes = plt.subplots(2, 2, figsize=LARGE_FIGSIZE)
        fig.suptitle(title, fontsize=14, fontweight='bold')
        
        # ---- Plot 1: Clipped fraction over time ----
        ax1 = axes[0, 0]
        if 'clipped_fraction' in metrics and len(metrics['clipped_fraction']) > 0:
            clipped = np.array(metrics['clipped_fraction']) * 100  # Convert to percentage
            updates = np.arange(len(clipped))
            
            ax1.plot(updates, clipped, alpha=0.3, color=COLOURS['primary'])
            if len(clipped) > 10:
                smoothed = uniform_filter1d(clipped, size=10, mode='nearest')
                ax1.plot(updates, smoothed, color=COLOURS['primary'], linewidth=2)
            
            ax1.set_xlabel('Update')
            ax1.set_ylabel('Clipped Fraction (%)')
            ax1.set_title('Fraction of Ratios Clipped')
            
            # Add annotation
            ax1.annotate('High clipping = PPO actively constraining updates',
                        xy=(0.5, 0.95), xycoords='axes fraction',
                        fontsize=9, ha='center', style='italic',
                        color=COLOURS['neutral'])
        else:
            ax1.text(0.5, 0.5, 'No clipping data available',
                    ha='center', va='center', transform=ax1.transAxes)
        ax1.grid(True, alpha=0.3)
        
        # ---- Plot 2: Ratio statistics ----
        ax2 = axes[0, 1]
        if 'ratio_mean' in metrics and len(metrics['ratio_mean']) > 0:
            ratio_mean = np.array(metrics['ratio_mean'])
            updates = np.arange(len(ratio_mean))
            
            ax2.plot(updates, ratio_mean, color=COLOURS['secondary'], 
                    linewidth=2, label='Ratio mean')
            
            if 'ratio_std' in metrics and len(metrics['ratio_std']) > 0:
                ratio_std = np.array(metrics['ratio_std'])
                ax2.fill_between(updates, 
                               ratio_mean - ratio_std,
                               ratio_mean + ratio_std,
                               alpha=0.3, color=COLOURS['secondary'],
                               label='±1 std')
            
            # Add clipping boundaries
            ax2.axhline(y=1.2, color=COLOURS['danger'], linestyle='--', 
                       alpha=0.7, label='Upper clip (1+ε)')
            ax2.axhline(y=0.8, color=COLOURS['danger'], linestyle='--', 
                       alpha=0.7, label='Lower clip (1-ε)')
            ax2.axhline(y=1.0, color=COLOURS['neutral'], linestyle='-', 
                       alpha=0.5, label='No change')
            
            ax2.set_xlabel('Update')
            ax2.set_ylabel('Probability Ratio')
            ax2.set_title('Probability Ratio Statistics')
            ax2.legend(loc='upper right', fontsize=8)
            ax2.set_ylim([0.5, 1.5])
        else:
            ax2.text(0.5, 0.5, 'No ratio data available',
                    ha='center', va='center', transform=ax2.transAxes)
        ax2.grid(True, alpha=0.3)
        
        # ---- Plot 3: KL divergence ----
        ax3 = axes[1, 0]
        if 'kl_divergence' in metrics and len(metrics['kl_divergence']) > 0:
            kl = np.array(metrics['kl_divergence'])
            updates = np.arange(len(kl))
            
            ax3.plot(updates, kl, alpha=0.3, color=COLOURS['warning'])
            if len(kl) > 10:
                smoothed = uniform_filter1d(kl, size=10, mode='nearest')
                ax3.plot(updates, smoothed, color=COLOURS['warning'], linewidth=2)
            
            # Add TRPO target line
            ax3.axhline(y=0.01, color=COLOURS['success'], linestyle='--',
                       alpha=0.7, label='TRPO target (δ=0.01)')
            
            ax3.set_xlabel('Update')
            ax3.set_ylabel('Approximate KL Divergence')
            ax3.set_title('Policy Change (KL Divergence)')
            ax3.legend(loc='upper right', fontsize=8)
            ax3.set_yscale('log')
        else:
            ax3.text(0.5, 0.5, 'No KL divergence data available',
                    ha='center', va='center', transform=ax3.transAxes)
        ax3.grid(True, alpha=0.3)
        
        # ---- Plot 4: Clipping vs KL relationship ----
        ax4 = axes[1, 1]
        if ('clipped_fraction' in metrics and 'kl_divergence' in metrics and
            len(metrics['clipped_fraction']) > 0 and len(metrics['kl_divergence']) > 0):
            
            clipped = np.array(metrics['clipped_fraction']) * 100
            kl = np.array(metrics['kl_divergence'])
            
            # Use same length
            min_len = min(len(clipped), len(kl))
            clipped = clipped[:min_len]
            kl = kl[:min_len]
            
            # Colour by training progress
            colours = plt.cm.viridis(np.linspace(0, 1, min_len))
            ax4.scatter(kl, clipped, c=colours, alpha=0.6, s=20)
            
            # Add colorbar
            sm = plt.cm.ScalarMappable(cmap='viridis', 
                                       norm=plt.Normalize(vmin=0, vmax=min_len))
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax4, label='Update')
            
            ax4.set_xlabel('KL Divergence')
            ax4.set_ylabel('Clipped Fraction (%)')
            ax4.set_title('Clipping vs Policy Change')
            ax4.set_xscale('log')
        else:
            ax4.text(0.5, 0.5, 'Insufficient data for scatter plot',
                    ha='center', va='center', transform=ax4.transAxes)
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return self._save_figure(fig, filename)
    
    def plot_clipped_objective_explanation(
        self,
        epsilon: float = 0.2,
        title: str = "PPO Clipped Objective Visualisation",
        filename: str = "04_clipped_objective.png"
    ) -> str:
        """
        Create educational visualisation of PPO clipped objective.
        
        Recreates Figure 1 from the PPO paper showing how
        clipping works for positive and negative advantages.
        
        Args:
            epsilon: Clipping parameter
            title: Figure title
            filename: Output filename
        
        Returns:
            Path to saved figure
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle(title, fontsize=14, fontweight='bold')
        
        # Ratio range
        r = np.linspace(0.0, 2.0, 1000)
        
        # ---- Left plot: Positive advantage (A > 0) ----
        ax1 = axes[0]
        A_pos = 1.0  # Positive advantage
        
        # Unclipped objective: r * A
        unclipped_pos = r * A_pos
        
        # Clipped objective: clip(r, 1-eps, 1+eps) * A
        r_clipped = np.clip(r, 1 - epsilon, 1 + epsilon)
        clipped_pos = r_clipped * A_pos
        
        # PPO objective: min(unclipped, clipped)
        ppo_pos = np.minimum(unclipped_pos, clipped_pos)
        
        ax1.plot(r, unclipped_pos, 'b--', linewidth=2, label='Unclipped: r·A', alpha=0.7)
        ax1.plot(r, clipped_pos, 'r--', linewidth=2, label='Clipped: clip(r)·A', alpha=0.7)
        ax1.plot(r, ppo_pos, 'g-', linewidth=3, label='PPO: min(...)')
        
        # Mark clipping boundaries
        ax1.axvline(x=1-epsilon, color='gray', linestyle=':', alpha=0.5)
        ax1.axvline(x=1+epsilon, color='gray', linestyle=':', alpha=0.5)
        ax1.axvline(x=1.0, color='black', linestyle='-', alpha=0.3)
        
        # Annotations
        ax1.annotate(f'1-ε = {1-epsilon}', xy=(1-epsilon, -0.3), fontsize=10, ha='center')
        ax1.annotate(f'1+ε = {1+epsilon}', xy=(1+epsilon, -0.3), fontsize=10, ha='center')
        ax1.annotate('Clipping prevents\nover-optimisation', 
                    xy=(1.5, 1.0), fontsize=10, ha='center',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        ax1.set_xlabel('Probability Ratio r(θ)', fontsize=12)
        ax1.set_ylabel('Objective L(θ)', fontsize=12)
        ax1.set_title('Positive Advantage (A > 0)\n"Good action - increase probability"', fontsize=11)
        ax1.legend(loc='upper left')
        ax1.set_xlim([0, 2])
        ax1.set_ylim([-0.5, 2.5])
        ax1.grid(True, alpha=0.3)
        
        # Shade the effective region
        ax1.fill_between(r, 0, ppo_pos, where=(ppo_pos > 0), 
                        alpha=0.2, color='green')
        
        # ---- Right plot: Negative advantage (A < 0) ----
        ax2 = axes[1]
        A_neg = -1.0  # Negative advantage
        
        # Unclipped objective
        unclipped_neg = r * A_neg
        
        # Clipped objective
        clipped_neg = r_clipped * A_neg
        
        # PPO objective
        ppo_neg = np.minimum(unclipped_neg, clipped_neg)
        
        ax2.plot(r, unclipped_neg, 'b--', linewidth=2, label='Unclipped: r·A', alpha=0.7)
        ax2.plot(r, clipped_neg, 'r--', linewidth=2, label='Clipped: clip(r)·A', alpha=0.7)
        ax2.plot(r, ppo_neg, 'g-', linewidth=3, label='PPO: min(...)')
        
        # Mark clipping boundaries
        ax2.axvline(x=1-epsilon, color='gray', linestyle=':', alpha=0.5)
        ax2.axvline(x=1+epsilon, color='gray', linestyle=':', alpha=0.5)
        ax2.axvline(x=1.0, color='black', linestyle='-', alpha=0.3)
        
        # Annotations
        ax2.annotate(f'1-ε = {1-epsilon}', xy=(1-epsilon, 0.3), fontsize=10, ha='center')
        ax2.annotate(f'1+ε = {1+epsilon}', xy=(1+epsilon, 0.3), fontsize=10, ha='center')
        ax2.annotate('Clipping prevents\ncomplete elimination', 
                    xy=(0.5, -1.0), fontsize=10, ha='center',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        ax2.set_xlabel('Probability Ratio r(θ)', fontsize=12)
        ax2.set_ylabel('Objective L(θ)', fontsize=12)
        ax2.set_title('Negative Advantage (A < 0)\n"Bad action - decrease probability"', fontsize=11)
        ax2.legend(loc='lower left')
        ax2.set_xlim([0, 2])
        ax2.set_ylim([-2.5, 0.5])
        ax2.grid(True, alpha=0.3)
        
        # Shade the effective region
        ax2.fill_between(r, ppo_neg, 0, where=(ppo_neg < 0), 
                        alpha=0.2, color='green')
        
        plt.tight_layout()
        return self._save_figure(fig, filename)
    
    # ================================================================
    # CLINICAL VISUALISATIONS
    # ================================================================
    
    def plot_glucose_profile(
        self,
        glucose_trace: List[float],
        insulin_trace: Optional[List[float]] = None,
        meal_times: Optional[List[float]] = None,
        sample_time_minutes: float = 5.0,
        target_range: Tuple[float, float] = (70, 180),
        title: str = "Glucose Profile",
        filename: str = "05_glucose_profile.png"
    ) -> str:
        """
        Plot a single episode glucose profile.
        
        Shows:
        - Glucose trace with target range
        - Insulin delivery (if provided)
        - Meal markers (if provided)
        
        Args:
            glucose_trace: Blood glucose values over time
            insulin_trace: Insulin doses over time
            meal_times: Times of meals (hours)
            sample_time_minutes: Time between samples
            target_range: Target glucose range (min, max)
            title: Figure title
            filename: Output filename
        
        Returns:
            Path to saved figure
        """
        n_samples = len(glucose_trace)
        time_hours = np.arange(n_samples) * sample_time_minutes / 60.0
        
        # Create figure with 2 subplots if insulin provided
        if insulin_trace is not None:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), 
                                           height_ratios=[3, 1], sharex=True)
        else:
            fig, ax1 = plt.subplots(1, 1, figsize=(12, 5))
        
        fig.suptitle(title, fontsize=14, fontweight='bold')
        
        # ---- Glucose plot ----
        glucose = np.array(glucose_trace)
        
        # Background zones
        ax1.axhspan(target_range[0], target_range[1], 
                   color=COLOURS['target_range'], alpha=0.3, label='Target range')
        ax1.axhspan(0, 70, color=COLOURS['hypo'], alpha=0.3, label='Hypoglycaemia')
        ax1.axhspan(180, 400, color=COLOURS['hyper'], alpha=0.3, label='Hyperglycaemia')
        
        # Glucose trace
        ax1.plot(time_hours, glucose, color=COLOURS['primary'], 
                linewidth=2, label='Blood glucose')
        
        # Target lines
        ax1.axhline(y=target_range[0], color=COLOURS['danger'], 
                   linestyle='--', alpha=0.7, linewidth=1)
        ax1.axhline(y=target_range[1], color=COLOURS['warning'], 
                   linestyle='--', alpha=0.7, linewidth=1)
        ax1.axhline(y=120, color=COLOURS['success'], 
                   linestyle=':', alpha=0.5, linewidth=1, label='Ideal (120)')
        
        # Meal markers
        if meal_times is not None:
            for meal_time in meal_times:
                ax1.axvline(x=meal_time, color=COLOURS['neutral'], 
                           linestyle='--', alpha=0.5)
                ax1.annotate('🍽️', xy=(meal_time, ax1.get_ylim()[1] * 0.95),
                           fontsize=12, ha='center')
        
        ax1.set_ylabel('Blood Glucose (mg/dL)', fontsize=12)
        ax1.set_ylim([40, 350])
        ax1.legend(loc='upper right', fontsize=9)
        ax1.grid(True, alpha=0.3)
        
        # Calculate and display metrics
        tir = np.mean((glucose >= 70) & (glucose <= 180)) * 100
        tbr = np.mean(glucose < 70) * 100
        tar = np.mean(glucose > 180) * 100
        
        metrics_text = f'TIR: {tir:.1f}%  |  TBR: {tbr:.1f}%  |  TAR: {tar:.1f}%'
        ax1.text(0.02, 0.98, metrics_text, transform=ax1.transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # ---- Insulin plot ----
        if insulin_trace is not None:
            insulin = np.array(insulin_trace)
            # Insulin trace is one shorter than glucose
            time_insulin = time_hours[1:len(insulin)+1]
            
            ax2.fill_between(time_insulin, 0, insulin, 
                           color=COLOURS['secondary'], alpha=0.5)
            ax2.plot(time_insulin, insulin, color=COLOURS['secondary'], linewidth=1)
            
            ax2.set_xlabel('Time (hours)', fontsize=12)
            ax2.set_ylabel('Insulin (U/hr)', fontsize=12)
            ax2.set_ylim([0, max(insulin) * 1.2 + 0.1])
            ax2.grid(True, alpha=0.3)
            
            # Total insulin
            total_insulin = np.sum(insulin) * sample_time_minutes / 60.0
            ax2.text(0.98, 0.95, f'Total: {total_insulin:.1f} U',
                    transform=ax2.transAxes, fontsize=10,
                    verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        else:
            ax1.set_xlabel('Time (hours)', fontsize=12)
        
        plt.tight_layout()
        return self._save_figure(fig, filename)
    
    def plot_clinical_metrics_comparison(
        self,
        results: List[Dict],
        metric_names: Optional[List[str]] = None,
        title: str = "Clinical Metrics Comparison",
        filename: str = "06_clinical_comparison.png"
    ) -> str:
        """
        Compare clinical metrics across multiple controllers.
        
        Args:
            results: List of result dictionaries with 'name' and metrics
            metric_names: Metrics to compare
            title: Figure title
            filename: Output filename
        
        Returns:
            Path to saved figure
        """
        if metric_names is None:
            metric_names = ['mean_tir', 'mean_tbr', 'mean_tar', 'mean_glucose']
        
        n_metrics = len(metric_names)
        n_controllers = len(results)
        
        fig, axes = plt.subplots(1, n_metrics, figsize=(4 * n_metrics, 5))
        fig.suptitle(title, fontsize=14, fontweight='bold')
        
        if n_metrics == 1:
            axes = [axes]
        
        labels_map = {
            'mean_tir': 'Time in Range (%)',
            'mean_tbr': 'Time Below Range (%)',
            'mean_tar': 'Time Above Range (%)',
            'mean_glucose': 'Mean Glucose (mg/dL)',
            'mean_cv': 'Glucose CV (%)',
            'mean_insulin': 'Total Insulin (U)',
            'mean_reward': 'Mean Reward'
        }
        
        colours_bar = plt.cm.Set2(np.linspace(0, 1, n_controllers))
        
        for idx, metric in enumerate(metric_names):
            ax = axes[idx]
            
            values = [r.get(metric, 0) for r in results]
            names = [r.get('name', f'Controller {i}') for i, r in enumerate(results)]
            
            bars = ax.bar(range(n_controllers), values, color=colours_bar)
            
            ax.set_xticks(range(n_controllers))
            ax.set_xticklabels(names, rotation=45, ha='right', fontsize=9)
            ax.set_ylabel(labels_map.get(metric, metric))
            ax.set_title(labels_map.get(metric, metric))
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add value labels on bars
            for bar, val in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                       f'{val:.1f}', ha='center', va='bottom', fontsize=9)
            
            # Add clinical targets
            if metric == 'mean_tir':
                ax.axhline(y=70, color=COLOURS['success'], linestyle='--',
                          alpha=0.7, label='Target (>70%)')
                ax.legend(fontsize=8)
            elif metric == 'mean_tbr':
                ax.axhline(y=4, color=COLOURS['danger'], linestyle='--',
                          alpha=0.7, label='Target (<4%)')
                ax.legend(fontsize=8)
        
        plt.tight_layout()
        return self._save_figure(fig, filename)
    
    # ================================================================
    # EXPERIMENT VISUALISATIONS
    # ================================================================
    
    def plot_epsilon_comparison(
        self,
        experiment_results: Dict,
        title: str = "Epsilon Comparison Experiment",
        filename: str = "07_epsilon_comparison.png"
    ) -> str:
        """
        Visualise epsilon comparison experiment results.
        
        Args:
            experiment_results: Dictionary with epsilon values as keys
            title: Figure title
            filename: Output filename
        
        Returns:
            Path to saved figure
        """
        fig, axes = plt.subplots(2, 2, figsize=LARGE_FIGSIZE)
        fig.suptitle(title, fontsize=14, fontweight='bold')
        
        epsilons = sorted(experiment_results.keys())
        colours_eps = plt.cm.viridis(np.linspace(0.2, 0.8, len(epsilons)))
        
        # ---- Plot 1: Learning curves ----
        ax1 = axes[0, 0]
        for eps, colour in zip(epsilons, colours_eps):
            data = experiment_results[eps]
            if 'episode_rewards' in data:
                rewards = data['episode_rewards']
                episodes = np.arange(len(rewards))
                
                # Smooth
                window = min(50, len(rewards) // 5)
                if window > 1:
                    smoothed = uniform_filter1d(rewards, size=window, mode='nearest')
                else:
                    smoothed = rewards
                
                ax1.plot(episodes, smoothed, color=colour, 
                        linewidth=2, label=f'ε = {eps}')
        
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Episode Reward')
        ax1.set_title('Learning Curves')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # ---- Plot 2: Final performance boxplot ----
        ax2 = axes[0, 1]
        final_rewards = []
        labels = []
        
        for eps in epsilons:
            data = experiment_results[eps]
            if 'episode_rewards' in data:
                rewards = data['episode_rewards']
                # Use last 20% of episodes
                n_final = max(10, len(rewards) // 5)
                final_rewards.append(rewards[-n_final:])
                labels.append(f'ε = {eps}')
        
        if final_rewards:
            bp = ax2.boxplot(final_rewards, labels=labels, patch_artist=True)
            for patch, colour in zip(bp['boxes'], colours_eps):
                patch.set_facecolor(colour)
                patch.set_alpha(0.6)
        
        ax2.set_ylabel('Final Episode Reward')
        ax2.set_title('Final Performance Distribution')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # ---- Plot 3: Time in Range comparison ----
        ax3 = axes[1, 0]
        tir_data = []
        
        for eps in epsilons:
            data = experiment_results[eps]
            if 'clinical_metrics' in data:
                tirs = [c.get('time_in_range', 0) for c in data['clinical_metrics']]
                n_final = max(10, len(tirs) // 5)
                tir_data.append(tirs[-n_final:])
            else:
                tir_data.append([0])
        
        if tir_data:
            bp = ax3.boxplot(tir_data, labels=labels, patch_artist=True)
            for patch, colour in zip(bp['boxes'], colours_eps):
                patch.set_facecolor(colour)
                patch.set_alpha(0.6)
        
        ax3.axhline(y=70, color=COLOURS['success'], linestyle='--',
                   alpha=0.7, label='Clinical target')
        ax3.set_ylabel('Time in Range (%)')
        ax3.set_title('Clinical Performance')
        ax3.legend(fontsize=8)
        ax3.grid(True, alpha=0.3, axis='y')
        
        # ---- Plot 4: Training stability ----
        ax4 = axes[1, 1]
        stability_data = []
        
        for eps in epsilons:
            data = experiment_results[eps]
            if 'episode_rewards' in data:
                rewards = np.array(data['episode_rewards'])
                # Compute rolling CV
                window = 50
                if len(rewards) > window:
                    rolling_std = np.array([
                        np.std(rewards[max(0, i-window):i+1])
                        for i in range(len(rewards))
                    ])
                    rolling_mean = np.array([
                        np.mean(rewards[max(0, i-window):i+1])
                        for i in range(len(rewards))
                    ])
                    cv = rolling_std / (np.abs(rolling_mean) + 1e-8)
                    stability_data.append(np.mean(cv[-len(cv)//5:]))
                else:
                    stability_data.append(np.std(rewards) / (np.abs(np.mean(rewards)) + 1e-8))
            else:
                stability_data.append(0)
        
        bars = ax4.bar(range(len(epsilons)), stability_data, color=colours_eps)
        ax4.set_xticks(range(len(epsilons)))
        ax4.set_xticklabels([f'ε = {eps}' for eps in epsilons])
        ax4.set_ylabel('Coefficient of Variation')
        ax4.set_title('Training Stability (lower = more stable)')
        ax4.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, val in zip(bars, stability_data):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f'{val:.3f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        return self._save_figure(fig, filename)
    
    def plot_summary_dashboard(
        self,
        episode_rewards: List[float],
        clinical_metrics: List[Dict],
        agent_metrics: Dict[str, List[float]],
        title: str = "PPO Glucose Control - Training Summary",
        filename: str = "08_summary_dashboard.png"
    ) -> str:
        """
        Create a single-page summary dashboard.
        
        Args:
            episode_rewards: Episode reward history
            clinical_metrics: Clinical metrics history
            agent_metrics: PPO training metrics
            title: Figure title
            filename: Output filename
        
        Returns:
            Path to saved figure
        """
        fig = plt.figure(figsize=(16, 12))
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        # Create grid
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        
        # ---- Large learning curve (top left, 2x2) ----
        ax1 = fig.add_subplot(gs[0:2, 0:2])
        
        rewards = np.array(episode_rewards)
        episodes = np.arange(len(rewards))
        window = min(50, len(rewards) // 5)
        
        ax1.plot(episodes, rewards, alpha=0.2, color=COLOURS['primary'])
        if window > 1:
            smoothed = uniform_filter1d(rewards, size=window, mode='nearest')
            ax1.plot(episodes, smoothed, color=COLOURS['primary'], linewidth=2)
        
        ax1.set_xlabel('Episode', fontsize=11)
        ax1.set_ylabel('Episode Reward', fontsize=11)
        ax1.set_title('Learning Curve', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # ---- Metrics box (top right) ----
        ax2 = fig.add_subplot(gs[0, 2:])
        ax2.axis('off')
        
        # Compute summary stats
        final_reward = np.mean(rewards[-50:]) if len(rewards) >= 50 else np.mean(rewards)
        final_std = np.std(rewards[-50:]) if len(rewards) >= 50 else np.std(rewards)
        final_tir = np.mean([c['time_in_range'] for c in clinical_metrics[-50:]]) if len(clinical_metrics) >= 50 else np.mean([c['time_in_range'] for c in clinical_metrics])
        final_tbr = np.mean([c['time_below_range'] for c in clinical_metrics[-50:]]) if len(clinical_metrics) >= 50 else np.mean([c['time_below_range'] for c in clinical_metrics])
        
        metrics_text = (
            f"Training Summary\n"
            f"{'─' * 30}\n"
            f"Total Episodes: {len(episode_rewards)}\n"
            f"Final Reward: {final_reward:.1f} ± {final_std:.1f}\n"
            f"Best Reward: {np.max(rewards):.1f}\n"
            f"{'─' * 30}\n"
            f"Clinical Metrics (final):\n"
            f"  Time in Range: {final_tir:.1f}%\n"
            f"  Time Below Range: {final_tbr:.1f}%\n"
            f"{'─' * 30}\n"
            f"Target TIR: >70%  ✓" if final_tir >= 70 else f"Target TIR: >70%  ✗"
        )
        
        ax2.text(0.1, 0.9, metrics_text, transform=ax2.transAxes,
                fontsize=11, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.3))
        
        # ---- Time in Range (middle right) ----
        ax3 = fig.add_subplot(gs[1, 2:])
        tirs = [c['time_in_range'] for c in clinical_metrics]
        ax3.plot(tirs, alpha=0.3, color=COLOURS['success'])
        if len(tirs) > window:
            ax3.plot(uniform_filter1d(tirs, size=window, mode='nearest'),
                    color=COLOURS['success'], linewidth=2)
        ax3.axhline(y=70, color=COLOURS['danger'], linestyle='--', alpha=0.7)
        ax3.set_ylabel('TIR (%)')
        ax3.set_title('Time in Range', fontsize=10)
        ax3.set_ylim([0, 100])
        ax3.grid(True, alpha=0.3)
        
        # ---- Clipping fraction (bottom left) ----
        ax4 = fig.add_subplot(gs[2, 0])
        if 'clipped_fraction' in agent_metrics:
            clipped = np.array(agent_metrics['clipped_fraction']) * 100
            ax4.plot(clipped, color=COLOURS['primary'], alpha=0.5)
            if len(clipped) > 10:
                ax4.plot(uniform_filter1d(clipped, size=10, mode='nearest'),
                        color=COLOURS['primary'], linewidth=2)
        ax4.set_ylabel('Clipped (%)')
        ax4.set_title('Clipping Activity', fontsize=10)
        ax4.grid(True, alpha=0.3)
        
        # ---- Policy loss (bottom) ----
        ax5 = fig.add_subplot(gs[2, 1])
        if 'policy_loss' in agent_metrics:
            ax5.plot(agent_metrics['policy_loss'], color=COLOURS['secondary'], alpha=0.5)
        ax5.set_ylabel('Loss')
        ax5.set_title('Policy Loss', fontsize=10)
        ax5.grid(True, alpha=0.3)
        
        # ---- Value loss (bottom) ----
        ax6 = fig.add_subplot(gs[2, 2])
        if 'value_loss' in agent_metrics:
            ax6.plot(agent_metrics['value_loss'], color=COLOURS['warning'], alpha=0.5)
        ax6.set_ylabel('Loss')
        ax6.set_title('Value Loss', fontsize=10)
        ax6.grid(True, alpha=0.3)
        
        # ---- Entropy (bottom right) ----
        ax7 = fig.add_subplot(gs[2, 3])
        if 'entropy' in agent_metrics:
            ax7.plot(agent_metrics['entropy'], color=COLOURS['neutral'], alpha=0.5)
        ax7.set_ylabel('Entropy')
        ax7.set_title('Policy Entropy', fontsize=10)
        ax7.grid(True, alpha=0.3)
        
        return self._save_figure(fig, filename)


def generate_all_visualisations(
    results_dir: str,
    output_dir: Optional[str] = None
) -> List[str]:
    """
    Generate all visualisations from saved training results.
    
    Args:
        results_dir: Directory containing training results
        output_dir: Directory to save figures (defaults to results_dir/figures)
    
    Returns:
        List of generated figure paths
    """
    if output_dir is None:
        output_dir = os.path.join(results_dir, "figures")
    
    # Load metrics
    metrics_path = os.path.join(results_dir, "metrics.json")
    if not os.path.exists(metrics_path):
        print(f"No metrics found at {metrics_path}")
        return []
    
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
    
    # Create visualiser
    vis = PPOVisualiser(save_dir=output_dir)
    
    print(f"\nGenerating visualisations from {results_dir}...")
    generated = []
    
    # Learning curves
    if 'episode_rewards' in metrics:
        path = vis.plot_learning_curves(
            episode_rewards=metrics['episode_rewards'],
            episode_timesteps=metrics.get('episode_timesteps'),
            clinical_metrics=metrics.get('clinical_metrics')
        )
        generated.append(path)
    
    # Training metrics
    if 'agent_metrics' in metrics:
        path = vis.plot_training_metrics(metrics['agent_metrics'])
        if path:
            generated.append(path)
        
        path = vis.plot_clipping_behaviour(metrics['agent_metrics'])
        generated.append(path)
    
    # Clipped objective explanation
    path = vis.plot_clipped_objective_explanation()
    generated.append(path)
    
    # Summary dashboard
    if all(k in metrics for k in ['episode_rewards', 'clinical_metrics', 'agent_metrics']):
        path = vis.plot_summary_dashboard(
            episode_rewards=metrics['episode_rewards'],
            clinical_metrics=metrics['clinical_metrics'],
            agent_metrics=metrics['agent_metrics']
        )
        generated.append(path)
    
    print(f"\nGenerated {len(generated)} figures in {output_dir}")
    return generated


def main():
    """Test visualisation functions."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate visualisations")
    parser.add_argument(
        "--results-dir", type=str, required=True,
        help="Directory containing training results"
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory for figures"
    )
    
    args = parser.parse_args()
    
    generate_all_visualisations(args.results_dir, args.output_dir)


if __name__ == "__main__":
    # If run directly, generate example plots
    print("Generating example visualisations...")
    
    vis = PPOVisualiser(save_dir="./test_figures")
    
    # Generate clipped objective explanation
    vis.plot_clipped_objective_explanation()
    
    # Generate fake data for testing
    n_episodes = 500
    fake_rewards = np.cumsum(np.random.randn(n_episodes) * 5) + np.linspace(0, 200, n_episodes)
    fake_clinical = [
        {'time_in_range': 40 + i/n_episodes * 40 + np.random.randn() * 5,
         'time_below_range': max(0, 10 - i/n_episodes * 8 + np.random.randn()),
         'time_above_range': max(0, 50 - i/n_episodes * 35 + np.random.randn() * 3),
         'mean_glucose': 160 - i/n_episodes * 40 + np.random.randn() * 10,
         'glucose_cv': 30 - i/n_episodes * 10 + np.random.randn() * 2,
         'total_insulin': 20 + np.random.randn() * 2}
        for i in range(n_episodes)
    ]
    
    vis.plot_learning_curves(
        episode_rewards=fake_rewards.tolist(),
        clinical_metrics=fake_clinical
    )
    
    print("\nTest visualisations complete!")