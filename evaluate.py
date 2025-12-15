"""
Evaluation and Baselines for Glucose Control
=============================================

Provides:
- Evaluation of trained PPO agents
- Baseline controllers for comparison:
  - Fixed basal rate
  - Standard basal-bolus therapy
  - Simple rule-based controller
  - PID controller (optional)
- Clinical metric computation
- Comparison visualisations

Author: [Your Name]
Date: December 2024
"""

import os
import json
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from environment import GlucoseInsulinEnv
from ppo_agent import PPOAgent


@dataclass
class EvaluationResult:
    """Container for evaluation results."""
    name: str
    rewards: List[float]
    clinical_metrics: List[Dict]
    glucose_traces: List[List[float]]
    insulin_traces: List[List[float]]
    
    @property
    def mean_reward(self) -> float:
        return np.mean(self.rewards)
    
    @property
    def std_reward(self) -> float:
        return np.std(self.rewards)
    
    @property
    def mean_tir(self) -> float:
        return np.mean([c['time_in_range'] for c in self.clinical_metrics])
    
    @property
    def mean_tbr(self) -> float:
        return np.mean([c['time_below_range'] for c in self.clinical_metrics])
    
    @property
    def mean_tar(self) -> float:
        return np.mean([c['time_above_range'] for c in self.clinical_metrics])
    
    def summary(self) -> Dict:
        """Return summary statistics."""
        return {
            'name': self.name,
            'mean_reward': self.mean_reward,
            'std_reward': self.std_reward,
            'mean_tir': self.mean_tir,
            'std_tir': np.std([c['time_in_range'] for c in self.clinical_metrics]),
            'mean_tbr': self.mean_tbr,
            'mean_tar': self.mean_tar,
            'mean_glucose': np.mean([c['mean_glucose'] for c in self.clinical_metrics]),
            'mean_cv': np.mean([c['glucose_cv'] for c in self.clinical_metrics]),
            'mean_insulin': np.mean([c['total_insulin'] for c in self.clinical_metrics]),
            'num_episodes': len(self.rewards)
        }


class BaselineController:
    """Base class for baseline controllers."""
    
    def __init__(self, name: str):
        self.name = name
    
    def get_action(self, state: np.ndarray, info: Dict = None) -> np.ndarray:
        """
        Get control action given current state.
        
        Args:
            state: Current observation
            info: Additional information (e.g., glucose value)
        
        Returns:
            Action (insulin dose)
        """
        raise NotImplementedError
    
    def reset(self) -> None:
        """Reset controller state (if any)."""
        pass


class FixedBasalController(BaselineController):
    """
    Fixed basal rate controller.
    
    Delivers a constant insulin infusion rate regardless
    of glucose levels. Simple but inflexible baseline.
    
    Typical basal rates: 0.5-2.0 U/hr for adults
    """
    
    def __init__(self, basal_rate: float = 1.0):
        """
        Args:
            basal_rate: Fixed insulin rate (U/hr)
        """
        super().__init__(name=f"Fixed Basal ({basal_rate} U/hr)")
        self.basal_rate = basal_rate
    
    def get_action(self, state: np.ndarray, info: Dict = None) -> np.ndarray:
        return np.array([self.basal_rate])


class BasalBolusController(BaselineController):
    """
    Standard basal-bolus therapy controller.
    
    Combines:
    - Constant basal rate
    - Meal boluses based on carbohydrate intake
    
    Uses insulin-to-carb ratio (ICR) for bolus calculation.
    """
    
    def __init__(
        self,
        basal_rate: float = 0.8,
        insulin_to_carb_ratio: float = 10.0,
        correction_factor: float = 50.0,
        target_glucose: float = 120.0
    ):
        """
        Args:
            basal_rate: Basal insulin rate (U/hr)
            insulin_to_carb_ratio: Grams of CHO covered by 1 unit insulin
            correction_factor: mg/dL drop per unit of insulin
            target_glucose: Target glucose for corrections (mg/dL)
        """
        super().__init__(name="Basal-Bolus Therapy")
        self.basal_rate = basal_rate
        self.icr = insulin_to_carb_ratio
        self.cf = correction_factor
        self.target = target_glucose
        
        # Track pending bolus
        self.pending_bolus = 0.0
    
    def get_action(self, state: np.ndarray, info: Dict = None) -> np.ndarray:
        # Base action is basal rate
        action = self.basal_rate
        
        # Add any pending bolus
        if self.pending_bolus > 0:
            action += self.pending_bolus
            self.pending_bolus = 0.0
        
        # Check for meal (if info provided)
        if info is not None and info.get('cho_intake', 0) > 0:
            cho = info['cho_intake']
            # Calculate meal bolus
            meal_bolus = cho / self.icr
            
            # Add correction bolus if glucose is high
            glucose = info.get('glucose', self.target)
            if glucose > self.target + 50:
                correction = (glucose - self.target) / self.cf
                meal_bolus += correction
            
            # Add to pending (will be delivered next step)
            self.pending_bolus = meal_bolus
        
        return np.array([action])
    
    def reset(self) -> None:
        self.pending_bolus = 0.0


class RuleBasedController(BaselineController):
    """
    Simple rule-based glucose controller.
    
    Adjusts insulin based on current glucose level:
    - High glucose → increase insulin
    - Low glucose → decrease insulin
    - Target range → maintain basal
    
    More reactive than fixed basal but still simple.
    """
    
    def __init__(
        self,
        basal_rate: float = 1.0,
        target_glucose: float = 120.0,
        gain: float = 0.02,
        max_adjustment: float = 1.0
    ):
        """
        Args:
            basal_rate: Base insulin rate (U/hr)
            target_glucose: Target glucose level (mg/dL)
            gain: Proportional gain for adjustment
            max_adjustment: Maximum adjustment from basal (U/hr)
        """
        super().__init__(name="Rule-Based Controller")
        self.basal_rate = basal_rate
        self.target = target_glucose
        self.gain = gain
        self.max_adjustment = max_adjustment
    
    def get_action(self, state: np.ndarray, info: Dict = None) -> np.ndarray:
        # Extract glucose from state (first element, denormalised)
        glucose = state[0] * 200.0  # Denormalise
        
        # Calculate error
        error = glucose - self.target
        
        # Proportional adjustment
        adjustment = self.gain * error
        adjustment = np.clip(adjustment, -self.max_adjustment, self.max_adjustment)
        
        # Safety: reduce insulin if glucose is low
        if glucose < 80:
            adjustment = -self.max_adjustment
        elif glucose < 100:
            adjustment = min(adjustment, 0)
        
        # Calculate final action
        action = self.basal_rate + adjustment
        action = np.clip(action, 0.0, 5.0)  # Safety limits
        
        return np.array([action])


class PIDController(BaselineController):
    """
    PID (Proportional-Integral-Derivative) controller.
    
    Classic control theory approach:
    u(t) = Kp*e(t) + Ki*∫e(τ)dτ + Kd*de/dt
    
    Where e(t) = glucose(t) - target
    
    More sophisticated than rule-based but requires tuning.
    """
    
    def __init__(
        self,
        basal_rate: float = 1.0,
        target_glucose: float = 120.0,
        kp: float = 0.02,
        ki: float = 0.001,
        kd: float = 0.1,
        integral_limit: float = 50.0
    ):
        """
        Args:
            basal_rate: Base insulin rate (U/hr)
            target_glucose: Target glucose level (mg/dL)
            kp: Proportional gain
            ki: Integral gain
            kd: Derivative gain
            integral_limit: Anti-windup limit for integral term
        """
        super().__init__(name="PID Controller")
        self.basal_rate = basal_rate
        self.target = target_glucose
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral_limit = integral_limit
        
        # Controller state
        self.integral = 0.0
        self.prev_error = 0.0
    
    def get_action(self, state: np.ndarray, info: Dict = None) -> np.ndarray:
        # Extract glucose from state
        glucose = state[0] * 200.0  # Denormalise
        
        # Calculate error
        error = glucose - self.target
        
        # Proportional term
        p_term = self.kp * error
        
        # Integral term (with anti-windup)
        self.integral += error
        self.integral = np.clip(self.integral, -self.integral_limit, self.integral_limit)
        i_term = self.ki * self.integral
        
        # Derivative term
        d_term = self.kd * (error - self.prev_error)
        self.prev_error = error
        
        # PID output
        pid_output = p_term + i_term + d_term
        
        # Safety: reduce insulin if glucose is low
        if glucose < 70:
            pid_output = min(pid_output, -self.basal_rate * 0.5)
            self.integral = 0  # Reset integral
        
        # Calculate final action
        action = self.basal_rate + pid_output
        action = np.clip(action, 0.0, 5.0)
        
        return np.array([action])
    
    def reset(self) -> None:
        self.integral = 0.0
        self.prev_error = 0.0


def evaluate_controller(
    controller: BaselineController,
    env_config: Dict,
    num_episodes: int = 10,
    seed: int = 42
) -> EvaluationResult:
    """
    Evaluate a controller on the glucose environment.
    
    Args:
        controller: Controller to evaluate
        env_config: Environment configuration
        num_episodes: Number of evaluation episodes
        seed: Random seed
    
    Returns:
        EvaluationResult with all metrics
    """
    env = GlucoseInsulinEnv(**env_config, seed=seed)
    
    rewards = []
    clinical_metrics = []
    glucose_traces = []
    insulin_traces = []
    
    for episode in range(num_episodes):
        state, info = env.reset(seed=seed + episode)
        controller.reset()
        
        episode_reward = 0.0
        done = False
        
        while not done:
            # Get action from controller
            action = controller.get_action(state, info)
            
            # Take step
            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward
        
        rewards.append(episode_reward)
        clinical_metrics.append(env.get_episode_stats())
        glucose_traces.append(env.glucose_trace.copy())
        insulin_traces.append(env.insulin_trace.copy())
    
    return EvaluationResult(
        name=controller.name,
        rewards=rewards,
        clinical_metrics=clinical_metrics,
        glucose_traces=glucose_traces,
        insulin_traces=insulin_traces
    )


def evaluate_ppo_agent(
    agent: PPOAgent,
    env_config: Dict,
    num_episodes: int = 10,
    deterministic: bool = True,
    seed: int = 42
) -> EvaluationResult:
    """
    Evaluate a trained PPO agent.
    
    Args:
        agent: Trained PPO agent
        env_config: Environment configuration
        num_episodes: Number of evaluation episodes
        deterministic: Use deterministic policy
        seed: Random seed
    
    Returns:
        EvaluationResult with all metrics
    """
    env = GlucoseInsulinEnv(**env_config, seed=seed)
    
    rewards = []
    clinical_metrics = []
    glucose_traces = []
    insulin_traces = []
    
    for episode in range(num_episodes):
        state, _ = env.reset(seed=seed + episode)
        
        episode_reward = 0.0
        done = False
        
        while not done:
            action, _, _ = agent.get_action(state, deterministic=deterministic)
            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward
        
        rewards.append(episode_reward)
        clinical_metrics.append(env.get_episode_stats())
        glucose_traces.append(env.glucose_trace.copy())
        insulin_traces.append(env.insulin_trace.copy())
    
    return EvaluationResult(
        name="PPO Agent",
        rewards=rewards,
        clinical_metrics=clinical_metrics,
        glucose_traces=glucose_traces,
        insulin_traces=insulin_traces
    )


def compare_controllers(
    results: List[EvaluationResult],
    save_path: Optional[str] = None
) -> Dict:
    """
    Compare multiple controllers and generate summary.
    
    Args:
        results: List of EvaluationResult objects
        save_path: Optional path to save comparison
    
    Returns:
        Comparison dictionary
    """
    comparison = {
        'controllers': [r.name for r in results],
        'summaries': [r.summary() for r in results]
    }
    
    # Print comparison table
    print("\n" + "=" * 80)
    print("Controller Comparison")
    print("=" * 80)
    print(f"{'Controller':<25} {'Reward':>10} {'TIR (%)':>10} {'TBR (%)':>10} {'TAR (%)':>10}")
    print("-" * 80)
    
    for result in results:
        s = result.summary()
        print(f"{s['name']:<25} {s['mean_reward']:>10.1f} {s['mean_tir']:>10.1f} "
              f"{s['mean_tbr']:>10.1f} {s['mean_tar']:>10.1f}")
    
    print("=" * 80)
    
    # Find best controller for each metric
    best_reward = max(results, key=lambda r: r.mean_reward)
    best_tir = max(results, key=lambda r: r.mean_tir)
    lowest_tbr = min(results, key=lambda r: r.mean_tbr)
    
    print(f"\nBest reward: {best_reward.name} ({best_reward.mean_reward:.1f})")
    print(f"Best TIR: {best_tir.name} ({best_tir.mean_tir:.1f}%)")
    print(f"Lowest TBR: {lowest_tbr.name} ({lowest_tbr.mean_tbr:.1f}%)")
    
    if save_path:
        with open(save_path, 'w') as f:
            json.dump(comparison, f, indent=2)
        print(f"\nComparison saved to: {save_path}")
    
    return comparison


def load_and_evaluate_agent(
    model_path: str,
    env_config: Dict,
    num_episodes: int = 10,
    seed: int = 42
) -> EvaluationResult:
    """
    Load a saved PPO agent and evaluate it.
    
    Args:
        model_path: Path to saved model
        env_config: Environment configuration
        num_episodes: Number of evaluation episodes
        seed: Random seed
    
    Returns:
        EvaluationResult
    """
    # Determine state/action dimensions from env
    temp_env = GlucoseInsulinEnv(**env_config)
    state_dim = temp_env.observation_space.shape[0]
    action_dim = temp_env.action_space.shape[0]
    
    # Create and load agent
    agent = PPOAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        action_low=0.0,
        action_high=env_config.get('max_insulin_dose', 5.0)
    )
    agent.load(model_path)
    
    # Evaluate
    return evaluate_ppo_agent(agent, env_config, num_episodes, seed=seed)


def run_baseline_comparison(
    model_path: Optional[str] = None,
    env_config: Optional[Dict] = None,
    num_episodes: int = 10,
    seed: int = 42,
    save_dir: Optional[str] = None
) -> Dict:
    """
    Run comparison between PPO agent and baselines.
    
    Args:
        model_path: Path to trained PPO model (optional)
        env_config: Environment configuration
        num_episodes: Episodes per controller
        seed: Random seed
        save_dir: Directory to save results
    
    Returns:
        Comparison results
    """
    # Default environment config
    if env_config is None:
        env_config = {
            'max_insulin_dose': 5.0,
            'episode_length_hours': 24.0,
            'sample_time_minutes': 5.0,
            'patient_variability': True,
            'meal_variability': True
        }
    
    results = []
    
    # Baseline controllers
    print("\nEvaluating baseline controllers...")
    
    baselines = [
        FixedBasalController(basal_rate=1.0),
        BasalBolusController(basal_rate=0.8, insulin_to_carb_ratio=10.0),
        RuleBasedController(basal_rate=1.0, target_glucose=120.0),
        PIDController(basal_rate=1.0, target_glucose=120.0)
    ]
    
    for controller in baselines:
        print(f"  Evaluating {controller.name}...")
        result = evaluate_controller(controller, env_config, num_episodes, seed)
        results.append(result)
    
    # PPO agent (if model provided)
    if model_path is not None:
        print(f"  Evaluating PPO Agent...")
        ppo_result = load_and_evaluate_agent(model_path, env_config, num_episodes, seed)
        results.append(ppo_result)
    
    # Compare
    save_path = None
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "comparison.json")
    
    comparison = compare_controllers(results, save_path)
    
    return {
        'results': results,
        'comparison': comparison
    }


def main():
    """Main evaluation script."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate glucose controllers")
    parser.add_argument(
        "--model", type=str, default=None,
        help="Path to trained PPO model"
    )
    parser.add_argument(
        "--episodes", type=int, default=10,
        help="Number of evaluation episodes"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--save-dir", type=str, default="./results/evaluation",
        help="Directory to save results"
    )
    
    args = parser.parse_args()
    
    # Run comparison
    run_baseline_comparison(
        model_path=args.model,
        num_episodes=args.episodes,
        seed=args.seed,
        save_dir=args.save_dir
    )


if __name__ == "__main__":
    main()