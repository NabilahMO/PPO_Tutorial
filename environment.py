"""
Glucose-Insulin Environment for Reinforcement Learning
=======================================================

STABILITY-FIXED VERSION - Rewards scaled for stable training.

A Gymnasium environment simulating Type 1 Diabetes glucose dynamics
for training RL agents to control insulin delivery.

Based on:
- Bergman Minimal Model for glucose-insulin dynamics
- Clinical reward functions from Zhu et al. (2020)

Author: [Your Name]
Date: December 2024
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, List, Optional, Tuple, Any


class GlucoseInsulinEnv(gym.Env):
    """
    Glucose-Insulin Control Environment.
    
    Simulates a Type 1 Diabetes patient over 24 hours.
    """
    
    metadata = {'render_modes': ['human']}
    
    def __init__(
        self,
        max_insulin_dose: float = 5.0,
        episode_length_hours: float = 24.0,
        sample_time_minutes: float = 5.0,
        target_glucose_min: float = 70.0,
        target_glucose_max: float = 180.0,
        initial_glucose_mean: float = 120.0,
        initial_glucose_std: float = 20.0,
        patient_variability: bool = True,
        meal_variability: bool = True,
        seed: Optional[int] = None
    ):
        super().__init__()
        
        self.max_insulin_dose = max_insulin_dose
        self.episode_length_hours = episode_length_hours
        self.sample_time_minutes = sample_time_minutes
        self.sample_time_hours = sample_time_minutes / 60.0
        self.max_steps = int(episode_length_hours * 60 / sample_time_minutes)
        
        self.target_min = target_glucose_min
        self.target_max = target_glucose_max
        self.initial_glucose_mean = initial_glucose_mean
        self.initial_glucose_std = initial_glucose_std
        self.patient_variability = patient_variability
        self.meal_variability = meal_variability
        
        self.action_space = spaces.Box(
            low=0.0, high=max_insulin_dose, shape=(1,), dtype=np.float32
        )
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32
        )
        
        self._np_random = None
        self.seed_value = seed
        self._initialise_state_variables()
    
    def _initialise_state_variables(self) -> None:
        self.glucose = self.initial_glucose_mean
        self.glucose_history = [self.glucose, self.glucose, self.glucose]
        self.plasma_insulin = 10.0
        self.insulin_action = 0.0
        self.insulin_on_board = 0.0
        self.current_cho = 0.0
        self.time_since_meal = 6.0
        self.current_step = 0
        self.current_time = 0.0
        self.glucose_trace = []
        self.insulin_trace = []
        self.total_reward = 0.0
        self.patient_params = {}
        self.meal_schedule = []
    
    def _generate_patient_parameters(self) -> Dict[str, float]:
        if self.patient_variability:
            weight = self._np_random.uniform(50, 100)
            insulin_sensitivity = self._np_random.uniform(0.7, 1.3)
        else:
            weight = 75.0
            insulin_sensitivity = 1.0
        
        params = {
            'weight': weight,
            'insulin_sensitivity': insulin_sensitivity,
            'p1': 0.028 * (self._np_random.uniform(0.8, 1.2) if self.patient_variability else 1.0),
            'p2': 0.015 * (self._np_random.uniform(0.8, 1.2) if self.patient_variability else 1.0),
            'p3': 8.5e-6 * insulin_sensitivity,
            'n': 0.14 * (self._np_random.uniform(0.8, 1.2) if self.patient_variability else 1.0),
            'Gb': 110.0,
            'Ib': 10.0,
            'Vg': 1.6 * weight,
            'Vi': 0.05 * weight,
            'tau_iob': self._np_random.uniform(3.0, 5.0) if self.patient_variability else 4.0,
        }
        return params
    
    def _generate_meal_schedule(self) -> List[Dict[str, float]]:
        if self.meal_variability:
            meals = [
                {'time': self._np_random.uniform(6.5, 7.5), 'cho': self._np_random.uniform(40, 60)},
                {'time': self._np_random.uniform(11.5, 12.5), 'cho': self._np_random.uniform(55, 85)},
                {'time': self._np_random.uniform(17.5, 18.5), 'cho': self._np_random.uniform(65, 95)}
            ]
        else:
            meals = [
                {'time': 7.0, 'cho': 50},
                {'time': 12.0, 'cho': 70},
                {'time': 18.0, 'cho': 80}
            ]
        return sorted(meals, key=lambda x: x['time'])
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        
        if seed is not None:
            self._np_random = np.random.default_rng(seed)
        elif self._np_random is None:
            self._np_random = np.random.default_rng(self.seed_value)
        
        self.patient_params = self._generate_patient_parameters()
        self.meal_schedule = self._generate_meal_schedule()
        
        self.glucose = np.clip(
            self._np_random.normal(self.initial_glucose_mean, self.initial_glucose_std),
            70, 200
        )
        self.glucose_history = [self.glucose, self.glucose, self.glucose]
        self.plasma_insulin = self.patient_params['Ib']
        self.insulin_action = 0.0
        self.insulin_on_board = 0.0
        self.current_cho = 0.0
        self.time_since_meal = 6.0
        self.current_step = 0
        self.current_time = 0.0
        self.glucose_trace = [self.glucose]
        self.insulin_trace = []
        self.total_reward = 0.0
        
        return self._get_observation(), self._get_info()
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        insulin_dose = float(np.clip(action[0], 0.0, self.max_insulin_dose))
        
        self._process_meals()
        self._simulate_dynamics(insulin_dose)
        
        self.current_step += 1
        self.current_time += self.sample_time_hours
        self.time_since_meal += self.sample_time_hours
        
        reward = self._compute_reward()
        self.total_reward += reward
        
        self.glucose_history.append(self.glucose)
        if len(self.glucose_history) > 3:
            self.glucose_history.pop(0)
        
        self.glucose_trace.append(self.glucose)
        self.insulin_trace.append(insulin_dose)
        
        terminated = self.glucose < 40
        truncated = self.current_step >= self.max_steps
        
        return self._get_observation(), reward, terminated, truncated, self._get_info()
    
    def _process_meals(self) -> None:
        self.current_cho = 0.0
        for meal in self.meal_schedule:
            if abs(self.current_time - meal['time']) < self.sample_time_hours / 2:
                self.current_cho = meal['cho']
                self.time_since_meal = 0.0
                break
    
    def _simulate_dynamics(self, insulin_dose: float) -> None:
        params = self.patient_params
        dt = self.sample_time_minutes
        insulin_rate = insulin_dose * 1000 / 60
        
        if self.current_cho > 0:
            meal_rate = self.current_cho * 1000 / params['Vg'] / 120
        else:
            meal_rate = 0.0
        
        n_substeps = 10
        sub_dt = dt / n_substeps
        
        G = self.glucose
        X = self.insulin_action
        I = self.plasma_insulin
        
        for _ in range(n_substeps):
            dG = -params['p1'] * (G - params['Gb']) - X * G + meal_rate
            dX = -params['p2'] * X + params['p3'] * (I - params['Ib'])
            dI = -params['n'] * (I - params['Ib']) + insulin_rate / params['Vi']
            
            G = max(G + dG * sub_dt, 1.0)
            X = max(X + dX * sub_dt, 0.0)
            I = max(I + dI * sub_dt, 0.0)
        
        self.glucose = G
        self.insulin_action = X
        self.plasma_insulin = I
        
        tau = params['tau_iob']
        decay = np.exp(-self.sample_time_hours / tau)
        self.insulin_on_board = self.insulin_on_board * decay + insulin_dose * self.sample_time_hours
        
        self.glucose = max(self.glucose + self._np_random.normal(0, 1.0), 1.0)
    
    def _compute_reward(self) -> float:
        """
        Compute reward - SCALED DOWN by 10x for training stability.
        """
        G = self.glucose
        
        # Zone-based rewards (SCALED by 0.1)
        if G < 54:
            base_reward = -0.4      # Severe hypo
        elif G < 70:
            base_reward = -0.2      # Hypo
        elif G <= 180:
            base_reward = 0.1       # Target range
        elif G <= 250:
            base_reward = -0.05     # Mild hyper
        elif G <= 400:
            base_reward = -0.1      # Hyper
        else:
            base_reward = -0.2      # Severe hyper
        
        # Centre bonus
        if 100 <= G <= 140:
            base_reward += 0.05
        
        # Rate penalty
        if len(self.glucose_history) >= 2:
            dG = abs(self.glucose - self.glucose_history[-2])
            base_reward -= 0.001 * max(0, dG - 2)
        
        # IOB penalty
        base_reward -= 0.0001 * self.insulin_on_board
        
        # Severe hypo termination penalty
        if G < 40:
            base_reward -= 1.0
        
        return base_reward
    
    def _get_observation(self) -> np.ndarray:
        return np.array([
            self.glucose / 200.0,
            self.glucose_history[-2] / 200.0,
            self.glucose_history[-3] / 200.0,
            self.plasma_insulin / 50.0,
            self.insulin_on_board / 10.0,
            self.current_cho / 100.0,
            self.time_since_meal / 6.0,
            self.patient_params['weight'] / 100.0
        ], dtype=np.float32)
    
    def _get_info(self) -> Dict[str, Any]:
        return {
            'glucose': self.glucose,
            'plasma_insulin': self.plasma_insulin,
            'insulin_on_board': self.insulin_on_board,
            'current_cho': self.current_cho,
            'time': self.current_time,
            'step': self.current_step,
            'patient_weight': self.patient_params.get('weight', 75),
            'patient_insulin_sensitivity': self.patient_params.get('insulin_sensitivity', 1.0),
            'meal_schedule': self.meal_schedule
        }
    
    def get_episode_stats(self) -> Dict[str, float]:
        if len(self.glucose_trace) == 0:
            return {}
        
        glucose = np.array(self.glucose_trace)
        n_samples = len(glucose)
        
        tir = np.sum((glucose >= 70) & (glucose <= 180)) / n_samples * 100
        tbr = np.sum(glucose < 70) / n_samples * 100
        tbr_severe = np.sum(glucose < 54) / n_samples * 100
        tar = np.sum(glucose > 180) / n_samples * 100
        tar_severe = np.sum(glucose > 250) / n_samples * 100
        
        mean_glucose = np.mean(glucose)
        std_glucose = np.std(glucose)
        cv = (std_glucose / mean_glucose * 100) if mean_glucose > 0 else 0
        
        total_insulin = np.sum(self.insulin_trace) * self.sample_time_hours if self.insulin_trace else 0
        
        return {
            'time_in_range': tir,
            'time_below_range': tbr,
            'time_below_range_severe': tbr_severe,
            'time_above_range': tar,
            'time_above_range_severe': tar_severe,
            'mean_glucose': mean_glucose,
            'std_glucose': std_glucose,
            'glucose_cv': cv,
            'total_insulin': total_insulin,
            'total_reward': self.total_reward,
            'num_steps': len(glucose)
        }
    
    def render(self, mode: str = 'human') -> None:
        print(f"Step {self.current_step:3d} | Time {self.current_time:5.1f}h | "
              f"Glucose {self.glucose:6.1f} mg/dL | IOB {self.insulin_on_board:4.2f} U")


def test_environment():
    """Test the environment."""
    print("Testing Glucose-Insulin Environment...")
    print("=" * 60)
    
    env = GlucoseInsulinEnv(patient_variability=True, meal_variability=True, seed=42)
    state, info = env.reset(seed=42)
    
    print(f"Initial state shape: {state.shape}")
    print(f"Patient weight: {info['patient_weight']:.1f} kg")
    
    print("\nRunning 24h simulation with fixed basal (1.0 U/hr)...")
    
    total_reward = 0
    done = False
    step = 0
    
    while not done:
        action = np.array([1.0])
        state, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated
        if step % 48 == 0:
            env.render()
        step += 1
    
    stats = env.get_episode_stats()
    print(f"\nEpisode Statistics:")
    print(f"  Total reward: {stats['total_reward']:.2f}")
    print(f"  Time in Range: {stats['time_in_range']:.1f}%")
    print(f"  Time Below Range: {stats['time_below_range']:.1f}%")
    print(f"  Mean glucose: {stats['mean_glucose']:.1f} mg/dL")
    print("\n✓ Environment test complete!")


if __name__ == "__main__":
    test_environment()