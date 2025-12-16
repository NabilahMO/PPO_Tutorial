"""
Glucose-Insulin Control Environment for PPO
=============================================

Simulates Type 1 Diabetes glucose dynamics for reinforcement learning.

Based on:
- Bergman Minimal Model for glucose-insulin dynamics
- Zhu et al. (2020) reward function design
- FDA UVA/Padova simulator patterns

This environment models:
- Glucose absorption from meals
- Insulin pharmacokinetics and pharmacodynamics
- Patient variability through physiological parameters
- Realistic meal schedules and disturbances
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Tuple, Optional, List


class GlucoseInsulinEnv(gym.Env):
    """
    Reinforcement Learning Environment for Glucose Control in Type 1 Diabetes.
    
    The agent learns to dose insulin to maintain blood glucose within
    the target range (70-180 mg/dL) whilst avoiding dangerous
    hypoglycaemia (< 70 mg/dL) and hyperglycaemia (> 180 mg/dL).
    
    State Space (8 dimensions):
        - Current glucose (mg/dL)
        - Previous glucose (mg/dL) 
        - Glucose 2 steps ago (mg/dL)
        - Plasma insulin (mU/L)
        - Insulin-on-board (U)
        - Recent carbohydrate intake (g)
        - Time since last meal (hours)
        - Patient weight (kg)
    
    Action Space:
        - Continuous insulin dose [0, max_dose] U/hr
    
    Reward Function (Zhu et al., 2020):
        - Zone-based rewards with asymmetric penalties
        - Hypoglycaemia penalised 2-4x more than hyperglycaemia
    """
    
    metadata = {'render_modes': ['human', 'rgb_array']}
    
    def __init__(
        self,
        max_insulin_dose: float = 5.0,      # Maximum insulin dose (U/hr)
        episode_length_hours: float = 24.0,  # Episode duration (hours)
        sample_time_minutes: float = 5.0,    # Decision interval (minutes)
        target_glucose_min: float = 70.0,    # Target range lower bound (mg/dL)
        target_glucose_max: float = 180.0,   # Target range upper bound (mg/dL)
        patient_variability: bool = True,    # Enable inter-patient variability
        meal_variability: bool = True,       # Enable meal timing/size variability
        seed: Optional[int] = None,
        render_mode: Optional[str] = None
    ):
        """
        Initialise the Glucose-Insulin environment.
        
        Args:
            max_insulin_dose: Maximum allowable insulin infusion rate (U/hr)
            episode_length_hours: Duration of one episode in hours
            sample_time_minutes: Time between decisions in minutes
            target_glucose_min: Lower bound of target glucose range (mg/dL)
            target_glucose_max: Upper bound of target glucose range (mg/dL)
            patient_variability: If True, randomise patient parameters
            meal_variability: If True, randomise meal times and sizes
            seed: Random seed for reproducibility
            render_mode: Rendering mode ('human' or 'rgb_array')
        """
        super().__init__()
        
        # Environment parameters
        self.max_insulin_dose = max_insulin_dose
        self.episode_length_hours = episode_length_hours
        self.sample_time = sample_time_minutes / 60.0  # Convert to hours
        self.target_glucose_min = target_glucose_min
        self.target_glucose_max = target_glucose_max
        self.patient_variability = patient_variability
        self.meal_variability = meal_variability
        self.render_mode = render_mode
        
        # Calculate episode steps
        self.max_steps = int(episode_length_hours * 60 / sample_time_minutes)
        
        # Random number generator
        self.np_random = np.random.default_rng(seed)
        
        # Define action space: continuous insulin dose [0, max_dose]
        self.action_space = spaces.Box(
            low=0.0,
            high=max_insulin_dose,
            shape=(1,),
            dtype=np.float32
        )
        
        # Define observation space (8 dimensions)
        # All values normalised to reasonable ranges for neural network
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(8,),
            dtype=np.float32
        )
        
        # Physiological constants (will be personalised per patient)
        self.patient_params = None
        
        # State variables
        self.glucose = 0.0
        self.glucose_history = []
        self.insulin = 0.0
        self.insulin_action = 0.0
        self.iob = 0.0  # Insulin-on-board
        self.cho_absorbed = 0.0
        self.time_since_meal = 0.0
        self.current_step = 0
        self.current_time = 0.0  # Hours from midnight
        
        # Meal schedule
        self.meals = []
        self.meal_index = 0
        
        # Episode tracking
        self.glucose_trace = []
        self.insulin_trace = []
        self.reward_trace = []
        self.meal_trace = []
    
    def _generate_patient_parameters(self) -> Dict[str, float]:
        """
        Generate patient-specific physiological parameters.
        
        Based on Bergman Minimal Model parameters with realistic
        inter-patient variability ranges from clinical literature.
        
        Returns:
            Dictionary of patient parameters
        """
        if self.patient_variability:
            params = {
                # Body weight (kg) - affects insulin sensitivity and distribution
                'weight': self.np_random.uniform(50.0, 100.0),
                
                # Glucose effectiveness (1/min) - insulin-independent glucose uptake
                'p1': self.np_random.uniform(0.025, 0.035),
                
                # Insulin action rate constants
                'p2': self.np_random.uniform(0.01, 0.02),   # Insulin action decay
                'p3': self.np_random.uniform(5e-6, 15e-6),  # Insulin action activation
                
                # Insulin clearance rate (1/min)
                'n': self.np_random.uniform(0.1, 0.2),
                
                # Volume of distribution for glucose (dL)
                'Vg': self.np_random.uniform(1.4, 1.8),  # dL/kg, multiplied by weight
                
                # Volume of distribution for insulin (L)
                'Vi': self.np_random.uniform(0.04, 0.06),  # L/kg, multiplied by weight
                
                # Basal glucose (mg/dL)
                'Gb': self.np_random.uniform(100.0, 130.0),
                
                # Basal insulin (mU/L)
                'Ib': self.np_random.uniform(10.0, 20.0),
                
                # Meal absorption parameters
                'k_abs': self.np_random.uniform(0.02, 0.04),  # Absorption rate (1/min)
                'f_cho': self.np_random.uniform(0.8, 1.0),    # Bioavailability fraction
                
                # Insulin-on-board decay time constant (hours)
                'tau_iob': self.np_random.uniform(3.0, 5.0),
                
                # Individual insulin sensitivity multiplier
                'insulin_sensitivity': self.np_random.uniform(0.7, 1.3),
            }
        else:
            # Default "average" patient for testing/debugging
            params = {
                'weight': 70.0,
                'p1': 0.03,
                'p2': 0.015,
                'p3': 1e-5,
                'n': 0.15,
                'Vg': 1.6,
                'Vi': 0.05,
                'Gb': 110.0,
                'Ib': 15.0,
                'k_abs': 0.03,
                'f_cho': 0.9,
                'tau_iob': 4.0,
                'insulin_sensitivity': 1.0,
            }
        
        # Compute derived parameters
        params['Vg_total'] = params['Vg'] * params['weight'] / 10.0  # Total Vg in dL
        params['Vi_total'] = params['Vi'] * params['weight']  # Total Vi in L
        
        return params
    
    def _generate_meal_schedule(self) -> List[Dict]:
        """
        Generate meal schedule for the episode.
        
        Standard meal protocol based on clinical studies:
        - Breakfast: ~7:00 AM, 50g CHO
        - Lunch: ~12:00 PM, 70g CHO
        - Dinner: ~6:00 PM, 80g CHO
        
        Returns:
            List of meal dictionaries with 'time' and 'cho' keys
        """
        if self.meal_variability:
            meals = [
                {
                    'time': 7.0 + self.np_random.uniform(-0.5, 0.5),   # Breakfast
                    'cho': 50.0 + self.np_random.uniform(-10.0, 10.0)
                },
                {
                    'time': 12.0 + self.np_random.uniform(-0.5, 0.5),  # Lunch
                    'cho': 70.0 + self.np_random.uniform(-15.0, 15.0)
                },
                {
                    'time': 18.0 + self.np_random.uniform(-0.5, 0.5),  # Dinner
                    'cho': 80.0 + self.np_random.uniform(-15.0, 15.0)
                }
            ]
        else:
            meals = [
                {'time': 7.0, 'cho': 50.0},
                {'time': 12.0, 'cho': 70.0},
                {'time': 18.0, 'cho': 80.0}
            ]
        
        return sorted(meals, key=lambda x: x['time'])
    
    def _get_observation(self) -> np.ndarray:
        """
        Construct normalised observation vector.
        
        Normalisation helps neural network training by keeping
        all inputs in similar ranges.
        
        Returns:
            Normalised state vector (8 dimensions)
        """
        # Get glucose history
        g_current = self.glucose
        g_prev = self.glucose_history[-1] if len(self.glucose_history) >= 1 else g_current
        g_prev2 = self.glucose_history[-2] if len(self.glucose_history) >= 2 else g_prev
        
        obs = np.array([
            g_current / 200.0,              # Normalise glucose by 200 mg/dL
            g_prev / 200.0,
            g_prev2 / 200.0,
            self.insulin / 50.0,            # Normalise insulin by 50 mU/L
            self.iob / 10.0,                # Normalise IOB by 10 U
            self.cho_absorbed / 100.0,      # Normalise CHO by 100g
            self.time_since_meal / 6.0,     # Normalise time by 6 hours
            self.patient_params['weight'] / 100.0  # Normalise weight by 100 kg
        ], dtype=np.float32)
        
        return obs
    
    def _simulate_dynamics(self, insulin_dose: float, cho_intake: float) -> None:
        """
        Simulate glucose-insulin dynamics for one time step.
        
        Uses simplified Bergman Minimal Model equations:
        - dG/dt = -p1*(G - Gb) - X*G + Ra/Vg
        - dX/dt = -p2*X + p3*I
        - dI/dt = -n*(I - Ib) + u/Vi
        
        Args:
            insulin_dose: Insulin infusion rate (U/hr)
            cho_intake: Carbohydrate intake (g) - only non-zero at meal times
        """
        p = self.patient_params
        dt = self.sample_time * 60.0  # Convert to minutes for model
        
        # Number of sub-steps for numerical stability
        n_substeps = 10
        dt_sub = dt / n_substeps
        
        for _ in range(n_substeps):
            # Glucose appearance rate from meal (mg/dL/min)
            Ra = self.cho_absorbed * p['k_abs'] * p['f_cho'] * 1000.0 / p['Vg_total']
            
            # Glucose dynamics (Bergman Minimal Model)
            dG = (-p['p1'] * (self.glucose - p['Gb']) 
                  - self.insulin_action * self.glucose 
                  + Ra) * dt_sub
            
            # Insulin action dynamics
            dX = (-p['p2'] * self.insulin_action 
                  + p['p3'] * p['insulin_sensitivity'] * self.insulin) * dt_sub
            
            # Insulin dynamics
            # Convert insulin dose from U/hr to mU/min: U/hr * 1000 / 60
            insulin_infusion = insulin_dose * 1000.0 / 60.0  # mU/min
            dI = (-p['n'] * (self.insulin - p['Ib']) 
                  + insulin_infusion / p['Vi_total']) * dt_sub
            
            # Update states
            self.glucose = max(0.0, self.glucose + dG)
            self.insulin_action = max(0.0, self.insulin_action + dX)
            self.insulin = max(0.0, self.insulin + dI)
            
            # Meal absorption (first-order decay)
            self.cho_absorbed = max(0.0, self.cho_absorbed * np.exp(-p['k_abs'] * dt_sub))
        
        # Add meal if at meal time
        if cho_intake > 0:
            self.cho_absorbed += cho_intake
            self.time_since_meal = 0.0
        
        # Update insulin-on-board (exponential decay)
        self.iob = self.iob * np.exp(-self.sample_time / p['tau_iob']) + insulin_dose * self.sample_time
        
        # Add small physiological noise
        noise = self.np_random.normal(0, 1.0)
        self.glucose = max(20.0, self.glucose + noise)
    
    def _compute_reward(self) -> float:
        """
    Compute reward based on current glucose level.
    
    SCALED DOWN by factor of 10 for training stability.
        """
    G = self.glucose
    
    # Zone-based rewards (SCALED DOWN)
    if G < 54:
        base_reward = -0.4  # Was -4.0
    elif G < 70:
        base_reward = -0.2  # Was -2.0
    elif G <= 180:
        base_reward = 0.1   # Was +1.0
    elif G <= 250:
        base_reward = -0.05 # Was -0.5
    elif G <= 400:
        base_reward = -0.1  # Was -1.0
    else:
        base_reward = -0.2  # Was -2.0
    
    # Bonus for being near centre of target (scaled)
    if 100 <= G <= 140:
        base_reward += 0.05  # Was +0.5
    
    # Rate of change penalty (scaled)
    if len(self.glucose_history) >= 2:
        dG = abs(self.glucose - self.glucose_history[-2])
        rate_penalty = -0.01 * max(0, dG - 2) / 10  # Was -0.1
        base_reward += rate_penalty
    
    # Insulin usage penalty (scaled)
    iob_penalty = -0.001 * self.insulin_on_board / 10  # Was -0.01
    base_reward += iob_penalty
    
    # Severe hypoglycaemia termination penalty (scaled)
    if G < 40:
        base_reward -= 1.0  # Was -10.0
    
    return base_reward
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        """
        Reset the environment for a new episode.
        
        Args:
            seed: Random seed
            options: Additional options (not used)
        
        Returns:
            Initial observation and info dictionary
        """
        super().reset(seed=seed)
        
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        
        # Generate patient parameters
        self.patient_params = self._generate_patient_parameters()
        
        # Generate meal schedule
        self.meals = self._generate_meal_schedule()
        self.meal_index = 0
        
        # Initialise state variables
        # Start with glucose in normal-ish range (random around 120)
        self.glucose = 120.0 + self.np_random.uniform(-30.0, 30.0)
        self.glucose_history = [self.glucose, self.glucose]
        
        # Start with basal insulin levels
        self.insulin = self.patient_params['Ib']
        self.insulin_action = 0.0
        self.iob = 0.0
        self.cho_absorbed = 0.0
        self.time_since_meal = 4.0  # Assume some time since last meal
        
        # Start at midnight (0:00) or random time
        self.current_time = 0.0
        self.current_step = 0
        
        # Clear episode traces
        self.glucose_trace = [self.glucose]
        self.insulin_trace = []
        self.reward_trace = []
        self.meal_trace = []
        
        # Get initial observation
        obs = self._get_observation()
        
        info = {
            'patient_weight': self.patient_params['weight'],
            'patient_insulin_sensitivity': self.patient_params['insulin_sensitivity'],
            'meal_schedule': self.meals.copy()
        }
        
        return obs, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one environment step.
        
        Args:
            action: Insulin dose [0, max_dose] U/hr
        
        Returns:
            observation: Next state
            reward: Reward for this step
            terminated: Whether episode ended (e.g., severe hypoglycaemia)
            truncated: Whether episode was cut short (time limit)
            info: Additional information
        """
        # Extract insulin dose from action
        insulin_dose = float(np.clip(action[0], 0.0, self.max_insulin_dose))
        
        # Check for meal at current time
        cho_intake = 0.0
        if self.meal_index < len(self.meals):
            meal = self.meals[self.meal_index]
            if self.current_time >= meal['time']:
                cho_intake = meal['cho']
                self.meal_trace.append({
                    'time': self.current_time,
                    'cho': cho_intake,
                    'step': self.current_step
                })
                self.meal_index += 1
        
        # Update glucose history before simulation
        self.glucose_history.append(self.glucose)
        if len(self.glucose_history) > 3:
            self.glucose_history.pop(0)
        
        # Simulate physiological dynamics
        self._simulate_dynamics(insulin_dose, cho_intake)
        
        # Update time
        self.current_step += 1
        self.current_time += self.sample_time
        self.time_since_meal += self.sample_time
        
        # Compute reward
        reward = self._compute_reward()
        
        # Track episode data
        self.glucose_trace.append(self.glucose)
        self.insulin_trace.append(insulin_dose)
        self.reward_trace.append(reward)
        
        # Check termination conditions
        terminated = False
        truncated = False
        
        # Severe hypoglycaemia terminates episode (safety)
        if self.glucose < 40:
            terminated = True
            reward -= 10.0  # Additional penalty for dangerous state
        
        # Time limit reached
        if self.current_step >= self.max_steps:
            truncated = True
        
        # Construct observation
        obs = self._get_observation()
        
        # Info dictionary
        info = {
            'glucose': self.glucose,
            'insulin_dose': insulin_dose,
            'iob': self.iob,
            'time': self.current_time,
            'step': self.current_step,
            'in_target_range': self.target_glucose_min <= self.glucose <= self.target_glucose_max,
            'cho_intake': cho_intake
        }
        
        return obs, reward, terminated, truncated, info
    
    def render(self):
        """Render the current state (text-based)."""
        if self.render_mode == 'human':
            status = "✓ IN RANGE" if 70 <= self.glucose <= 180 else "⚠ OUT OF RANGE"
            if self.glucose < 70:
                status = "⚠ HYPO"
            elif self.glucose > 180:
                status = "⚠ HYPER"
            
            print(f"Time: {self.current_time:.1f}h | "
                  f"Glucose: {self.glucose:.1f} mg/dL {status} | "
                  f"IOB: {self.iob:.2f} U | "
                  f"Last dose: {self.insulin_trace[-1] if self.insulin_trace else 0:.2f} U/hr")
    
    def get_episode_stats(self) -> Dict:
        """
        Calculate clinical metrics for the episode.
        
        Returns:
            Dictionary of clinical metrics
        """
        if len(self.glucose_trace) == 0:
            return {}
        
        glucose_array = np.array(self.glucose_trace)
        
        # Time in Range (TIR): 70-180 mg/dL
        tir = np.mean((glucose_array >= 70) & (glucose_array <= 180)) * 100
        
        # Time Below Range (TBR): < 70 mg/dL
        tbr = np.mean(glucose_array < 70) * 100
        
        # Time Above Range (TAR): > 180 mg/dL
        tar = np.mean(glucose_array > 180) * 100
        
        # Severe hypoglycaemia: < 54 mg/dL
        severe_hypo = np.mean(glucose_array < 54) * 100
        
        # Mean glucose
        mean_glucose = np.mean(glucose_array)
        
        # Glucose variability (Coefficient of Variation)
        cv = (np.std(glucose_array) / np.mean(glucose_array)) * 100 if np.mean(glucose_array) > 0 else 0
        
        # Total insulin delivered
        total_insulin = sum(self.insulin_trace) * self.sample_time if self.insulin_trace else 0
        
        # Episode reward
        total_reward = sum(self.reward_trace)
        
        return {
            'time_in_range': tir,
            'time_below_range': tbr,
            'time_above_range': tar,
            'severe_hypoglycaemia': severe_hypo,
            'mean_glucose': mean_glucose,
            'glucose_cv': cv,
            'total_insulin': total_insulin,
            'total_reward': total_reward,
            'episode_length': len(self.glucose_trace),
            'num_meals': len(self.meal_trace)
        }


def test_environment():
    """Test the environment with random actions."""
    print("Testing Glucose-Insulin Environment...")
    print("=" * 60)
    
    env = GlucoseInsulinEnv(patient_variability=True, meal_variability=True)
    obs, info = env.reset(seed=42)
    
    print(f"\nPatient Parameters:")
    print(f"  Weight: {info['patient_weight']:.1f} kg")
    print(f"  Insulin sensitivity: {info['patient_insulin_sensitivity']:.2f}")
    print(f"\nMeal Schedule:")
    for meal in info['meal_schedule']:
        print(f"  {meal['time']:.1f}h: {meal['cho']:.0f}g CHO")
    
    print(f"\nInitial observation shape: {obs.shape}")
    print(f"Initial glucose: {env.glucose:.1f} mg/dL")
    
    print("\n" + "=" * 60)
    print("Running episode with fixed basal insulin (1.0 U/hr)...")
    print("=" * 60 + "\n")
    
    total_reward = 0
    done = False
    step = 0
    
    while not done:
        # Fixed basal rate
        action = np.array([1.0])
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated
        
        # Print every hour
        if step % 12 == 0:
            env.render()
        
        step += 1
    
    print("\n" + "=" * 60)
    print("Episode Complete!")
    print("=" * 60)
    
    stats = env.get_episode_stats()
    print(f"\nClinical Metrics:")
    print(f"  Time in Range (70-180): {stats['time_in_range']:.1f}%")
    print(f"  Time Below Range (<70): {stats['time_below_range']:.1f}%")
    print(f"  Time Above Range (>180): {stats['time_above_range']:.1f}%")
    print(f"  Mean Glucose: {stats['mean_glucose']:.1f} mg/dL")
    print(f"  Glucose CV: {stats['glucose_cv']:.1f}%")
    print(f"  Total Insulin: {stats['total_insulin']:.1f} U")
    print(f"  Total Reward: {stats['total_reward']:.1f}")
    
    print("\n✓ Environment test passed!")


if __name__ == "__main__":
    test_environment()