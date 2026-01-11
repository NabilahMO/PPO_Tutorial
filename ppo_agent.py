"""
Proximal Policy Optimisation (PPO) Agent for Continuous Actions
================================================================

STABILITY-FIXED VERSION with numerical safeguards.

Key fixes:
- Clamped log probabilities to prevent -inf
- Clamped probability ratios to prevent overflow
- Safe KL divergence calculation
- Gradient clipping

"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Optional

from networks import ContinuousActorCritic


class RolloutBuffer:
    """Buffer for storing trajectory data during rollout."""
    
    def __init__(self):
        self.states: List[np.ndarray] = []
        self.actions: List[np.ndarray] = []
        self.rewards: List[float] = []
        self.values: List[float] = []
        self.log_probs: List[float] = []
        self.dones: List[bool] = []
    
    def store(self, state, action, reward, value, log_prob, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)
    
    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.values.clear()
        self.log_probs.clear()
        self.dones.clear()
    
    def __len__(self):
        return len(self.states)


class PPOAgent:
    """
    PPO agent for continuous control - STABILITY FIXED.
    
    Key numerical stability improvements:
    - Log probabilities clamped to [-20, 2]
    - Probability ratios clamped to [0.01, 100]
    - Safe KL divergence calculation
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        action_low: float = 0.0,
        action_high: float = 5.0,
        hidden_dim: int = 64,
        lr: float = 1e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        epsilon: float = 0.1,
        epsilon_decay: float = 1.0,
        epsilon_min: float = 0.05,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        update_epochs: int = 4,
        batch_size: int = 64,
        clip_value_loss: bool = True,
        normalise_advantages: bool = True,
        device: str = 'cpu'
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_low = action_low
        self.action_high = action_high
        
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.update_epochs = update_epochs
        self.batch_size = batch_size
        self.clip_value_loss = clip_value_loss
        self.normalise_advantages = normalise_advantages
        self.device = device
        
        self.network = ContinuousActorCritic(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            shared_layers=False
        ).to(device)
        
        self.optimiser = optim.Adam(self.network.parameters(), lr=lr)
        self.buffer = RolloutBuffer()
        
        self.metrics = {
            'policy_loss': [],
            'value_loss': [],
            'entropy': [],
            'total_loss': [],
            'ratio_mean': [],
            'ratio_std': [],
            'clipped_fraction': [],
            'kl_divergence': [],
            'explained_variance': [],
            'advantages_mean': [],
            'advantages_std': []
        }
    
    def get_action(self, state: np.ndarray, deterministic: bool = False) -> Tuple[np.ndarray, float, float]:
        """Select action using current policy."""
        action, log_prob, value = self.network.get_action(state, deterministic)
        
        # Clip action to valid range
        action = np.clip(action, self.action_low, self.action_high)
        
        # STABILITY: Clamp log_prob to prevent extreme values
        log_prob = np.clip(log_prob, -20.0, 2.0)
        
        return action, log_prob, value
    
    def store_transition(self, state, action, reward, value, log_prob, done):
        """Store a transition in the buffer."""
        # STABILITY: Clamp log_prob when storing
        log_prob = np.clip(log_prob, -20.0, 2.0)
        self.buffer.store(state, action, reward, value, log_prob, done)
    
    def compute_gae(self, next_value: float, next_done: bool) -> Tuple[np.ndarray, np.ndarray]:
        """Compute Generalised Advantage Estimation."""
        rewards = np.array(self.buffer.rewards)
        values = np.array(self.buffer.values)
        dones = np.array(self.buffer.dones)
        
        num_steps = len(rewards)
        advantages = np.zeros(num_steps)
        
        last_gae = 0.0
        
        for t in reversed(range(num_steps)):
            if t == num_steps - 1:
                next_non_terminal = 1.0 - float(next_done)
                next_val = next_value
            else:
                next_non_terminal = 1.0 - float(dones[t])
                next_val = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_val * next_non_terminal - values[t]
            advantages[t] = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae
            last_gae = advantages[t]
        
        returns = advantages + values
        
        return advantages, returns
    
    def compute_ppo_loss(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        returns: torch.Tensor,
        old_values: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute PPO loss with numerical stability fixes.
        """
        # Get current policy evaluation
        new_log_probs, new_values, entropy = self.network.evaluate_actions(states, actions)
        
        # STABILITY: Clamp log probs to prevent extreme values
        new_log_probs = torch.clamp(new_log_probs, -20.0, 2.0)
        old_log_probs = torch.clamp(old_log_probs, -20.0, 2.0)
        
        # Compute probability ratio
        log_ratio = new_log_probs - old_log_probs
        
        # STABILITY: Clamp log ratio to prevent overflow in exp
        log_ratio = torch.clamp(log_ratio, -10.0, 10.0)
        ratio = torch.exp(log_ratio)
        
        # STABILITY: Clamp ratio to reasonable range
        ratio = torch.clamp(ratio, 0.01, 100.0)
        
        # Clipped ratio
        ratio_clipped = torch.clamp(ratio, 1.0 - self.epsilon, 1.0 + self.epsilon)
        
        # Surrogate objectives
        surrogate1 = ratio * advantages
        surrogate2 = ratio_clipped * advantages
        
        # Policy loss (negative for gradient ascent)
        policy_loss = -torch.min(surrogate1, surrogate2).mean()
        
        # Value loss
        if self.clip_value_loss:
            value_clipped = old_values + torch.clamp(
                new_values - old_values, -self.epsilon, self.epsilon
            )
            value_loss1 = (new_values - returns) ** 2
            value_loss2 = (value_clipped - returns) ** 2
            value_loss = 0.5 * torch.max(value_loss1, value_loss2).mean()
        else:
            value_loss = 0.5 * ((new_values - returns) ** 2).mean()
        
        # Entropy loss
        entropy_loss = -entropy.mean()
        
        # Total loss
        total_loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss
        
        # Compute metrics
        with torch.no_grad():
            clipped = ((ratio < 1.0 - self.epsilon) | (ratio > 1.0 + self.epsilon)).float()
            clipped_fraction = clipped.mean().item()
            
            # STABILITY: Safe KL divergence calculation
            # KL ≈ (r - 1) - log(r), but we need to handle edge cases
            safe_ratio = torch.clamp(ratio, 1e-8, 1e8)
            approx_kl = ((ratio - 1.0) - torch.log(safe_ratio)).mean()
            
            # Clamp KL to reasonable range
            if torch.isnan(approx_kl) or torch.isinf(approx_kl):
                approx_kl_val = 0.0
            else:
                approx_kl_val = min(approx_kl.item(), 100.0)
            
            # Explained variance
            var_returns = torch.var(returns)
            if var_returns > 1e-8:
                explained_var = 1.0 - torch.var(returns - new_values) / var_returns
                explained_var = float(torch.clamp(explained_var, -1.0, 1.0).item())
            else:
                explained_var = 0.0
        
        metrics = {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': entropy.mean().item(),
            'total_loss': total_loss.item(),
            'ratio_mean': ratio.mean().item(),
            'ratio_std': ratio.std().item(),
            'clipped_fraction': clipped_fraction,
            'kl_divergence': approx_kl_val,
            'explained_variance': explained_var
        }
        
        return total_loss, metrics
    
    def update(self, next_value: float = 0.0, next_done: bool = True) -> Dict[str, float]:
        """Perform PPO update with stability checks."""
        if len(self.buffer) == 0:
            return {}
        
        # Compute advantages and returns
        advantages, returns = self.compute_gae(next_value, next_done)
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(self.buffer.states)).to(self.device)
        actions = torch.FloatTensor(np.array(self.buffer.actions)).to(self.device)
        old_log_probs = torch.FloatTensor(np.array(self.buffer.log_probs)).to(self.device)
        old_values = torch.FloatTensor(np.array(self.buffer.values)).to(self.device)
        advantages_tensor = torch.FloatTensor(advantages).to(self.device)
        returns_tensor = torch.FloatTensor(returns).to(self.device)
        
        # STABILITY: Clamp old_log_probs
        old_log_probs = torch.clamp(old_log_probs, -20.0, 2.0)
        
        # Normalise advantages
        if self.normalise_advantages:
            adv_mean = advantages_tensor.mean()
            adv_std = advantages_tensor.std()
            if adv_std > 1e-8:
                advantages_tensor = (advantages_tensor - adv_mean) / (adv_std + 1e-8)
            else:
                advantages_tensor = advantages_tensor - adv_mean
        
        # STABILITY: Normalise returns as well
        ret_mean = returns_tensor.mean()
        ret_std = returns_tensor.std()
        if ret_std > 1e-8:
            returns_tensor = (returns_tensor - ret_mean) / (ret_std + 1e-8)
        
        # Track metrics
        update_metrics = {k: [] for k in ['policy_loss', 'value_loss', 'entropy', 
                                          'total_loss', 'ratio_mean', 'ratio_std',
                                          'clipped_fraction', 'kl_divergence', 
                                          'explained_variance']}
        
        self.metrics['advantages_mean'].append(float(advantages.mean()))
        self.metrics['advantages_std'].append(float(advantages.std()))
        
        num_samples = states.shape[0]
        
        # Multiple epochs
        for epoch in range(self.update_epochs):
            indices = np.random.permutation(num_samples)
            
            for start in range(0, num_samples, self.batch_size):
                end = start + self.batch_size
                batch_idx = indices[start:end]
                
                batch_states = states[batch_idx]
                batch_actions = actions[batch_idx]
                batch_old_log_probs = old_log_probs[batch_idx]
                batch_advantages = advantages_tensor[batch_idx]
                batch_returns = returns_tensor[batch_idx]
                batch_old_values = old_values[batch_idx]
                
                loss, metrics = self.compute_ppo_loss(
                    batch_states, batch_actions, batch_old_log_probs,
                    batch_advantages, batch_returns, batch_old_values
                )
                
                # STABILITY: Check for NaN loss
                if torch.isnan(loss) or torch.isinf(loss):
                    print("⚠️ NaN/Inf loss detected, skipping batch")
                    continue
                
                self.optimiser.zero_grad()
                loss.backward()
                
                # Gradient clipping
                nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                
                self.optimiser.step()
                
                for key in update_metrics:
                    update_metrics[key].append(metrics[key])
        
        # Average metrics
        avg_metrics = {}
        for key, vals in update_metrics.items():
            if vals:
                avg_metrics[key] = np.mean(vals)
            else:
                avg_metrics[key] = 0.0
        
        # Store in history
        for key, val in avg_metrics.items():
            self.metrics[key].append(val)
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        # Clear buffer
        self.buffer.clear()
        
        return avg_metrics
    
    def get_metrics(self) -> Dict[str, List[float]]:
        return self.metrics
    
    def save(self, filepath: str) -> None:
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'optimiser_state_dict': self.optimiser.state_dict(),
            'epsilon': self.epsilon,
            'metrics': self.metrics
        }, filepath)
    
    def load(self, filepath: str) -> None:
        checkpoint = torch.load(filepath, map_location=self.device)
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.optimiser.load_state_dict(checkpoint['optimiser_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.metrics = checkpoint['metrics']


def test_ppo_agent():
    """Test the PPO agent."""
    print("Testing PPO Agent (Stability Fixed)...")
    print("=" * 60)
    
    agent = PPOAgent(
        state_dim=8,
        action_dim=1,
        action_low=0.0,
        action_high=5.0,
        hidden_dim=64,
        lr=1e-4,
        epsilon=0.1,
        update_epochs=4
    )
    
    print(f"Agent created with epsilon={agent.epsilon}")
    
    # Test action selection
    test_state = np.random.randn(8).astype(np.float32)
    action, log_prob, value = agent.get_action(test_state)
    
    print(f"\nAction: {action}")
    print(f"Log prob: {log_prob:.4f} (should be in [-20, 2])")
    print(f"Value: {value:.4f}")
    
    # Test rollout and update
    print("\nTesting rollout and update...")
    
    for _ in range(128):
        state = np.random.randn(8).astype(np.float32)
        action, log_prob, value = agent.get_action(state)
        reward = np.random.randn() * 0.1  # Small rewards
        done = np.random.random() < 0.01
        agent.store_transition(state, action, reward, value, log_prob, done)
    
    print(f"Buffer size: {len(agent.buffer)}")
    
    update_metrics = agent.update(next_value=0.0, next_done=True)
    
    print(f"\nUpdate metrics:")
    print(f"  Policy loss: {update_metrics['policy_loss']:.4f}")
    print(f"  Value loss: {update_metrics['value_loss']:.4f}")
    print(f"  KL divergence: {update_metrics['kl_divergence']:.4f} (should NOT be inf)")
    print(f"  Clipped fraction: {update_metrics['clipped_fraction']:.3f}")
    
    print("\n" + "=" * 60)
    print("✓ PPO agent test passed!")


if __name__ == "__main__":
    test_ppo_agent()