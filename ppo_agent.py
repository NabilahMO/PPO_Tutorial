"""
Proximal Policy Optimisation (PPO) Agent for Continuous Actions
================================================================

Implementation of PPO with clipped surrogate objective for
continuous action spaces (Gaussian policy).

Key components:
- Clipped surrogate objective (Equation 7 from PPO paper)
- Generalised Advantage Estimation (GAE)
- Value function clipping (optional)
- Entropy bonus for exploration

Based on:
- Schulman et al. (2017) "Proximal Policy Optimization Algorithms"
- Implementation best practices from Stable-Baselines3 and CleanRL
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Optional

from networks import ContinuousActorCritic


class RolloutBuffer:
    """
    Buffer for storing trajectory data during rollout.
    
    Stores transitions (s, a, r, s', done) along with
    log probabilities and value estimates needed for PPO.
    """
    
    def __init__(self):
        """Initialise empty buffer."""
        self.states: List[np.ndarray] = []
        self.actions: List[np.ndarray] = []
        self.rewards: List[float] = []
        self.values: List[float] = []
        self.log_probs: List[float] = []
        self.dones: List[bool] = []
    
    def store(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        value: float,
        log_prob: float,
        done: bool
    ) -> None:
        """
        Store a single transition.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            value: Value estimate V(s)
            log_prob: Log probability of action under policy
            done: Whether episode ended
        """
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)
    
    def clear(self) -> None:
        """Clear all stored data."""
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.values.clear()
        self.log_probs.clear()
        self.dones.clear()
    
    def __len__(self) -> int:
        """Return number of stored transitions."""
        return len(self.states)


class PPOAgent:
    """
    Proximal Policy Optimisation agent for continuous control.
    
    PPO maintains a policy (actor) and value function (critic),
    updating them using the clipped surrogate objective to ensure
    stable learning without destructively large policy changes.
    
    Key hyperparameters:
    - epsilon: Clipping parameter (default 0.2)
    - gamma: Discount factor (default 0.99)
    - gae_lambda: GAE parameter (default 0.95)
    - update_epochs: Number of optimisation epochs per update (default 10)
    
    Attributes:
        network: Actor-Critic neural network
        optimiser: Adam optimiser
        buffer: Rollout buffer for storing trajectories
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        action_low: float = 0.0,
        action_high: float = 5.0,
        hidden_dim: int = 64,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        epsilon: float = 0.2,
        epsilon_decay: float = 1.0,
        epsilon_min: float = 0.1,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        update_epochs: int = 10,
        batch_size: int = 64,
        clip_value_loss: bool = True,
        normalise_advantages: bool = True,
        device: str = 'cpu'
    ):
        """
        Initialise PPO agent.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            action_low: Minimum action value
            action_high: Maximum action value
            hidden_dim: Size of network hidden layers
            lr: Learning rate for Adam optimiser
            gamma: Discount factor for future rewards
            gae_lambda: Lambda for Generalised Advantage Estimation
            epsilon: PPO clipping parameter
            epsilon_decay: Decay factor for epsilon (per update)
            epsilon_min: Minimum epsilon value
            value_coef: Coefficient for value function loss
            entropy_coef: Coefficient for entropy bonus
            max_grad_norm: Maximum gradient norm for clipping
            update_epochs: Number of optimisation epochs per update
            batch_size: Mini-batch size for updates
            clip_value_loss: Whether to clip value function loss
            normalise_advantages: Whether to normalise advantages
            device: Device to run on ('cpu' or 'cuda')
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_low = action_low
        self.action_high = action_high
        
        # Hyperparameters
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
        
        # Create network
        self.network = ContinuousActorCritic(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            shared_layers=False  # Separate networks often work better
        ).to(device)
        
        # Optimiser
        self.optimiser = optim.Adam(self.network.parameters(), lr=lr)
        
        # Rollout buffer
        self.buffer = RolloutBuffer()
        
        # Metrics tracking
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
    
    def get_action(
        self,
        state: np.ndarray,
        deterministic: bool = False
    ) -> Tuple[np.ndarray, float, float]:
        """
        Select action using current policy.
        
        Args:
            state: Current state
            deterministic: If True, return mean action (for evaluation)
        
        Returns:
            action: Selected action (clipped to valid range)
            log_prob: Log probability of action
            value: State value estimate
        """
        action, log_prob, value = self.network.get_action(state, deterministic)
        
        # Clip action to valid range
        action = np.clip(action, self.action_low, self.action_high)
        
        return action, log_prob, value
    
    def store_transition(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        value: float,
        log_prob: float,
        done: bool
    ) -> None:
        """Store a transition in the buffer."""
        self.buffer.store(state, action, reward, value, log_prob, done)
    
    def compute_gae(
        self,
        next_value: float,
        next_done: bool
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Generalised Advantage Estimation (GAE).
        
        GAE balances bias and variance in advantage estimation:
        - lambda=0: High bias, low variance (1-step TD)
        - lambda=1: Low bias, high variance (Monte Carlo)
        - lambda=0.95: Good balance (typical default)
        
        Formula:
        A_t = delta_t + (gamma * lambda) * delta_{t+1} + ...
        where delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
        
        Args:
            next_value: Value estimate of final state
            next_done: Whether final state is terminal
        
        Returns:
            advantages: GAE advantages for each timestep
            returns: Computed returns (advantages + values)
        """
        rewards = np.array(self.buffer.rewards)
        values = np.array(self.buffer.values)
        dones = np.array(self.buffer.dones)
        
        num_steps = len(rewards)
        advantages = np.zeros(num_steps)
        
        # Compute GAE backwards
        last_gae = 0.0
        
        for t in reversed(range(num_steps)):
            if t == num_steps - 1:
                next_non_terminal = 1.0 - float(next_done)
                next_val = next_value
            else:
                next_non_terminal = 1.0 - float(dones[t])
                next_val = values[t + 1]
            
            # TD error: delta = r + gamma * V(s') - V(s)
            delta = rewards[t] + self.gamma * next_val * next_non_terminal - values[t]
            
            # GAE: A = delta + gamma * lambda * A'
            advantages[t] = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae
            last_gae = advantages[t]
        
        # Returns = advantages + values
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
        Compute the PPO clipped surrogate loss.
        
        This is the core of PPO - the clipped objective ensures
        policy updates stay within a "trust region" without
        requiring complex constrained optimisation.
        
        Loss = L_policy + c1 * L_value - c2 * entropy_bonus
        
        Args:
            states: Batch of states
            actions: Batch of actions taken
            old_log_probs: Log probs under old policy
            advantages: Computed advantages
            returns: Computed returns
            old_values: Value estimates under old policy
        
        Returns:
            total_loss: Combined loss for optimisation
            metrics: Dictionary of loss components and statistics
        """
        # Get current policy evaluation
        new_log_probs, new_values, entropy = self.network.evaluate_actions(states, actions)
        
        # ============================================================
        # STEP 1: Compute probability ratio
        # r(theta) = pi_new(a|s) / pi_old(a|s)
        # In log space: r = exp(log_pi_new - log_pi_old)
        # ============================================================
        ratio = torch.exp(new_log_probs - old_log_probs)
        
        # ============================================================
        # STEP 2: Compute clipped ratio
        # clip(r, 1-epsilon, 1+epsilon)
        # ============================================================
        ratio_clipped = torch.clamp(ratio, 1.0 - self.epsilon, 1.0 + self.epsilon)
        
        # ============================================================
        # STEP 3: Compute surrogate objectives
        # L1 = r * A (unclipped)
        # L2 = clip(r) * A (clipped)
        # ============================================================
        surrogate1 = ratio * advantages
        surrogate2 = ratio_clipped * advantages
        
        # ============================================================
        # STEP 4: Take minimum (pessimistic bound)
        # L_policy = -min(L1, L2)
        # Negative because we want to maximise, but optimisers minimise
        # ============================================================
        policy_loss = -torch.min(surrogate1, surrogate2).mean()
        
        # ============================================================
        # STEP 5: Value function loss
        # L_value = (V(s) - R)^2
        # ============================================================
        if self.clip_value_loss:
            # Clipped value loss (optional, can help stability)
            value_clipped = old_values + torch.clamp(
                new_values - old_values, -self.epsilon, self.epsilon
            )
            value_loss1 = (new_values - returns) ** 2
            value_loss2 = (value_clipped - returns) ** 2
            value_loss = 0.5 * torch.max(value_loss1, value_loss2).mean()
        else:
            value_loss = 0.5 * ((new_values - returns) ** 2).mean()
        
        # ============================================================
        # STEP 6: Entropy bonus (encourages exploration)
        # L_entropy = -H(pi)
        # ============================================================
        entropy_loss = -entropy.mean()
        
        # ============================================================
        # STEP 7: Combine losses
        # L_total = L_policy + c1 * L_value + c2 * L_entropy
        # ============================================================
        total_loss = (
            policy_loss 
            + self.value_coef * value_loss 
            + self.entropy_coef * entropy_loss
        )
        
        # Compute additional metrics
        with torch.no_grad():
            # Fraction of ratios that were clipped
            clipped = (
                (ratio < 1.0 - self.epsilon) | (ratio > 1.0 + self.epsilon)
            ).float()
            clipped_fraction = clipped.mean().item()
            
            # Approximate KL divergence
            # KL ≈ (r - 1) - log(r)
            approx_kl = ((ratio - 1.0) - torch.log(ratio)).mean().item()
            
            # Explained variance
            # How well does V(s) predict R?
            var_returns = torch.var(returns)
            if var_returns > 0:
                explained_var = 1.0 - torch.var(returns - new_values) / var_returns
                explained_var = explained_var.item()
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
            'kl_divergence': approx_kl,
            'explained_variance': explained_var
        }
        
        return total_loss, metrics
    
    def update(self, next_value: float = 0.0, next_done: bool = True) -> Dict[str, float]:
        """
        Perform PPO update using collected experience.
        
        This method:
        1. Computes advantages using GAE
        2. Normalises advantages (optional)
        3. Runs multiple epochs of mini-batch updates
        4. Decays epsilon (optional)
        
        Args:
            next_value: Value estimate of final state
            next_done: Whether final state is terminal
        
        Returns:
            Dictionary of average metrics from update
        """
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
        
        # Normalise advantages
        if self.normalise_advantages:
            advantages_tensor = (
                (advantages_tensor - advantages_tensor.mean()) 
                / (advantages_tensor.std() + 1e-8)
            )
        
        # Track metrics for this update
        update_metrics = {
            'policy_loss': [],
            'value_loss': [],
            'entropy': [],
            'total_loss': [],
            'ratio_mean': [],
            'ratio_std': [],
            'clipped_fraction': [],
            'kl_divergence': [],
            'explained_variance': []
        }
        
        # Store advantage statistics
        self.metrics['advantages_mean'].append(advantages.mean())
        self.metrics['advantages_std'].append(advantages.std())
        
        num_samples = states.shape[0]
        
        # Multiple epochs of updates
        for epoch in range(self.update_epochs):
            # Random permutation for mini-batches
            indices = np.random.permutation(num_samples)
            
            for start in range(0, num_samples, self.batch_size):
                end = start + self.batch_size
                batch_indices = indices[start:end]
                
                # Get mini-batch
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages_tensor[batch_indices]
                batch_returns = returns_tensor[batch_indices]
                batch_old_values = old_values[batch_indices]
                
                # Compute loss
                loss, metrics = self.compute_ppo_loss(
                    batch_states,
                    batch_actions,
                    batch_old_log_probs,
                    batch_advantages,
                    batch_returns,
                    batch_old_values
                )
                
                # Gradient descent
                self.optimiser.zero_grad()
                loss.backward()
                
                # Gradient clipping
                nn.utils.clip_grad_norm_(
                    self.network.parameters(), 
                    self.max_grad_norm
                )
                
                self.optimiser.step()
                
                # Track metrics
                for key in update_metrics:
                    update_metrics[key].append(metrics[key])
        
        # Average metrics over all batches/epochs
        avg_metrics = {key: np.mean(vals) for key, vals in update_metrics.items()}
        
        # Store in history
        for key, val in avg_metrics.items():
            self.metrics[key].append(val)
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        # Clear buffer
        self.buffer.clear()
        
        return avg_metrics
    
    def get_metrics(self) -> Dict[str, List[float]]:
        """Return all tracked metrics."""
        return self.metrics
    
    def save(self, filepath: str) -> None:
        """
        Save agent state to file.
        
        Args:
            filepath: Path to save file
        """
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'optimiser_state_dict': self.optimiser.state_dict(),
            'epsilon': self.epsilon,
            'metrics': self.metrics
        }, filepath)
    
    def load(self, filepath: str) -> None:
        """
        Load agent state from file.
        
        Args:
            filepath: Path to load file
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.optimiser.load_state_dict(checkpoint['optimiser_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.metrics = checkpoint['metrics']


def test_ppo_agent():
    """Test the PPO agent implementation."""
    print("Testing PPO Agent...")
    print("=" * 60)
    
    # Create agent
    state_dim = 8
    action_dim = 1
    
    agent = PPOAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        action_low=0.0,
        action_high=5.0,
        hidden_dim=64,
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        epsilon=0.2,
        update_epochs=4,
        batch_size=32
    )
    
    print(f"\nAgent created:")
    print(f"  State dim: {state_dim}")
    print(f"  Action dim: {action_dim}")
    print(f"  Action range: [{agent.action_low}, {agent.action_high}]")
    print(f"  Epsilon: {agent.epsilon}")
    
    # Test action selection
    print("\nTesting action selection...")
    test_state = np.random.randn(state_dim).astype(np.float32)
    
    action, log_prob, value = agent.get_action(test_state, deterministic=False)
    print(f"  Stochastic action: {action}")
    print(f"  Log prob: {log_prob:.4f}")
    print(f"  Value: {value:.4f}")
    
    action_det, _, _ = agent.get_action(test_state, deterministic=True)
    print(f"  Deterministic action: {action_det}")
    
    # Test rollout and update
    print("\nTesting rollout buffer and update...")
    
    # Simulate a rollout
    num_steps = 128
    for _ in range(num_steps):
        state = np.random.randn(state_dim).astype(np.float32)
        action, log_prob, value = agent.get_action(state)
        reward = np.random.randn()
        done = np.random.random() < 0.01
        
        agent.store_transition(state, action, reward, value, log_prob, done)
    
    print(f"  Buffer size: {len(agent.buffer)}")
    
    # Perform update
    update_metrics = agent.update(next_value=0.0, next_done=True)
    
    print(f"\n  Update metrics:")
    print(f"    Policy loss: {update_metrics['policy_loss']:.4f}")
    print(f"    Value loss: {update_metrics['value_loss']:.4f}")
    print(f"    Entropy: {update_metrics['entropy']:.4f}")
    print(f"    Clipped fraction: {update_metrics['clipped_fraction']:.3f}")
    print(f"    KL divergence: {update_metrics['kl_divergence']:.4f}")
    
    print(f"\n  Buffer size after update: {len(agent.buffer)}")
    
    # Test save/load
    print("\nTesting save/load...")
    import tempfile
    import os
    
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "test_agent.pt")
        agent.save(filepath)
        print(f"  Saved to {filepath}")
        
        # Create new agent and load
        new_agent = PPOAgent(state_dim, action_dim)
        new_agent.load(filepath)
        print(f"  Loaded successfully")
        
        # Check epsilon was restored
        print(f"  Restored epsilon: {new_agent.epsilon}")
    
    print("\n" + "=" * 60)
    print("✓ All PPO agent tests passed!")


if __name__ == "__main__":
    test_ppo_agent()