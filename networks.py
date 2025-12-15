"""
Neural Networks for Continuous Action PPO
==========================================

Actor-Critic architecture for continuous action spaces.

The Actor outputs parameters of a Gaussian distribution (mean and std),
from which actions are sampled. This allows for smooth, continuous
control signals ideal for insulin dosing.

The Critic estimates state values for advantage computation.

Based on:
- Schulman et al. (2017) PPO paper architecture recommendations
- Standard practices for continuous control (SAC, TD3)
"""

import torch
import torch.nn as nn
import numpy as np
from torch.distributions import Normal
from typing import Tuple, Optional


def initialise_weights(module: nn.Module, gain: float = np.sqrt(2)) -> None:
    """
    Orthogonal weight initialisation (PPO best practice).
    
    Orthogonal initialisation helps with gradient flow and is
    recommended in the PPO paper for policy networks.
    
    Args:
        module: Neural network module to initialise
        gain: Scaling factor for weights
    """
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight, gain=gain)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0.0)


class ContinuousActorCritic(nn.Module):
    """
    Combined Actor-Critic network for continuous action spaces.
    
    Architecture:
    - Shared feature extractor (optional, can be separate)
    - Actor head: outputs mean and log_std of Gaussian policy
    - Critic head: outputs state value V(s)
    
    The Gaussian policy allows smooth continuous actions,
    essential for precise insulin dosing control.
    
    Attributes:
        actor_mean: Network outputting action means
        actor_log_std: Learnable log standard deviation
        critic: Network outputting state values
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 64,
        log_std_min: float = -20.0,
        log_std_max: float = 2.0,
        shared_layers: bool = False
    ):
        """
        Initialise the Actor-Critic network.
        
        Args:
            state_dim: Dimension of state space (8 for glucose env)
            action_dim: Dimension of action space (1 for insulin dose)
            hidden_dim: Size of hidden layers
            log_std_min: Minimum log standard deviation (numerical stability)
            log_std_max: Maximum log standard deviation (exploration control)
            shared_layers: If True, actor and critic share feature layers
        """
        super(ContinuousActorCritic, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.shared_layers = shared_layers
        
        if shared_layers:
            # Shared feature extractor
            self.shared = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh()
            )
            
            # Actor head (outputs mean)
            self.actor_mean = nn.Linear(hidden_dim, action_dim)
            
            # Critic head (outputs value)
            self.critic_head = nn.Linear(hidden_dim, 1)
        else:
            # Separate actor network
            self.actor_net = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh()
            )
            self.actor_mean = nn.Linear(hidden_dim, action_dim)
            
            # Separate critic network
            self.critic_net = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh()
            )
            self.critic_head = nn.Linear(hidden_dim, 1)
        
        # Learnable log standard deviation
        # Initialised to 0 → std = 1 (good starting exploration)
        self.actor_log_std = nn.Parameter(torch.zeros(action_dim))
        
        # Apply weight initialisation
        self.apply(lambda m: initialise_weights(m, gain=np.sqrt(2)))
        
        # Smaller initialisation for output layers (helps stability)
        initialise_weights(self.actor_mean, gain=0.01)
        initialise_weights(self.critic_head, gain=1.0)
    
    def forward(
        self,
        state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the network.
        
        Args:
            state: State tensor of shape (batch_size, state_dim)
        
        Returns:
            action_mean: Mean of action distribution (batch_size, action_dim)
            action_std: Std of action distribution (batch_size, action_dim)
            value: State value estimate (batch_size, 1)
        """
        if self.shared_layers:
            features = self.shared(state)
            action_mean = self.actor_mean(features)
            value = self.critic_head(features)
        else:
            actor_features = self.actor_net(state)
            action_mean = self.actor_mean(actor_features)
            
            critic_features = self.critic_net(state)
            value = self.critic_head(critic_features)
        
        # Clamp log_std for numerical stability
        log_std = torch.clamp(self.actor_log_std, self.log_std_min, self.log_std_max)
        action_std = log_std.exp().expand_as(action_mean)
        
        return action_mean, action_std, value
    
    def get_action(
        self,
        state: np.ndarray,
        deterministic: bool = False
    ) -> Tuple[np.ndarray, float, float]:
        """
        Sample an action from the policy.
        
        For training: sample from Gaussian (stochastic)
        For evaluation: return mean (deterministic)
        
        Args:
            state: Current state (numpy array)
            deterministic: If True, return mean action (no sampling)
        
        Returns:
            action: Selected action (numpy array)
            log_prob: Log probability of the action
            value: State value estimate
        """
        # Convert to tensor
        if isinstance(state, np.ndarray):
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
        else:
            state_tensor = state.unsqueeze(0) if state.dim() == 1 else state
        
        with torch.no_grad():
            action_mean, action_std, value = self.forward(state_tensor)
            
            if deterministic:
                action = action_mean
                # Log prob of mean action under Gaussian
                dist = Normal(action_mean, action_std)
                log_prob = dist.log_prob(action).sum(dim=-1)
            else:
                # Sample from Gaussian distribution
                dist = Normal(action_mean, action_std)
                action = dist.sample()
                log_prob = dist.log_prob(action).sum(dim=-1)
        
        return (
            action.squeeze(0).numpy(),
            log_prob.item(),
            value.squeeze().item()
        )
    
    def evaluate_actions(
        self,
        states: torch.Tensor,
        actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions for PPO update.
        
        Given states and actions taken, compute:
        - Log probabilities under current policy
        - State values
        - Policy entropy (for exploration bonus)
        
        Args:
            states: Batch of states (batch_size, state_dim)
            actions: Batch of actions taken (batch_size, action_dim)
        
        Returns:
            log_probs: Log probabilities of actions (batch_size,)
            values: State value estimates (batch_size,)
            entropy: Policy entropy (batch_size,)
        """
        action_mean, action_std, values = self.forward(states)
        
        # Create Gaussian distribution
        dist = Normal(action_mean, action_std)
        
        # Log probability of the actions
        log_probs = dist.log_prob(actions).sum(dim=-1)
        
        # Entropy of the policy (for exploration bonus)
        entropy = dist.entropy().sum(dim=-1)
        
        return log_probs, values.squeeze(-1), entropy
    
    def get_value(self, state: np.ndarray) -> float:
        """
        Get state value estimate only.
        
        Useful for computing advantages at end of episode.
        
        Args:
            state: Current state
        
        Returns:
            State value estimate
        """
        if isinstance(state, np.ndarray):
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
        else:
            state_tensor = state.unsqueeze(0) if state.dim() == 1 else state
        
        with torch.no_grad():
            _, _, value = self.forward(state_tensor)
        
        return value.squeeze().item()


class SeparateActorCritic(nn.Module):
    """
    Alternative: Completely separate Actor and Critic networks.
    
    Use this when:
    - You want different learning rates for actor/critic
    - You suspect shared representations hurt performance
    - You're experimenting with different architectures
    
    For glucose control, separate networks can sometimes
    perform better as the value function and policy may
    need different state representations.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 64,
        actor_layers: int = 2,
        critic_layers: int = 2,
        log_std_min: float = -20.0,
        log_std_max: float = 2.0
    ):
        """
        Initialise separate Actor and Critic networks.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            hidden_dim: Size of hidden layers
            actor_layers: Number of hidden layers in actor
            critic_layers: Number of hidden layers in critic
            log_std_min: Minimum log standard deviation
            log_std_max: Maximum log standard deviation
        """
        super(SeparateActorCritic, self).__init__()
        
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        # Build actor network
        actor_layers_list = []
        actor_layers_list.append(nn.Linear(state_dim, hidden_dim))
        actor_layers_list.append(nn.Tanh())
        
        for _ in range(actor_layers - 1):
            actor_layers_list.append(nn.Linear(hidden_dim, hidden_dim))
            actor_layers_list.append(nn.Tanh())
        
        self.actor = nn.Sequential(*actor_layers_list)
        self.actor_mean = nn.Linear(hidden_dim, action_dim)
        self.actor_log_std = nn.Parameter(torch.zeros(action_dim))
        
        # Build critic network
        critic_layers_list = []
        critic_layers_list.append(nn.Linear(state_dim, hidden_dim))
        critic_layers_list.append(nn.Tanh())
        
        for _ in range(critic_layers - 1):
            critic_layers_list.append(nn.Linear(hidden_dim, hidden_dim))
            critic_layers_list.append(nn.Tanh())
        
        self.critic = nn.Sequential(*critic_layers_list)
        self.critic_head = nn.Linear(hidden_dim, 1)
        
        # Initialise weights
        self.apply(lambda m: initialise_weights(m, gain=np.sqrt(2)))
        initialise_weights(self.actor_mean, gain=0.01)
        initialise_weights(self.critic_head, gain=1.0)
    
    def forward(
        self,
        state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through both networks."""
        # Actor forward
        actor_features = self.actor(state)
        action_mean = self.actor_mean(actor_features)
        
        log_std = torch.clamp(self.actor_log_std, self.log_std_min, self.log_std_max)
        action_std = log_std.exp().expand_as(action_mean)
        
        # Critic forward
        critic_features = self.critic(state)
        value = self.critic_head(critic_features)
        
        return action_mean, action_std, value
    
    def get_action(
        self,
        state: np.ndarray,
        deterministic: bool = False
    ) -> Tuple[np.ndarray, float, float]:
        """Sample action from policy."""
        if isinstance(state, np.ndarray):
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
        else:
            state_tensor = state.unsqueeze(0) if state.dim() == 1 else state
        
        with torch.no_grad():
            action_mean, action_std, value = self.forward(state_tensor)
            
            dist = Normal(action_mean, action_std)
            
            if deterministic:
                action = action_mean
            else:
                action = dist.sample()
            
            log_prob = dist.log_prob(action).sum(dim=-1)
        
        return (
            action.squeeze(0).numpy(),
            log_prob.item(),
            value.squeeze().item()
        )
    
    def evaluate_actions(
        self,
        states: torch.Tensor,
        actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate actions for PPO update."""
        action_mean, action_std, values = self.forward(states)
        
        dist = Normal(action_mean, action_std)
        log_probs = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        
        return log_probs, values.squeeze(-1), entropy
    
    def get_value(self, state: np.ndarray) -> float:
        """Get state value estimate."""
        if isinstance(state, np.ndarray):
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
        else:
            state_tensor = state.unsqueeze(0) if state.dim() == 1 else state
        
        with torch.no_grad():
            _, _, value = self.forward(state_tensor)
        
        return value.squeeze().item()


def test_networks():
    """Test the neural network implementations."""
    print("Testing Neural Networks...")
    print("=" * 60)
    
    state_dim = 8  # Glucose environment state dimension
    action_dim = 1  # Insulin dose
    batch_size = 32
    
    # Test ContinuousActorCritic
    print("\n1. Testing ContinuousActorCritic (shared layers)...")
    
    network = ContinuousActorCritic(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=64,
        shared_layers=True
    )
    
    # Test forward pass
    test_state = torch.randn(batch_size, state_dim)
    action_mean, action_std, value = network(test_state)
    
    print(f"   Input shape: {test_state.shape}")
    print(f"   Action mean shape: {action_mean.shape}")
    print(f"   Action std shape: {action_std.shape}")
    print(f"   Value shape: {value.shape}")
    
    # Test get_action
    single_state = np.random.randn(state_dim).astype(np.float32)
    action, log_prob, val = network.get_action(single_state)
    
    print(f"\n   Single state action sampling:")
    print(f"   Action: {action}")
    print(f"   Log prob: {log_prob:.4f}")
    print(f"   Value: {val:.4f}")
    
    # Test evaluate_actions
    test_actions = torch.randn(batch_size, action_dim)
    log_probs, values, entropy = network.evaluate_actions(test_state, test_actions)
    
    print(f"\n   Batch evaluation:")
    print(f"   Log probs shape: {log_probs.shape}")
    print(f"   Values shape: {values.shape}")
    print(f"   Entropy shape: {entropy.shape}")
    print(f"   Mean entropy: {entropy.mean().item():.4f}")
    
    # Test SeparateActorCritic
    print("\n2. Testing SeparateActorCritic...")
    
    separate_network = SeparateActorCritic(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=64
    )
    
    action_mean, action_std, value = separate_network(test_state)
    print(f"   Action mean shape: {action_mean.shape}")
    print(f"   Value shape: {value.shape}")
    
    # Test deterministic vs stochastic actions
    print("\n3. Testing deterministic vs stochastic actions...")
    
    actions_stochastic = []
    actions_deterministic = []
    
    for _ in range(10):
        act_stoch, _, _ = network.get_action(single_state, deterministic=False)
        act_det, _, _ = network.get_action(single_state, deterministic=True)
        actions_stochastic.append(act_stoch[0])
        actions_deterministic.append(act_det[0])
    
    print(f"   Stochastic actions (should vary): {[f'{a:.3f}' for a in actions_stochastic]}")
    print(f"   Deterministic actions (should be same): {[f'{a:.3f}' for a in actions_deterministic]}")
    
    # Count parameters
    total_params = sum(p.numel() for p in network.parameters())
    trainable_params = sum(p.numel() for p in network.parameters() if p.requires_grad)
    
    print(f"\n4. Network statistics:")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    
    print("\n" + "=" * 60)
    print("✓ All network tests passed!")


if __name__ == "__main__":
    test_networks()