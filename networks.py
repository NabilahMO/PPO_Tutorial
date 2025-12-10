"""
Neural Networks for PPO
=======================
This file contains the policy and value function networks.

Key Concepts:
- Actor (Policy): Maps states → action probabilities
- Critic (Value Function): Maps states → expected return
- We use a shared network for efficiency (optional)
"""

import torch
import torch.nn as nn
from torch.distributions import Categorical
import numpy as np


class ActorCritic(nn.Module):
    """
    Combined Actor-Critic Network
    
    Architecture:
    - Shared feature extractor (2 hidden layers)
    - Actor head: outputs action probabilities
    - Critic head: outputs state value
    
    This is more efficient than separate networks and often works better.
    """
    
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        """
        Args:
            state_dim: Dimension of state space (e.g., 4 for CartPole)
            action_dim: Number of discrete actions (e.g., 2 for CartPole)
            hidden_dim: Size of hidden layers
        """
        super(ActorCritic, self).__init__()
        
        # Shared feature extractor
        # These layers learn useful representations of the state
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),  # Tanh activation is common in RL
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )
        
        # Actor head: produces action logits (unnormalized probabilities)
        # Why logits? More numerically stable than direct probabilities
        self.actor = nn.Linear(hidden_dim, action_dim)
        
        # Critic head: produces value estimate V(s)
        # Single output: expected return from this state
        self.critic = nn.Linear(hidden_dim, 1)
        
        # Initialize weights properly
        # Good initialization helps with training stability
        self._initialize_weights()
    
    def _initialize_weights(self):
        """
        Orthogonal initialization for better training
        This is a best practice from the PPO paper
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0.0)
        
        # Small initialization for policy head (encourages exploration)
        nn.init.orthogonal_(self.actor.weight, gain=0.01)
        # Value head gets standard initialization
        nn.init.orthogonal_(self.critic.weight, gain=1.0)
    
    def forward(self, state):
        """
        Forward pass through the network
        
        Args:
            state: torch.Tensor of shape (batch_size, state_dim)
        
        Returns:
            action_logits: (batch_size, action_dim) - unnormalized action probs
            value: (batch_size, 1) - state value estimate
        """
        # Extract features
        features = self.shared(state)
        
        # Get action logits and value
        action_logits = self.actor(features)
        value = self.critic(features)
        
        return action_logits, value
    
    def get_action(self, state, deterministic=False):
        """
        Sample an action from the policy
        
        Args:
            state: numpy array or torch.Tensor
            deterministic: If True, return argmax action (for evaluation)
        
        Returns:
            action: Selected action (int)
            log_prob: Log probability of the action
            value: State value estimate
        """
        # Convert to tensor if needed
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state).unsqueeze(0)
        
        # Get action logits and value
        action_logits, value = self.forward(state)
        
        # Create categorical distribution from logits
        # This handles the softmax and sampling automatically
        action_dist = Categorical(logits=action_logits)
        
        if deterministic:
            # Take the most likely action (for evaluation)
            action = torch.argmax(action_logits, dim=-1)
        else:
            # Sample from the distribution (for training)
            action = action_dist.sample()
        
        # Get log probability of the action
        # We need this for the PPO loss calculation
        log_prob = action_dist.log_prob(action)
        
        return action.item(), log_prob, value.squeeze()
    
    def evaluate_actions(self, states, actions):
        """
        Evaluate actions that were taken (used during PPO update)
        
        Args:
            states: (batch_size, state_dim)
            actions: (batch_size,) - actions that were taken
        
        Returns:
            log_probs: Log probabilities of the actions
            values: State value estimates
            entropy: Policy entropy (for exploration bonus)
        """
        # Forward pass
        action_logits, values = self.forward(states)
        
        # Create distribution
        action_dist = Categorical(logits=action_logits)
        
        # Get log probabilities of the actions that were taken
        log_probs = action_dist.log_prob(actions)
        
        # Calculate entropy: H = -Σ p(a) log p(a)
        # Higher entropy = more exploration
        # This will decrease as policy becomes more confident
        entropy = action_dist.entropy()
        
        return log_probs, values.squeeze(), entropy


class SeparateActorCritic(nn.Module):
    """
    Alternative: Separate Actor and Critic networks
    
    Use this if:
    - You want different learning rates for actor/critic
    - Your state representation needs are very different
    - You're experimenting with different architectures
    
    Generally, shared networks work better, but this is here for completeness.
    """
    
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(SeparateActorCritic, self).__init__()
        
        # Actor network
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim)
        )
        
        # Critic network
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0.0)
    
    def forward(self, state):
        action_logits = self.actor(state)
        value = self.critic(state)
        return action_logits, value
    
    def get_action(self, state, deterministic=False):
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state).unsqueeze(0)
        
        action_logits, value = self.forward(state)
        action_dist = Categorical(logits=action_logits)
        
        if deterministic:
            action = torch.argmax(action_logits, dim=-1)
        else:
            action = action_dist.sample()
        
        log_prob = action_dist.log_prob(action)
        return action.item(), log_prob, value.squeeze()
    
    def evaluate_actions(self, states, actions):
        action_logits, values = self.forward(states)
        action_dist = Categorical(logits=action_logits)
        log_probs = action_dist.log_prob(actions)
        entropy = action_dist.entropy()
        return log_probs, values.squeeze(), entropy


# Quick test to ensure networks work
if __name__ == "__main__":
    print("Testing ActorCritic network...")
    
    # Create a simple network
    state_dim = 4  # CartPole state dimension
    action_dim = 2  # CartPole has 2 actions
    network = ActorCritic(state_dim, action_dim)
    
    # Test with random state
    test_state = torch.randn(1, state_dim)
    action_logits, value = network(test_state)
    
    print(f"State shape: {test_state.shape}")
    print(f"Action logits shape: {action_logits.shape}")
    print(f"Value shape: {value.shape}")
    print(f"Value estimate: {value.item():.3f}")
    
    # Test action sampling
    action, log_prob, value = network.get_action(test_state)
    print(f"\nSampled action: {action}")
    print(f"Log probability: {log_prob.item():.3f}")
    print(f"Value: {value.item():.3f}")
    
    print("\n✓ Network tests passed!")