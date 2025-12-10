"""
Proximal Policy Optimization (PPO) Agent
=========================================

This is the HEART of your tutorial. Every line is crucial.

Key Innovation: The Clipped Surrogate Objective
- Prevents destructively large policy updates
- Simple to implement (just a clipping operation!)
- Achieves TRPO's stability without complex math

The algorithm:
1. Collect data using current policy
2. Compute advantages (how good were the actions?)
3. Update policy multiple times using the SAME data (key difference from vanilla PG!)
4. Use clipping to keep updates safe
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque


class PPOAgent:
    """
    Proximal Policy Optimization Agent
    
    This class implements the complete PPO algorithm including:
    - Clipped surrogate objective (the star of the show!)
    - Generalized Advantage Estimation (GAE)
    - Value function learning
    - Entropy bonus for exploration
    """
    
    def __init__(
        self,
        network,
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        epsilon=0.2,
        value_coef=0.5,
        entropy_coef=0.01,
        max_grad_norm=0.5,
        update_epochs=10,
        batch_size=64
    ):
        """
        Args:
            network: ActorCritic neural network
            lr: Learning rate
            gamma: Discount factor (how much we care about future rewards)
            gae_lambda: GAE parameter (bias-variance tradeoff)
            epsilon: PPO clipping parameter (typically 0.1-0.3)
            value_coef: Weight for value loss in total loss
            entropy_coef: Weight for entropy bonus (encourages exploration)
            max_grad_norm: Gradient clipping threshold
            update_epochs: How many times to update on each batch of data
            batch_size: Minibatch size for updates
        """
        self.network = network
        self.optimizer = optim.Adam(network.parameters(), lr=lr)
        
        # PPO hyperparameters
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.epsilon = epsilon  # THE KEY PARAMETER!
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.update_epochs = update_epochs
        self.batch_size = batch_size
        
        # Storage for collected experience
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        
        # Metrics for visualization
        self.metrics = {
            'policy_loss': [],
            'value_loss': [],
            'entropy': [],
            'ratio_mean': [],
            'ratio_std': [],
            'clipped_fraction': [],
            'kl_divergence': [],
            'explained_variance': [],
            'advantages_mean': [],
            'advantages_std': []
        }
    
    def store_transition(self, state, action, reward, value, log_prob, done):
        """
        Store a single transition (s, a, r, v, log_p, done)
        
        This is called after each environment step during data collection
        """
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)
    
    def compute_gae(self, next_value):
        """
        Compute Generalized Advantage Estimation (GAE)
        
        GAE is a way to estimate advantages that balances:
        - Low bias (accurate estimates)
        - Low variance (stable learning)
        
        Formula: A_t = δ_t + (γλ)δ_{t+1} + (γλ)²δ_{t+2} + ...
        where δ_t = r_t + γV(s_{t+1}) - V(s_t) is the TD error
        
        Why GAE?
        - λ=0: Low variance, high bias (only 1-step TD error)
        - λ=1: High variance, low bias (Monte Carlo return)
        - λ=0.95: Sweet spot (typically)
        
        Args:
            next_value: Value estimate for the state after the last stored state
        
        Returns:
            advantages: numpy array of advantage estimates
            returns: numpy array of return targets for value function
        """
        # Convert lists to numpy arrays
        rewards = np.array(self.rewards)
        values = np.array(self.values)
        dones = np.array(self.dones)
        
        # Initialize
        advantages = np.zeros_like(rewards)
        last_gae = 0
        
        # Compute GAE backwards through time
        # We go backwards because each advantage depends on future advantages
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                # Last timestep: use next_value as the bootstrap value
                next_value_t = next_value
                next_non_terminal = 1.0 - dones[t]
            else:
                # Other timesteps: use stored value
                next_value_t = values[t + 1]
                next_non_terminal = 1.0 - dones[t]
            
            # TD error: δ_t = r_t + γV(s_{t+1}) - V(s_t)
            # This measures how much better/worse the actual reward was vs expectation
            delta = rewards[t] + self.gamma * next_value_t * next_non_terminal - values[t]
            
            # GAE: A_t = δ_t + (γλ)δ_{t+1} + (γλ)²δ_{t+2} + ...
            # We compute this recursively: A_t = δ_t + (γλ)A_{t+1}
            advantages[t] = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae
            last_gae = advantages[t]
        
        # Returns are advantages + values
        # These are the targets for the value function
        # R_t = A_t + V(s_t) ≈ actual return from state s_t
        returns = advantages + values
        
        return advantages, returns
    
    def compute_ppo_loss(self, states, actions, old_log_probs, advantages, returns):
        """
        THE HEART OF PPO: Compute the clipped surrogate objective
        
        This is the key innovation that makes PPO work!
        
        The formula (Equation 7 from paper):
        L^CLIP(θ) = E[min(r_t(θ)A_t, clip(r_t(θ), 1-ε, 1+ε)A_t)]
        
        Where:
        - r_t(θ) = π_θ(a|s) / π_θ_old(a|s) is the probability ratio
        - A_t is the advantage
        - clip() constrains the ratio to [1-ε, 1+ε]
        - min() takes the pessimistic bound
        
        Args:
            states: Batch of states
            actions: Batch of actions taken
            old_log_probs: Log probs from old policy (when data was collected)
            advantages: Advantage estimates
            returns: Return targets for value function
        
        Returns:
            total_loss: Combined loss for optimization
            metrics: Dictionary of metrics for logging/visualization
        """
        # Evaluate actions with current policy
        # This gives us the NEW policy's perspective on the old actions
        new_log_probs, values, entropy = self.network.evaluate_actions(states, actions)
        
        # ============================================================
        # STEP 1: Compute the probability ratio
        # ============================================================
        # r(θ) = π_θ(a|s) / π_θ_old(a|s)
        # 
        # We work in log space for numerical stability:
        # r = exp(log π_θ(a|s) - log π_θ_old(a|s))
        #
        # What does this ratio mean?
        # - ratio = 1.0: policy hasn't changed
        # - ratio > 1.0: new policy is MORE likely to take this action
        # - ratio < 1.0: new policy is LESS likely to take this action
        ratio = torch.exp(new_log_probs - old_log_probs)
        
        # ============================================================
        # STEP 2: Compute the two surrogate objectives
        # ============================================================
        # Unclipped objective: r(θ) * A_t
        # This is what vanilla policy gradient would maximize
        surrogate1 = ratio * advantages
        
        # Clipped objective: clip(r(θ), 1-ε, 1+ε) * A_t
        # This constrains how much the policy can change
        ratio_clipped = torch.clamp(ratio, 1.0 - self.epsilon, 1.0 + self.epsilon)
        surrogate2 = ratio_clipped * advantages
        
        # ============================================================
        # STEP 3: Take the minimum (pessimistic bound)
        # ============================================================
        # L^CLIP = E[min(surrogate1, surrogate2)]
        #
        # Why min()?
        # - For good actions (A > 0): 
        #   If ratio > 1+ε, we clip it. Don't increase probability too much!
        # - For bad actions (A < 0):
        #   If ratio < 1-ε, we clip it. Don't decrease probability too much!
        #
        # This creates a "trust region" without complex constrained optimization
        policy_loss = -torch.min(surrogate1, surrogate2).mean()
        
        # Note the negative sign: we maximize the objective = minimize the negative
        
        # ============================================================
        # STEP 4: Value function loss
        # ============================================================
        # The critic should predict returns accurately
        # We use MSE: L_V = (V(s) - R)²
        value_loss = 0.5 * ((values - returns) ** 2).mean()
        
        # ============================================================
        # STEP 5: Entropy bonus
        # ============================================================
        # Entropy = -Σ p(a) log p(a)
        # Higher entropy = more exploration
        # We WANT high entropy early in training, so we ADD it as a bonus
        # (with a small coefficient so it doesn't dominate)
        entropy_loss = -entropy.mean()  # Negative because we want to maximize entropy
        
        # ============================================================
        # STEP 6: Combine losses
        # ============================================================
        # Total loss = policy_loss + c1 * value_loss + c2 * entropy_loss
        total_loss = (
            policy_loss + 
            self.value_coef * value_loss + 
            self.entropy_coef * entropy_loss
        )
        
        # ============================================================
        # STEP 7: Compute metrics for visualization/debugging
        # ============================================================
        with torch.no_grad():
            # What fraction of ratios were clipped?
            # High early (policy wants to change a lot), low later (policy converging)
            clipped = ((ratio < 1.0 - self.epsilon) | (ratio > 1.0 + self.epsilon)).float()
            clipped_fraction = clipped.mean().item()
            
            # KL divergence: another measure of policy change
            # KL(π_old || π_new) ≈ (log_prob_old - log_prob_new) * π_old
            # For small changes: KL ≈ 0.5 * (ratio - 1)²
            approx_kl = ((ratio - 1.0) - torch.log(ratio)).mean().item()
            
            # Explained variance: how well does value function predict returns?
            # 1.0 = perfect prediction, 0.0 = no better than predicting mean
            y_pred = values.detach().cpu().numpy()
            y_true = returns.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = 1 - np.var(y_true - y_pred) / (var_y + 1e-8)
        
        # Package metrics
        metrics = {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': entropy.mean().item(),
            'ratio_mean': ratio.mean().item(),
            'ratio_std': ratio.std().item(),
            'clipped_fraction': clipped_fraction,
            'kl_divergence': approx_kl,
            'explained_variance': explained_var
        }
        
        return total_loss, metrics
    
    def update(self):
        """
        Update the policy using collected experience
        
        This is called after collecting a batch of trajectories.
        
        Key difference from vanilla policy gradient:
        - Vanilla PG: Use data ONCE, then throw it away
        - PPO: Use data MULTIPLE times (update_epochs times)
        - Clipping makes this safe!
        
        Steps:
        1. Compute advantages using GAE
        2. Convert data to tensors
        3. For multiple epochs:
           a. Shuffle data
           b. Update on minibatches
        4. Clear storage
        """
        # Check if we have data
        if len(self.states) == 0:
            return
        
        # ============================================================
        # STEP 1: Compute advantages
        # ============================================================
        # Get value of next state (for bootstrapping)
        last_state = torch.FloatTensor(self.states[-1]).unsqueeze(0)
        with torch.no_grad():
            _, next_value = self.network(last_state)
            next_value = next_value.item()
        
        # Compute GAE
        advantages, returns = self.compute_gae(next_value)
        
        # Normalize advantages (common practice, improves stability)
        # After normalization: mean ≈ 0, std ≈ 1
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Store for metrics
        self.metrics['advantages_mean'].append(advantages.mean())
        self.metrics['advantages_std'].append(advantages.std())
        
        # ============================================================
        # STEP 2: Convert to tensors
        # ============================================================
        states = torch.FloatTensor(np.array(self.states))
        actions = torch.LongTensor(np.array(self.actions))
        old_log_probs = torch.FloatTensor(np.array(self.log_probs))
        advantages = torch.FloatTensor(advantages)
        returns = torch.FloatTensor(returns)
        
        # ============================================================
        # STEP 3: Multiple epochs of optimization
        # ============================================================
        # This is the key that allows PPO to be sample-efficient!
        # We use the same data multiple times, but clipping keeps us safe
        
        num_samples = states.shape[0]
        num_batches = num_samples // self.batch_size
        
        for epoch in range(self.update_epochs):
            # Shuffle indices for minibatch sampling
            indices = np.random.permutation(num_samples)
            
            # Iterate over minibatches
            for i in range(num_batches):
                # Get minibatch indices
                batch_indices = indices[i * self.batch_size:(i + 1) * self.batch_size]
                
                # Extract minibatch
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                
                # Compute loss
                loss, metrics = self.compute_ppo_loss(
                    batch_states,
                    batch_actions,
                    batch_old_log_probs,
                    batch_advantages,
                    batch_returns
                )
                
                # Gradient descent
                self.optimizer.zero_grad()
                loss.backward()
                
                # Clip gradients to prevent exploding gradients
                # This is another stabilization technique
                nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                
                self.optimizer.step()
                
                # Store metrics (only from last batch of last epoch for simplicity)
                if epoch == self.update_epochs - 1 and i == num_batches - 1:
                    for key, value in metrics.items():
                        self.metrics[key].append(value)
        
        # ============================================================
        # STEP 4: Clear storage for next batch
        # ============================================================
        self.clear_storage()
    
    def clear_storage(self):
        """Clear the experience buffer"""
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
    
    def get_metrics(self):
        """Return metrics for logging/visualization"""
        return self.metrics


# Quick test
if __name__ == "__main__":
    from networks import ActorCritic
    
    print("Testing PPO Agent...")
    
    # Create network and agent
    network = ActorCritic(state_dim=4, action_dim=2)
    agent = PPOAgent(network, epsilon=0.2)
    
    # Simulate collecting some data
    for _ in range(10):
        state = np.random.randn(4)
        action = np.random.randint(0, 2)
        reward = np.random.randn()
        value = np.random.randn()
        log_prob = np.random.randn()
        done = False
        
        agent.store_transition(state, action, reward, value, log_prob, done)
    
    # Test update
    print(f"Stored {len(agent.states)} transitions")
    agent.update()
    print("Update completed successfully!")
    
    print("\n✓ PPO Agent tests passed!")