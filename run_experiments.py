"""
Quick Experiments - Just Epsilon Comparison
"""

from experiments import PPOExperiment

# Quick epsilon experiment (most important!)
experiment = PPOExperiment(
    env_name="CartPole-v1",
    base_timesteps=30000,  # Shorter for speed
    num_seeds=2  # Fewer seeds for speed
)

print("Running quick epsilon comparison...")
print("This will take about 10-15 minutes\n")

results = experiment.experiment_epsilon(epsilon_values=[0.1, 0.2, 0.3])

print(f"\n✓ Done! Check: {experiment.experiment_dir}")
print("You now have epsilon_comparison.png for your tutorial!")