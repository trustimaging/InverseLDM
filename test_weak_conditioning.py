import torch
import numpy as np

# Test weak conditioning effect
print("Testing weak conditioning effect (strength=0.001)")
print("="*60)

# Simulate a typical noisy latent
batch_size = 4
channels = 3
height, width = 64, 64

# Create a noisy latent (typical scale from diffusion process)
noisy_latent = torch.randn(batch_size, channels, height, width) * 2.0
print(f"Noisy latent stats: mean={noisy_latent.mean():.4f}, std={noisy_latent.std():.4f}")

# Create a condition (even with normal scale)
condition = torch.randn(batch_size, channels, height, width)
print(f"Condition stats: mean={condition.mean():.4f}, std={condition.std():.4f}")

# Apply very weak conditioning (strength=0.001)
strength = 0.001
conditioned = noisy_latent + strength * condition

# Calculate the relative change
absolute_change = torch.abs(conditioned - noisy_latent)
relative_change = absolute_change.mean() / noisy_latent.std()

print(f"\nWith strength={strength}:")
print(f"  Absolute change: mean={absolute_change.mean():.6f}, max={absolute_change.max():.6f}")
print(f"  Relative change: {relative_change.item():.4%}")
print(f"  Conditioned stats: mean={conditioned.mean():.4f}, std={conditioned.std():.4f}")

# Compare with different strengths
print("\nComparison with different strength values:")
print("-"*60)
for test_strength in [0.001, 0.01, 0.1, 0.5, 1.0]:
    test_conditioned = noisy_latent + test_strength * condition
    test_change = torch.abs(test_conditioned - noisy_latent).mean() / noisy_latent.std()
    print(f"Strength={test_strength:>5}: relative change = {test_change.item():>7.3%}")

print("\nConclusion: With strength=0.001, the conditioning has minimal impact (<0.1% change)")
print("This should allow the model to train mostly unconditioned while still having the conditioning pipeline active.") 