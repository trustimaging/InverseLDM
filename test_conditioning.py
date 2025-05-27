import torch
import matplotlib.pyplot as plt
import numpy as np
from invldm.utils.slice_condition import create_slice_condition

# Test the slice conditioning for different slice numbers
slice_numbers = [140, 160, 180, 200]
target_shape = (3, 64, 64)  # 3 channels, 64x64 spatial

fig, axes = plt.subplots(len(slice_numbers), 4, figsize=(12, 12))
fig.suptitle('Improved Slice Conditioning Patterns', fontsize=16)

for i, slice_num in enumerate(slice_numbers):
    # Create a dummy filename with the slice number
    dummy_filename = f"vp_123456_sagittal_{slice_num}.npy"
    
    # Generate condition
    condition = create_slice_condition(dummy_filename, target_shape)
    
    # Plot each channel and the mean
    for j in range(3):
        im = axes[i, j].imshow(condition[j].numpy(), cmap='RdBu', vmin=-2, vmax=2)
        axes[i, j].set_title(f'Slice {slice_num}, Channel {j}')
        axes[i, j].axis('off')
        
    # Plot mean across channels
    mean_condition = condition.mean(dim=0).numpy()
    im = axes[i, 3].imshow(mean_condition, cmap='RdBu', vmin=-2, vmax=2)
    axes[i, 3].set_title(f'Slice {slice_num}, Mean')
    axes[i, 3].axis('off')

# Add colorbar
plt.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6)
plt.tight_layout()
plt.savefig('improved_conditioning.png', dpi=150, bbox_inches='tight')
plt.close()

# Test conditioning strength effect
print("\nTesting conditioning strength effect:")
print("="*50)

# Simulate noisy latent and condition
noisy_latent = torch.randn(1, 3, 64, 64) * 2.0  # Typical latent scale
condition = create_slice_condition("vp_123456_sagittal_170.npy", (3, 64, 64))

# Test different conditioning strengths
strengths = [0.3, 0.5, 0.7, 1.0]

for strength in strengths:
    # Scale condition to match noisy latent scale
    noisy_std = torch.std(noisy_latent)
    cond_std = torch.std(condition)
    
    if cond_std > 0:
        condition_scaled = condition * (noisy_std / cond_std)
    else:
        condition_scaled = condition
        
    # Apply conditioning
    conditioned = noisy_latent + strength * condition_scaled.unsqueeze(0)
    
    print(f"\nStrength = {strength}:")
    print(f"  Noisy latent: mean={noisy_latent.mean():.4f}, std={noisy_std:.4f}")
    print(f"  Condition: mean={condition.mean():.4f}, std={cond_std:.4f}")
    print(f"  Scaled condition: mean={condition_scaled.mean():.4f}, std={condition_scaled.std():.4f}")
    print(f"  Conditioned result: mean={conditioned.mean():.4f}, std={conditioned.std():.4f}")
    
    # Check how much the conditioning changes the signal
    signal_change = torch.abs(conditioned - noisy_latent).mean() / noisy_std
    print(f"  Relative signal change: {signal_change.item():.2%}")

print("\nConditioning test complete!") 