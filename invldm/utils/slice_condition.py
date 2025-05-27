import os
import torch
import numpy as np
import logging

def extract_slice_number(filepath):
    """
    Extract the slice number from a filename.
    Example: vp_996782_sagittal_168.npy -> 168
    """
    filename = os.path.basename(filepath)
    print(f"DEBUG-SLICE: Extracting slice number from filename: {filename}")
    name_parts = filename.split('_')
    
    # Print each part for debugging
    print(f"DEBUG-SLICE: Filename parts: {name_parts}")
    
    # Extract the part before the file extension
    if len(name_parts) >= 4:
        # For files with format: vp_996782_sagittal_168.npy
        slice_num_with_ext = name_parts[3]
        slice_num = slice_num_with_ext.split('.')[0]
        print(f"DEBUG-SLICE: Found slice number in position 3: {slice_num}")
    else:
        # Fallback for other filename formats
        name_without_ext = os.path.splitext(filename)[0]
        # Try to get the last part as the slice number
        slice_num = name_without_ext.split('_')[-1]
        print(f"DEBUG-SLICE: Using fallback, found slice: {slice_num}")
    
    try:
        slice_int = int(slice_num)
        print(f"DEBUG-SLICE: Successfully converted to int: {slice_int}")
        return slice_int
    except ValueError:
        # If conversion fails, return a default value
        print(f"DEBUG-SLICE: WARNING - Could not convert '{slice_num}' to integer, using default 0")
        return 0

def create_slice_condition(filepath, target_shape):
    """
    Create a condition tensor based on the slice number from filename.
    
    Args:
        filepath: Path to the file containing slice information
        target_shape: Target shape for the condition tensor (channels, height, width)
        
    Returns:
        Torch tensor with spatially varying patterns based on slice number
    """
    # Extract slice number from filename
    slice_num = extract_slice_number(filepath)
    print(f"DEBUG-SLICE: Extracted slice number: {slice_num}")
    
    # Unpack target shape
    if len(target_shape) == 3:
        channels, height, width = target_shape
    elif len(target_shape) == 4:
        batch_size, channels, height, width = target_shape
    else:
        raise ValueError(f"Expected target_shape to have 3 or 4 dimensions, got {len(target_shape)}")
    
    # Create condition tensor
    condition = torch.zeros(channels, height, width, dtype=torch.float32)
    
    # Normalize slice number to [0, 1] range
    # Assuming typical brain scans have 100-300 slices
    max_slice = 300
    normalized_value = min(slice_num / max_slice, 1.0)
    
    print(f"DEBUG-SLICE: Normalized slice value: {normalized_value}")
    
    # Create coordinate grids
    y_positions = torch.linspace(0, 1, height).view(-1, 1).expand(-1, width)
    x_positions = torch.linspace(0, 1, width).view(1, -1).expand(height, -1)
    
    # Create different patterns for each channel with stronger variation
    for c in range(channels):
        # Create a radial pattern with frequency based on slice number
        freq = 2.0 + normalized_value * 8.0  # Higher frequency for more variation
        radial = ((x_positions - 0.5)**2 + (y_positions - 0.5)**2).sqrt() * freq
        
        # Create patterns with phase based on slice number
        phase = normalized_value * 2 * np.pi
        
        if c == 0:
            # Channel 0: Radial waves with stronger amplitude
            pattern = torch.sin(radial * np.pi + phase)
            # Add diagonal gradient
            diagonal = (x_positions + y_positions) / 2
            pattern = pattern * 0.7 + diagonal * normalized_value * 0.3
        elif c == 1:
            # Channel 1: Horizontal waves modulated by slice number
            horizontal_freq = 3.0 + normalized_value * 5.0
            pattern = torch.sin(x_positions * horizontal_freq * np.pi + phase)
            # Add vertical gradient
            pattern = pattern * 0.7 + y_positions * (1 - normalized_value) * 0.3
        elif c == 2:
            # Channel 2: Vertical waves modulated by slice number
            vertical_freq = 3.0 + (1 - normalized_value) * 5.0
            pattern = torch.cos(y_positions * vertical_freq * np.pi - phase)
            # Add horizontal gradient
            pattern = pattern * 0.7 + x_positions * normalized_value * 0.3
        else:
            # Additional channels: Complex patterns
            pattern = torch.sin(radial * np.pi + phase * c) * torch.cos(x_positions * 4 * np.pi)
            pattern = pattern + torch.sin(y_positions * 4 * np.pi + phase)
        
        condition[c] = pattern
    
    # Scale the condition to have reasonable magnitude
    # Normalize to [-1, 1] range first
    condition = (condition - condition.mean()) / (condition.std() + 1e-8)
    
    # Apply a global modulation based on slice position
    # This creates different overall intensities for different slices
    slice_modulation = 0.5 + 0.5 * torch.sin(torch.tensor(normalized_value * np.pi))
    condition = condition * slice_modulation
    
    # Ensure reasonable value range
    condition = torch.clamp(condition, -2.0, 2.0)
    
    # Check for NaN values in the tensor
    if torch.isnan(condition).any():
        print(f"DEBUG-SLICE: ERROR - NaN detected in condition tensor")
        # Replace NaN values with zeros
        condition = torch.nan_to_num(condition, nan=0.0)
    
    # Stats for debugging
    print(f"DEBUG-SLICE: Condition stats - min: {condition.min().item():.4f}, max: {condition.max().item():.4f}, mean: {condition.mean().item():.4f}, std: {condition.std().item():.4f}")
    
    return condition 