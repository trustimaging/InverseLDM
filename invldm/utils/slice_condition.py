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
    Create a condition tensor based on the slice number.
    
    Args:
        filepath: Path to the file containing the slice number in its name
        target_shape: The shape of the output tensor (channel, height, width)
    
    Returns:
        A tensor with a single channel filled with the normalized slice number
    """
    print(f"DEBUG-SLICE: Creating condition tensor for {filepath} with shape {target_shape}")
    slice_num = extract_slice_number(filepath)
    
    # Create a tensor with the same spatial dimensions as the input
    # but with a single channel filled with the slice number
    condition = torch.zeros(target_shape)
    
    # Normalize the slice number to [0, 1] for the range we expect to see
    # Adjusted for slice values typically between 140 and 200
    min_slice = 160
    max_slice = 240
    
    # Check for potential division by zero
    if max_slice == min_slice:
        print(f"DEBUG-SLICE: ERROR - min_slice equals max_slice ({min_slice}), would cause division by zero")
        normalized_value = 0.5  # fallback value
    else:
        normalized_value = (slice_num - min_slice) / (max_slice - min_slice)
    
    # Clamp to [0, 1] to avoid extreme values
    normalized_value = max(0.0, min(1.0, normalized_value))
    
    print(f"DEBUG-SLICE: Slice number {slice_num} normalized to {normalized_value} (range: {min_slice}-{max_slice})")
    
    # Fill the tensor with the normalized slice number
    condition.fill_(normalized_value)
    
    # Check for NaN values in the tensor
    if torch.isnan(condition).any():
        print(f"DEBUG-SLICE: ERROR - NaN detected in condition tensor")
    
    # Stats for debugging
    print(f"DEBUG-SLICE: Condition tensor min: {condition.min().item()}, max: {condition.max().item()}")
    
    return condition 