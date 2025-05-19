import os
import torch
import numpy as np

def extract_slice_number(filepath):
    """
    Extract the slice number from a filename.
    Example: vp_996782_sagittal_168.npy -> 168
    """
    filename = os.path.basename(filepath)
    name_parts = filename.split('_')
    
    # Extract the part before the file extension
    if len(name_parts) >= 4:
        # For files with format: vp_996782_sagittal_168.npy
        slice_num_with_ext = name_parts[3]
        slice_num = slice_num_with_ext.split('.')[0]
    else:
        # Fallback for other filename formats
        name_without_ext = os.path.splitext(filename)[0]
        # Try to get the last part as the slice number
        slice_num = name_without_ext.split('_')[-1]
    
    try:
        return int(slice_num)
    except ValueError:
        # If conversion fails, return a default value
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
    slice_num = extract_slice_number(filepath)
    
    # Create a tensor with the same spatial dimensions as the input
    # but with a single channel filled with the slice number
    condition = torch.zeros(target_shape)
    
    # Normalize the slice number to [0, 1] assuming maximum slice number around 256
    # This can be adjusted based on actual data ranges
    normalized_value = (slice_num - 160) / (241 - 160)
    
    # Fill the tensor with the normalized slice number
    condition.fill_(normalized_value)
    
    return condition 