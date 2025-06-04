import os
import torch
import numpy as np
import logging

def extract_view_type(filepath_or_datapath):
    """
    Extract the view type from a filepath or data path.
    Examples: 
    - /path/to/sagittal/vp_123.npy -> 'sagittal'
    - /path/to/coronal/data -> 'coronal'
    - /path/to/axial/file.npy -> 'axial'
    """
    path = str(filepath_or_datapath).lower()
    print(f"DEBUG-VIEW: Extracting view type from path: {path}")
    
    if 'sagittal' in path:
        view_type = 'sagittal'
    elif 'coronal' in path:
        view_type = 'coronal'
    elif 'axial' in path:
        view_type = 'axial'
    else:
        # Default fallback
        print(f"DEBUG-VIEW: WARNING - Could not determine view type from path, defaulting to 'sagittal'")
        view_type = 'sagittal'
    
    print(f"DEBUG-VIEW: Extracted view type: {view_type}")
    return view_type

def create_view_condition(filepath_or_datapath, target_shape):
    """
    Create a condition tensor based on the view type from filepath or data path.
    
    Args:
        filepath_or_datapath: Path to the file or data directory containing view information
        target_shape: Target shape for the condition tensor (channels, height, width)
        
    Returns:
        Torch tensor with view-specific patterns
    """
    # Extract view type from path
    view_type = extract_view_type(filepath_or_datapath)
    print(f"DEBUG-VIEW: Creating condition for view type: {view_type}")
    
    # Unpack target shape
    if len(target_shape) == 3:
        channels, height, width = target_shape
    elif len(target_shape) == 4:
        batch_size, channels, height, width = target_shape
    else:
        raise ValueError(f"Expected target_shape to have 3 or 4 dimensions, got {len(target_shape)}")
    
    # Create condition tensor
    condition = torch.zeros(channels, height, width, dtype=torch.float32)
    
    # Map view types to numerical values
    view_mapping = {
        'sagittal': 0.0,   # Center value
        'coronal': 1.0,    # High value  
        'axial': -1.0      # Low value
    }
    
    view_value = view_mapping.get(view_type, 0.0)
    print(f"DEBUG-VIEW: View value for {view_type}: {view_value}")
    
    # Create coordinate grids
    y_positions = torch.linspace(0, 1, height).view(-1, 1).expand(-1, width)
    x_positions = torch.linspace(0, 1, width).view(1, -1).expand(height, -1)
    
    # Create different patterns for each channel based on view type
    for c in range(channels):
        if view_type == 'sagittal':
            # Sagittal: vertical patterns (side view, so emphasize vertical structure)
            if c == 0:
                # Vertical sinusoidal pattern
                pattern = torch.sin(y_positions * 4 * np.pi) * 0.5
                # Add slight horizontal modulation
                pattern = pattern + torch.sin(x_positions * 2 * np.pi) * 0.3
            elif c == 1:
                # Vertical gradient with modulation
                pattern = y_positions * 0.7 + torch.cos(y_positions * 6 * np.pi) * 0.3
            elif c == 2:
                # Radial pattern centered vertically
                radial = ((x_positions - 0.5)**2 + (y_positions - 0.5)**2).sqrt()
                pattern = torch.cos(radial * 8 * np.pi) * 0.5
            else:
                # Additional channels: combine patterns
                pattern = torch.sin(y_positions * 6 * np.pi) * torch.cos(x_positions * 2 * np.pi)
                
        elif view_type == 'coronal':
            # Coronal: horizontal patterns (front view, emphasize horizontal/depth structure)
            if c == 0:
                # Horizontal sinusoidal pattern
                pattern = torch.sin(x_positions * 4 * np.pi) * 0.5
                # Add slight vertical modulation
                pattern = pattern + torch.cos(y_positions * 2 * np.pi) * 0.3
            elif c == 1:
                # Horizontal gradient with modulation
                pattern = x_positions * 0.7 + torch.sin(x_positions * 6 * np.pi) * 0.3
            elif c == 2:
                # Concentric circles (radial from center)
                radial = ((x_positions - 0.5)**2 + (y_positions - 0.5)**2).sqrt()
                pattern = torch.sin(radial * 10 * np.pi) * 0.6
            else:
                # Additional channels: horizontal emphasis
                pattern = torch.cos(x_positions * 6 * np.pi) * torch.sin(y_positions * 2 * np.pi)
                
        elif view_type == 'axial':
            # Axial: radial/circular patterns (top view, emphasize radial structure)
            center_x, center_y = 0.5, 0.5
            if c == 0:
                # Strong radial pattern
                radial = ((x_positions - center_x)**2 + (y_positions - center_y)**2).sqrt()
                pattern = torch.sin(radial * 8 * np.pi) * 0.7
            elif c == 1:
                # Angular pattern (like spokes)
                angle = torch.atan2(y_positions - center_y, x_positions - center_x)
                pattern = torch.sin(angle * 6) * 0.6
                # Add radial modulation
                radial = ((x_positions - center_x)**2 + (y_positions - center_y)**2).sqrt()
                pattern = pattern * (1 - radial * 0.5)
            elif c == 2:
                # Concentric rings
                radial = ((x_positions - center_x)**2 + (y_positions - center_y)**2).sqrt()
                pattern = torch.cos(radial * 12 * np.pi) * 0.6
                # Add spiral component
                angle = torch.atan2(y_positions - center_y, x_positions - center_x)
                pattern = pattern + torch.sin(angle + radial * 6 * np.pi) * 0.3
            else:
                # Additional channels: complex radial patterns
                radial = ((x_positions - center_x)**2 + (y_positions - center_y)**2).sqrt()
                angle = torch.atan2(y_positions - center_y, x_positions - center_x)
                pattern = torch.sin(radial * 10 * np.pi + angle * 4) * 0.5
        
        condition[c] = pattern
    
    # Apply global view-specific modulation
    # This gives each view type a distinct overall intensity/character
    if view_type == 'sagittal':
        # Sagittal: moderate intensity, slight bias toward center
        global_modulation = 0.8 + 0.2 * torch.sin(y_positions * np.pi)
    elif view_type == 'coronal':
        # Coronal: higher intensity, bias toward horizontal center
        global_modulation = 1.0 + 0.3 * torch.sin(x_positions * np.pi)
    elif view_type == 'axial':
        # Axial: variable intensity, radial bias
        radial = ((x_positions - 0.5)**2 + (y_positions - 0.5)**2).sqrt()
        global_modulation = 0.9 + 0.4 * torch.cos(radial * 2 * np.pi)
    
    # Apply modulation to all channels
    condition = condition * global_modulation.unsqueeze(0)
    
    # Add view-specific constant bias to distinguish views
    condition = condition + view_value * 0.2
    
    # Normalize and scale the condition
    condition = (condition - condition.mean()) / (condition.std() + 1e-8)
    
    # Apply final scaling to ensure reasonable value range
    condition = torch.clamp(condition * 0.8, -2.0, 2.0)
    
    # Check for NaN values
    if torch.isnan(condition).any():
        print(f"DEBUG-VIEW: ERROR - NaN detected in condition tensor")
        condition = torch.nan_to_num(condition, nan=0.0)
    
    # Stats for debugging
    print(f"DEBUG-VIEW: Condition stats for {view_type} - min: {condition.min().item():.4f}, max: {condition.max().item():.4f}, mean: {condition.mean().item():.4f}, std: {condition.std().item():.4f}")
    
    return condition 