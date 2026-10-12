"""
scaling.py - Output scaling for training, and undoing it for evaluation.
"""
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def fit_scaling(train_output_at_x, scale_type=None, **kwargs):
    """Fit an output scaling on {key: array(n_samples, n_features)}.

    scale_type: None/'none', 'div_by_max', 'standardize' or 'minmax'.
    Returns the scaling config that is stored on the engine.
    """
    config = {'type': scale_type}
    if scale_type is None or scale_type == 'none':
        config['type'] = 'none'
        return config
        
    # one scaler for all outputs together
    all_vals = np.concatenate([arr.ravel() for arr in train_output_at_x.values()]).reshape(-1, 1)
    
    if scale_type == 'div_by_max':
        max_val = np.max(np.abs(all_vals))
        config['max_val'] = float(max_val) if max_val > 0 else 1.0
        
    elif scale_type == 'standardize':
        scaler = StandardScaler()
        scaler.fit(all_vals)
        config['scaler'] = scaler
        
    elif scale_type == 'minmax':
        scaler = MinMaxScaler()
        scaler.fit(all_vals)
        config['scaler'] = scaler
        
    else:
        raise ValueError(f"Unknown scale_type: {scale_type}")
        
    return config

def scale_data(data, config):
    """Scale data (dict of arrays, array or float) with the config."""
    scale_type = config.get('type', 'none')
    if scale_type == 'none' or scale_type is None:
        return data
        
    def transform(val):
        if scale_type == 'div_by_max':
            return val / config['max_val']
        elif scale_type == 'standardize' or scale_type == 'minmax':
            scaler = config['scaler']
            orig_shape = val.shape
            val_flat = val.ravel().reshape(-1, 1)
            val_scaled = scaler.transform(val_flat)
            return val_scaled.reshape(orig_shape)
        return val

    if isinstance(data, dict):
        scaled = {}
        for k, v in data.items():
            scaled[k] = transform(v)
        return scaled
    else:
        return transform(data)

def descale_data(data, config):
    """Undo the scaling (dict of arrays, array or float)."""
    scale_type = config.get('type', 'none')
    if scale_type == 'none' or scale_type is None:
        return data
        
    def inv_transform(val):
        if scale_type == 'div_by_max':
            return val * config['max_val']
        elif scale_type == 'standardize' or scale_type == 'minmax':
            scaler = config['scaler']
            orig_shape = val.shape
            val_flat = val.ravel().reshape(-1, 1)
            val_descaled = scaler.inverse_transform(val_flat)
            return val_descaled.reshape(orig_shape)
        return val

    if isinstance(data, dict):
        descaled = {}
        for k, v in data.items():
            descaled[k] = inv_transform(v)
        return descaled
    else:
        return inv_transform(data)

def descale_std(std_scaled, config):
    """Undo the scaling for a std (only the factor applies, no shift)."""
    scale_type = config.get('type', 'none')
    if scale_type == 'none' or scale_type is None:
        return std_scaled
        
    def inv_transform_std(val):
        if scale_type == 'div_by_max':
            return val * config['max_val']
        elif scale_type == 'standardize':
            scaler = config['scaler']
            return val * scaler.scale_[0]
        elif scale_type == 'minmax':
            scaler = config['scaler']
            return val * scaler.data_range_[0]
        return val

    if isinstance(std_scaled, dict):
        descaled = {}
        for k, v in std_scaled.items():
            descaled[k] = inv_transform_std(v)
        return descaled
    else:
        return inv_transform_std(std_scaled)
