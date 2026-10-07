"""Compute surrogate validation metrics (R², RMSE, Pearson-R, NSE, etc.)."""

import numpy as np


def validation_error(true_y, sim_y, sim_std=None, std_metrics=False):
    """
    Estimates different evaluation (validation) criteria for a surrogate model, for each output location. Results for
    each output type are saved under different keys in a dictionary.
    Args:
        true_y: array [mc_valid, n_obs]
            simulator outputs for valid_samples
        sim_y: dict, with an array [mc_valid, n_obs] for each output type.
            surrogate/emulator's outputs for valid_samples.
        sim_std: dict, with an array [mc_valid, n_obs] for each output type.
            Surrogate/emulator standard deviation
        std_metrics: bool
            True to estimate error-based validation criteria. Default is False

    Returns: dict
        with validation criteria, a key for each criteria, and a subkey for each output type
    """
    criteria_dict = {'rmse': dict(),
                        'mse': dict(),
                        'nse': dict(),
                        'r2': dict(),
                        'pearson_r': dict(),
                        'mean_error': dict(),
                        'std_error': dict()}

    if std_metrics:
        criteria_dict['norm_error'] = dict()
        criteria_dict['P95'] = dict()
        criteria_dict['DS'] = dict()

    for i, key in enumerate(sim_y):
        sm_out = sim_y[key]
        try:
            sm_std = sim_std[key]
        except: 
            sim_std = None
            std_metrics = False

        t_mat = np.asarray(true_y[key])
        p_mat = np.asarray(sm_out)
        n_obs = p_mat.shape[1]

        res = t_mat - p_mat
        mean_t = np.mean(t_mat, axis=0)
        mean_p = np.mean(p_mat, axis=0)
        var_t = np.var(t_mat, axis=0)
        std_t = np.std(t_mat, axis=0)
        std_p = np.std(p_mat, axis=0)

        # RMSE & MSE
        criteria_dict['rmse'][key] = np.sqrt(np.mean(res ** 2, axis=0))
        criteria_dict['mse'][key] = np.mean(res ** 2, axis=0)

        # R2 & NSE: only defined where true data has meaningful variance (var_true >= 1e-12)
        valid_var = var_t >= 1e-12
        ss_res = np.sum(res ** 2, axis=0)
        ss_tot = np.sum((t_mat - mean_t) ** 2, axis=0)
        
        r2_arr = np.full(n_obs, np.nan)
        r2_arr[valid_var] = 1.0 - (ss_res[valid_var] / ss_tot[valid_var])
        criteria_dict['r2'][key] = r2_arr
        criteria_dict['nse'][key] = r2_arr.copy()

        # Mean error: only defined where true mean is non-zero (mean_true >= 1e-12)
        valid_mean = np.abs(mean_t) >= 1e-12
        mean_err_arr = np.full(n_obs, np.nan)
        mean_err_arr[valid_mean] = np.abs(mean_t[valid_mean] - mean_p[valid_mean]) / np.abs(mean_t[valid_mean])
        criteria_dict['mean_error'][key] = mean_err_arr

        # Std error: only defined where true std is non-zero (std_true >= 1e-12)
        std_err_arr = np.full(n_obs, np.nan)
        std_err_arr[valid_var] = np.abs(std_t[valid_var] - std_p[valid_var]) / std_t[valid_var]
        criteria_dict['std_error'][key] = std_err_arr

        # Pearson correlation coefficient R
        pearson_arr = np.full(n_obs, np.nan)
        var_p = np.var(p_mat, axis=0)
        for j in range(n_obs):
            if valid_var[j] and var_p[j] >= 1e-12:
                pearson_arr[j] = np.corrcoef(t_mat[:, j], p_mat[:, j])[0, 1]
        criteria_dict['pearson_r'][key] = pearson_arr

        # Norm error, P95, Dawid Score
        if std_metrics and sim_std is not None:
            upper_ci = p_mat + (1.96 * sm_std)
            lower_ci = p_mat - (1.96 * sm_std)

            # Normalized error (only where sim_std >= 1e-12)
            valid_std = sm_std >= 1e-12
            ind_val = np.full_like(p_mat, np.nan)
            np.divide(p_mat - t_mat, sm_std, out=ind_val, where=valid_std)
            criteria_dict['norm_error'][key] = np.nanmean(ind_val ** 2, axis=0)

            # P95
            p95 = np.where((t_mat <= upper_ci) & (t_mat >= lower_ci), 1.0, 0.0)
            criteria_dict['P95'][key] = np.mean(p95, axis=0)

            # Dawid Score
            ds_val = np.full_like(p_mat, np.nan)
            with np.errstate(divide='ignore', invalid='ignore'):
                ds_val = ((p_mat - t_mat) ** 2) / (sm_std ** 2) + np.log(sm_std ** 2)
            criteria_dict['DS'][key] = np.nanmean(ds_val, axis=0)

    return criteria_dict
