import numpy as np
from itertools import combinations
from numba import njit

# Numba decorator
njit_use = njit

@njit_use()
def apply_the_algebraic_distance_to_the_decreed_variable(
    X_num, Y_num, x_num_is_acting, y_num_is_acting, num_is_decreed, is_categorical
):
    """
    Computes the meta-decreed (Hallé-Hannan) distance for structured spaces.
    """
    nx_samples, n_features = X_num.shape
    ny_samples = Y_num.shape[0]

    D_features = np.zeros((nx_samples, ny_samples, n_features))
        
    for k1 in range(nx_samples):
        for k2 in range(ny_samples):
            for i in range(n_features):
                x_val = X_num[k1, i]
                y_val = Y_num[k2, i]
                
                # Strict compliance: GOWER distance for purely categorical variables
                if is_categorical[i]:
                    dist = 0.0 if x_val == y_val else 1.0
                elif num_is_decreed[i]:
                    # Numerically stable calculation using hypot instead of sqrt(1+x^2)
                    dist = (2 * np.abs(x_val - y_val)) / (np.hypot(1, x_val) * np.hypot(1, y_val))
                    
                    x_act = x_num_is_acting[k1, i]
                    y_act = y_num_is_acting[k2, i]
                    if not x_act and not y_act:
                        dist = 0.0
                    elif x_act != y_act:
                        dist = 1.0
                else:
                    dist = np.abs(x_val - y_val)
                    
                D_features[k1, k2, i] = dist
            
    return D_features

def center_kernel(K):
    """ Centers a kernel matrix empirically in O(N^2) time. """
    return K - np.mean(K, axis=0, keepdims=True) - np.mean(K, axis=1, keepdims=True) + np.mean(K)

def compute_hsic(K, L):
    """ Computes the unbiased HSIC estimator given kernel matrices K and L. """
    n = K.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    Kc = H @ K @ H
    Lc = H @ L @ H
    return np.trace(Kc @ Lc) / ((n - 1) ** 2)

def compute_kernel_from_dist(D_matrix, length_scale=1.0):
    return np.exp(- (D_matrix ** 2) / (2 * length_scale ** 2))

def compute_theta_median_heuristic(D_features, alpha_smoothing=3.0):
    """ 
    Computes theta length scales using the median heuristic on the distance matrix. 
    alpha_smoothing: Multiplier for the median distance. alpha > 1 forces the kernel to be smoother,
                     preventing the variance from shattering into high-order interactions.
    """
    n_features = D_features.shape[2]
    n_samples = D_features.shape[0]
    theta = np.zeros(n_features)
    for i in range(n_features):
        dist_triu = D_features[:, :, i][np.triu_indices(n_samples, k=1)]
        med = np.median(dist_triu)
        if med == 0.0:
            med = np.mean(dist_triu)
            if med == 0.0: med = 1.0
        
        # Apply smoothing factor
        smoothed_length_scale = alpha_smoothing * med
        theta[i] = 1.0 / (2 * smoothed_length_scale**2)
    return theta

def hsic_anova_hierarchical(X, Y, x_is_acting, num_is_decreed, is_categorical, theta_scales=None, var_names=None, max_order=3, use_smt_theta=True):
    """
    Computes HSIC-ANOVA decomposition up to max_order.
    Returns the terms explaining up to 95% of the total variance, or all computed terms.
    """
    n_samples, n_features = X.shape
    
    # 1. Output kernel L and its centered version Lc
    if Y.ndim == 1: Y = Y.reshape(-1, 1)
    D_Y = np.abs(Y[:, None, :] - Y[None, :, :]).sum(axis=-1)
    l_Y = np.std(Y) if np.std(Y) > 0 else 1.0
    L = compute_kernel_from_dist(D_Y, l_Y)
    Lc = center_kernel(L)
    
    # 2. Input feature-wise distances and base kernels
    D_X_features = apply_the_algebraic_distance_to_the_decreed_variable(
        X_num=X, Y_num=X, 
        x_num_is_acting=x_is_acting, y_num_is_acting=x_is_acting, 
        num_is_decreed=num_is_decreed, is_categorical=is_categorical
    )
    
    # 3. Determine theta scales
    if not use_smt_theta:
        print("    [HSIC] Bypassing SMT theta. Computing theta via median heuristic on raw distances...")
        theta_scales = compute_theta_median_heuristic(D_X_features)
    elif theta_scales is None:
        raise ValueError("theta_scales must be provided if use_smt_theta is True.")
        
    base_centered_kernels = []
    for i in range(n_features):
        K_i = np.exp(- theta_scales[i] * (D_X_features[:, :, i] ** 2))
        # Da Veiga (2015): For true ANOVA decomposition in RKHS, base kernels must be centered
        Kc_i = center_kernel(K_i)
        base_centered_kernels.append(Kc_i)
        
    # Global HSIC is theoretically computed on Prod(1 + Kc_i) - 1
    K_global = np.ones((n_samples, n_samples))
    for Kc_i in base_centered_kernels:
        K_global *= (1 + Kc_i)
    K_global -= 1
    
    # O(N^2) trace equivalence: np.trace(A @ B) == np.sum(A * B.T)
    global_hsic = np.sum(K_global * Lc) / ((n_samples - 1) ** 2)
    
    # 3. Compute ANOVA terms iteratively
    results = []
    
    # We want to collect terms until we explain 95% of the variance
    for order in range(1, max_order + 1):
        for subset in combinations(range(n_features), order):
            # Compute subset kernel
            K_subset = np.ones((n_samples, n_samples))
            for idx in subset:
                K_subset *= base_centered_kernels[idx]
                
            val_hsic = np.sum(K_subset * Lc) / ((n_samples - 1) ** 2)
            # Avoid negative HSIC (numerical artifacts)
            val_hsic = max(0.0, val_hsic)
            
            # Name of the interaction
            name = " & ".join([var_names[idx] for idx in subset])
            
            results.append({
                "subset": subset,
                "name": name,
                "order": order,
                "hsic": val_hsic,
                "variance_share": val_hsic / global_hsic
            })

    # Sort results by variance share descending
    results.sort(key=lambda x: x["variance_share"], reverse=True)
    
    # Filter up to 95% cumulative variance
    filtered_results = []
    cumulative_variance = 0.0
    for r in results:
        filtered_results.append(r)
        cumulative_variance += r["variance_share"]
        if cumulative_variance >= 0.95:
            break
            
    return filtered_results, global_hsic
