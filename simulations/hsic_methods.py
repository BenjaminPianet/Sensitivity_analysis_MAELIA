import numpy as np
from itertools import combinations
from numba import njit
from scipy.optimize import minimize

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

def compute_theta_kta(D_features, Y, theta_mask=None):
    """
    Optimizes theta length scales by maximizing the Kernel Target Alignment (KTA)
    between the additive input RKHS kernel and the output kernel.
    """
    n_samples = D_features.shape[0]
    n_features = D_features.shape[2]
    
    print("    [KTA] Computing Target Kernel K_Y...")
    Y_dist = np.abs(Y - Y.T)
    med_Y = np.median(Y_dist[np.triu_indices(n_samples, k=1)])
    if med_Y == 0.0: med_Y = 1.0
    K_Y = np.exp(- (Y_dist**2) / (2 * med_Y**2))
    
    # Fast centering for K_Y O(N^2)
    K_Y_row = K_Y.mean(axis=1, keepdims=True)
    K_Y_col = K_Y.mean(axis=0, keepdims=True)
    Kc_Y = K_Y - K_Y_row - K_Y_col + K_Y.mean()
    
    norm_Kc_Y = np.linalg.norm(Kc_Y, 'fro')
    
    D2_features = D_features**2
    
    def kta_objective(theta):
        Kc_X_sum = np.zeros((n_samples, n_samples))
        for i in range(n_features):
            if theta[i] > 1e-6:
                K_i = np.exp(- theta[i] * D2_features[:, :, i])
                # Fast centering O(N^2) instead of O(N^3)
                K_row = K_i.mean(axis=1, keepdims=True)
                K_col = K_i.mean(axis=0, keepdims=True)
                Kc_i = K_i - K_row - K_col + K_i.mean()
                
                Kc_X_sum += Kc_i
                
        inner_prod = np.sum(Kc_X_sum * Kc_Y)
        norm_Kc_X = np.linalg.norm(Kc_X_sum, 'fro')
        
        if norm_Kc_X < 1e-10: return 0.0
        return -(inner_prod / (norm_Kc_X * norm_Kc_Y))

    # Init with smoothed median heuristic or RF mask
    if theta_mask is not None:
        theta_init = np.copy(theta_mask)
        # Lock theta to 0 if RF eliminated it
        bounds = [(0, 0) if theta_mask[i] <= 0.001 else (0.001, 100.0) for i in range(n_features)]
    else:
        theta_init = compute_theta_median_heuristic(D_features, alpha_smoothing=2.0)
        bounds = [(0, 100.0) for _ in range(n_features)]
    
    print("    [KTA] Optimizing length scales (L-BFGS-B)...")
    res = minimize(kta_objective, theta_init, method='L-BFGS-B', bounds=bounds, options={'maxiter': 50, 'disp': False})
    
    print(f"    [KTA] Optimization finished. Alignment Score: {-res.fun:.4f}")
    theta_opt = res.x
    theta_opt[theta_opt < 1e-3] = 0.0  # Zero-out useless features
    return theta_opt

def hsic_anova_hierarchical(X, Y, x_is_acting, num_is_decreed, is_categorical, theta_scales=None, var_names=None, max_order=3, use_smt_theta=True, use_kta=False):
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
    
    # 2. Kernel scaling (theta)
    if use_kta and theta_scales is not None:
        print("    [Coupling RF & KTA] Using RF theta as a mask for KTA optimization...")
        theta = compute_theta_kta(D_X_features, Y, theta_mask=theta_scales)
    elif use_kta:
        print("    Using Kernel Target Alignment (KTA) optimization for theta scales.")
        theta = compute_theta_kta(D_X_features, Y)
    elif theta_scales is not None and use_smt_theta:
        print("    Using provided RF theta scales.")
        theta = theta_scales
    else:
        print("    Using Smoothed Median Heuristic for theta scales.")
        theta = compute_theta_median_heuristic(D_X_features)
        
    base_centered_kernels = []
    for i in range(n_features):
        K_i = np.exp(- theta[i] * (D_X_features[:, :, i] ** 2))
        # Da Veiga (2015): For true ANOVA decomposition in RKHS, base kernels must be centered
        Kc_i = center_kernel(K_i)
        base_centered_kernels.append(Kc_i)
        
    # Global HSIC is theoretically computed on Prod(1 + Kc_i) - 1
    K_global = np.ones((n_samples, n_samples))
    for Kc_i in base_centered_kernels:
        K_global *= (1 + Kc_i)
    K_global -= 1
    
    # O(N^2) trace equivalence: np.trace(A @ B) == np.sum(A * B.T)
    total_trace = np.sum(K_global * Lc) / ((n_samples - 1) ** 2)
    
    # 3. Compute ANOVA terms iteratively
    results = []
    
    # 5. Compute HSIC for each combination
    print("    Evaluating combinations...")
    for order in range(1, max_order + 1):
        for combo in combinations(range(n_features), order):
            K_A = np.ones((n_samples, n_samples))
            
            # Compute joint activation probability for this combination
            joint_acting = np.ones(n_samples, dtype=bool)
            
            for idx in combo:
                K_A *= base_centered_kernels[idx]
                if num_is_decreed[idx]:
                    joint_acting &= x_is_acting[:, idx]
            
            p_A = np.mean(joint_acting)
            
            # Fast trace estimator (sum of Hadamard product)
            trace_val = np.sum(K_A * Lc) / ((n_samples - 1) ** 2)
            
            # Adjusted trace (Intrinsic sensitivity)
            # We divide by p_A^2 because the cross-covariance trace scales quadratically with sparsity
            adj_trace_val = trace_val / (p_A**2 + 1e-8)
            
            if trace_val > 0.0001 * total_trace:
                name = " & ".join([var_names[i] for i in combo]) if var_names else f"Combo {combo}"
                results.append({
                    'order': order,
                    'combo': combo,
                    'name': name,
                    'trace': trace_val,
                    'adj_trace': adj_trace_val,
                    'p_A': p_A
                })
                
    # Calculate sum of adjusted traces for normalization
    sum_adj_trace = sum(r['adj_trace'] for r in results)
    
    # Sort by INTRINSIC variance
    results.sort(key=lambda x: x['adj_trace'], reverse=True)
    
    # Print the table
    print("\n" + "="*95)
    print("                      MAELIA SENSITIVITY ANALYSIS (HSIC-ANOVA)")
    print("===============================================================================================")
    print(f"Global HSIC (Total Dependency): {total_trace:e}\n")
    print("Order  | Global Var % | Intrinsic Var % | Act. Freq | Interacting Variables")
    print("-" * 95)
    
    explained_global = 0.0
    explained_intrinsic = 0.0
    filtered_results = []
    
    for r in results:
        share = r['trace'] / total_trace
        adj_share = r['adj_trace'] / sum_adj_trace if sum_adj_trace > 0 else 0
        
        explained_global += share
        explained_intrinsic += adj_share
        filtered_results.append(r)
        
        print(f"  {r['order']}    |    {share*100:6.2f}%    |     {adj_share*100:6.2f}%    |   {r['p_A']*100:5.1f}%   | {r['name']}")
        
        # Stop printing if we reached 95% of the INTRINSIC variance
        if explained_intrinsic > 0.95:
            break
            
    print("-" * 95)
    print(f"Total intrinsic variance explained by these terms: {explained_intrinsic*100:.2f}%")
    print(f"Total global variance explained by these terms: {explained_global*100:.2f}%")
    print("===============================================================================================")
    
    return filtered_results, total_trace
