import numpy as np
from SALib.sample import saltelli
import warnings

def generate_constrained_saltelli(design_space, n_samples, calc_second_order=True):
    """
    Generates a Saltelli sample mapped to an SMT design space and enforces 
    hierarchical/categorical constraints.
    
    WARNING: Applying constraints (rounding, imputing values for inactive variables)
    will likely break the strict structural properties of the Saltelli matrices (A, B, Ci).
    This may introduce bias when calculating Sobol indices.
    
    Parameters:
    -----------
    design_space : smt.design_space.DesignSpace or AdsgDesignSpaceImpl
        The SMT design space defining the variables and constraints.
    n_samples : int
        The base number of samples N. Total generated will be N * (2D + 2) 
        if calc_second_order=True.
    calc_second_order : bool
        Whether to generate cross matrices for second order indices.
        
    Returns:
    --------
    X_corrected : np.ndarray
        The constrained Saltelli sampling matrix.
    is_acting : np.ndarray
        Boolean matrix indicating which variables are active for each point.
    """
    
    # 1. Get continuous bounds of the relaxed design space
    # SMT uses continuous relaxations internally for categorical/integer variables
    bounds = design_space.get_unfolded_num_bounds()
    num_vars = len(bounds)
    
    # 2. Define SALib problem
    problem = {
        'num_vars': num_vars,
        'names': [f'x_{i}' for i in range(num_vars)],
        'bounds': bounds
    }
    
    # 3. Generate raw Saltelli sample
    print(f"Generating raw Saltelli sample with N={n_samples}...")
    # Suppress SALib warnings if necessary
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        X_raw = saltelli.sample(problem, n_samples, calc_second_order=calc_second_order)
    
    print(f"Raw shape: {X_raw.shape[0]} points x {num_vars} variables")
    
    # 4. Enforce constraints using SMT
    print("Applying SMT constraints (rounding, hierarchy imputation)...")
    X_corrected, is_acting = design_space.correct_get_acting(X_raw)
    
    # 5. Analyze the degradation
    # Check for duplicates introduced by rounding/projection
    unique_points = np.unique(X_corrected, axis=0)
    n_duplicates = X_corrected.shape[0] - unique_points.shape[0]
    
    if n_duplicates > 0:
        warnings.warn(
            f"{n_duplicates} duplicate points were created after applying constraints! \n"
            "This confirms the Saltelli matrix structure is degraded. \n"
            "Sobol index calculations using SALib will likely be biased."
        )
    else:
        print("No exact duplicates introduced, but structural dependencies (A, B matrices) might still be altered by hierarchical imputations.")
        
    return X_corrected, is_acting

if __name__ == "__main__":
    print("This module provides the generate_constrained_saltelli function.")
    print("Usage example:")
    print("  from generate_saltelli_smt import generate_constrained_saltelli")
    print("  X_saltelli, is_acting = generate_constrained_saltelli(agri_design_space, n_samples=128)")
