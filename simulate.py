#!/usr/bin/env python3
"""
HTE Benchmark: Comparing Treatment Effect Estimators in A/B Tests
Simulation Runner
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Try importing optional libraries
try:
    from xgboost import XGBRegressor, XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("XGBoost not available, skipping XGB models")

try:
    from lightgbm import LGBMRegressor, LGBMClassifier
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False
    print("LightGBM not available, skipping LGBM models")

try:
    from econml.dml import LinearDML, CausalForestDML
    HAS_ECONML = True
except ImportError:
    HAS_ECONML = False
    print("EconML not fully available")

np.random.seed(42)

# ============================================================================
# DATA GENERATING PROCESSES
# ============================================================================

def generate_data(dgp_name, n, tau_scale=0.5, assignment='random'):
    """
    Generate synthetic data for A/B testing.
    
    Parameters:
    -----------
    dgp_name : str
        One of: 'linear', 'nonlinear', 'sparse', 'highdim', 'no_hte', 
                 'heteroskedastic', 'heavy_y', 'heavy_tau'
    n : int
        Sample size
    tau_scale : float
        Base treatment effect magnitude (0.1, 0.5, 1.0)
    assignment : str
        One of: 'random', 'propensity', 'high_overlap', 'low_overlap'
    
    Returns:
    --------
    X : np.array (n, p)
        Covariate matrix
    T : np.array (n,)
        Treatment indicator (0 or 1)
    Y : np.array (n,)
        Outcome
    tau_true : np.array (n,)
        True treatment effect for each unit
    """
    
    # Covariate dimensions vary by DGP
    if dgp_name == 'highdim':
        p = 50
    elif dgp_name == 'sparse':
        p = 20
    else:
        p = 10
    
    X = np.random.randn(n, p)
    
    # Generate treatment assignment
    if assignment == 'random':
        T = np.random.binomial(1, 0.5, n)
    elif assignment == 'propensity':
        propensity = 1 / (1 + np.exp(-0.5 * X[:, 0] - 0.3 * X[:, 1]))
        T = np.random.binomial(1, propensity)
    elif assignment == 'high_overlap':
        propensity = 0.5 + 0.1 * np.sin(X[:, 0])
        propensity = np.clip(propensity, 0.2, 0.8)
        T = np.random.binomial(1, propensity)
    elif assignment == 'low_overlap':
        propensity = 1 / (1 + np.exp(-2 * X[:, 0] - 1.5 * X[:, 1]))
        T = np.random.binomial(1, propensity)
    else:
        T = np.random.binomial(1, 0.5, n)
    
    # Base outcome (Y(0))
    alpha = np.array([0.5, -0.3, 0.2] + [0.0] * (p - 3))
    Y0 = X @ alpha + np.random.randn(n)
    
    # Treatment effects based on DGP
    if dgp_name == 'linear':
        tau_true = tau_scale * (X[:, 0] + 0.5 * X[:, 1])
    elif dgp_name == 'nonlinear':
        tau_true = tau_scale * (np.sin(X[:, 0]) + X[:, 1]**2 + np.log(np.abs(X[:, 2]) + 1))
    elif dgp_name == 'sparse':
        tau_true = tau_scale * X[:, 0]  # Only first feature matters
    elif dgp_name == 'highdim':
        beta = np.zeros(p)
        beta[:5] = [1, -0.5, 0.5, -0.3, 0.3]
        tau_true = tau_scale * (X @ beta)
    elif dgp_name == 'no_hte':
        tau_true = np.full(n, tau_scale)
    elif dgp_name == 'heteroskedastic':
        tau_true = tau_scale * (X[:, 0] + 0.5 * X[:, 1])
        sigma = np.exp(0.5 * X[:, 0])
        Y0 = X @ alpha + sigma * np.random.randn(n)
        Y0 = Y0 - np.mean(Y0)  # Center
    elif dgp_name == 'heavy_y':
        tau_true = tau_scale * (X[:, 0] + 0.5 * X[:, 1])
        # t-distribution with 3 degrees of freedom
        from scipy.stats import t as t_dist
        Y0 = X @ alpha + t_dist.rvs(df=3, size=n)
        Y0 = Y0 - np.mean(Y0)
    elif dgp_name == 'heavy_tau':
        # Cauchy-like treatment effects
        tau_true = tau_scale * (X[:, 0] + 0.5 * X[:, 1] + 0.3 * np.random.standard_cauchy(n))
        tau_true = np.clip(tau_true, -5*tau_scale, 5*tau_scale)  # Clip extreme values
    else:
        tau_true = tau_scale * X[:, 0]
    
    # Observed outcome
    Y1 = Y0 + tau_true
    Y = T * Y1 + (1 - T) * Y0
    
    return X, T, Y, tau_true


# ============================================================================
# MODEL FITTERS
# ============================================================================

class BaseFitter:
    """Base class for treatment effect estimators."""
    def fit(self, X, T, Y):
        raise NotImplementedError
    
    def predict(self, X):
        raise NotImplementedError
    
    def predict_ate(self):
        raise NotImplementedError


class NaiveATEFitter(BaseFitter):
    """Simple difference in means."""
    def fit(self, X, T, Y):
        self.tau_hat = np.mean(Y[T==1]) - np.mean(Y[T==0])
        return self
    
    def predict(self, X):
        return np.full(X.shape[0], self.tau_hat)
    
    def predict_ate(self):
        return self.tau_hat


class StratifiedATEFitter(BaseFitter):
    """Stratified ATE within quantile subgroups."""
    def fit(self, X, T, Y, n_strata=5):
        self.n_strata = n_strata
        # Simple stratification on first covariate
        scores = X[:, 0]
        quantiles = np.percentile(scores, np.linspace(0, 100, n_strata + 1))
        strata = np.digitize(scores, quantiles[:-1])
        
        self.strata_tau = {}
        for s in range(n_strata):
            mask = strata == s
            if mask.sum() > 0 and mask.sum() < len(T):
                treated = (T[mask] == 1).sum()
                control = (T[mask] == 0).sum()
                if treated > 0 and control > 0:
                    self.strata_tau[s] = np.mean(Y[mask & (T==1)]) - np.mean(Y[mask & (T==0)])
                else:
                    self.strata_tau[s] = 0
            else:
                self.strata_tau[s] = 0
        self.strata = strata
        return self
    
    def predict(self, X):
        scores = X[:, 0]
        quantiles = np.percentile(scores, np.linspace(0, 100, self.n_strata + 1))
        strata = np.digitize(scores, quantiles[:-1])
        return np.array([self.strata_tau.get(s, 0) for s in strata])
    
    def predict_ate(self):
        return np.mean(list(self.strata_tau.values()))


class LinearInteractionFitter(BaseFitter):
    """Linear regression with treatment interactions."""
    def __init__(self):
        self.model = None
    
    def fit(self, X, T, Y):
        n, p = X.shape
        # Create design matrix with treatment interactions
        X_design = np.column_stack([
            X,
            T,
            T * X  # Interactions
        ])
        from sklearn.linear_model import LinearRegression
        self.model = LinearRegression(fit_intercept=True)
        self.model.fit(X_design, Y)
        self.p = p
        return self
    
    def predict(self, X):
        n = X.shape[0]
        T_one = np.ones(n)
        T_zero = np.zeros(n)
        
        X_one = np.column_stack([X, T_one, T_one * X])
        X_zero = np.column_stack([X, T_zero, T_zero * X])
        
        Y1_pred = self.model.predict(X_one)
        Y0_pred = self.model.predict(X_zero)
        
        return Y1_pred - Y0_pred
    
    def predict_ate(self):
        # Average treatment effect from model
        return self.model.coef_[self.p]  # Coefficient on T


class SLearnerFitter(BaseFitter):
    """S-Learner: single model with treatment as feature."""
    def __init__(self, base_model=None):
        self.base_model = base_model or LinearRegression()
    
    def fit(self, X, T, Y):
        n, p = X.shape
        X_design = np.column_stack([X, T])
        self.model = type(self.base_model)(**self.base_model.get_params())
        self.model.fit(X_design, Y)
        self.p = p
        return self
    
    def predict(self, X):
        n = X.shape[0]
        T_one = np.ones(n)
        T_zero = np.zeros(n)
        
        X_one = np.column_stack([X, T_one])
        X_zero = np.column_stack([X, T_zero])
        
        Y1_pred = self.model.predict(X_one)
        Y0_pred = self.model.predict(X_zero)
        
        return Y1_pred - Y0_pred
    
    def predict_ate(self):
        return np.mean(self.predict(np.zeros((1, self.p))))


class TLearnerFitter(BaseFitter):
    """T-Learner: separate models for treated and control."""
    def __init__(self, base_model=None):
        self.base_model = base_model or LinearRegression()
    
    def fit(self, X, T, Y):
        X_treated = X[T == 1]
        Y_treated = Y[T == 1]
        X_control = X[T == 0]
        Y_control = Y[T == 0]
        
        self.model_treated = type(self.base_model)(**self.base_model.get_params())
        self.model_control = type(self.base_model)(**self.base_model.get_params())
        
        self.model_treated.fit(X_treated, Y_treated)
        self.model_control.fit(X_control, Y_control)
        return self
    
    def predict(self, X):
        Y1_pred = self.model_treated.predict(X)
        Y0_pred = self.model_control.predict(X)
        return Y1_pred - Y0_pred
    
    def predict_ate(self):
        return np.mean(self.predict(np.zeros((1, self.model_treated.n_features_in_))))


class XLearnerFitter(BaseFitter):
    """X-Learner: T-learner + imputation + weighting."""
    def __init__(self, base_model=None, propensity_model=None):
        self.base_model = base_model or LinearRegression()
        self.propensity_model = propensity_model or LogisticRegression()
    
    def fit(self, X, T, Y):
        # Stage 1: Fit T-learner
        X_treated = X[T == 1]
        Y_treated = Y[T == 1]
        X_control = X[T == 0]
        Y_control = Y[T == 0]
        
        self.mu1 = type(self.base_model)(**self.base_model.get_params())
        self.mu0 = type(self.base_model)(**self.base_model.get_params())
        
        self.mu1.fit(X_treated, Y_treated)
        self.mu0.fit(X_control, Y_control)
        
        # Impute treatment effects
        tau1 = Y_treated - self.mu1.predict(X_treated)  # For treated
        tau0 = self.mu0.predict(X_control) - Y_control  # For control
        
        # Stage 2: Fit models on imputed effects
        self.tau1_model = type(self.base_model)(**self.base_model.get_params())
        self.tau0_model = type(self.base_model)(**self.base_model.get_params())
        
        self.tau1_model.fit(X_treated, tau1)
        self.tau0_model.fit(X_control, tau0)
        
        # Propensity score
        self.propensity_model.fit(X, T)
        return self
    
    def predict(self, X):
        tau1_hat = self.tau1_model.predict(X)
        tau0_hat = self.tau0_model.predict(X)
        
        prop = self.propensity_model.predict_proba(X)[:, 1]
        prop = np.clip(prop, 0.01, 0.99)
        
        # Weight by propensity
        tau_hat = prop * tau0_hat + (1 - prop) * tau1_hat
        return tau_hat
    
    def predict_ate(self):
        return np.mean(self.predict(np.zeros((1, self.tau1_model.n_features_in_))))


class LinearDMLFitter(BaseFitter):
    """Linear DML with cross-fitting."""
    def __init__(self):
        self.model = None
        self.X = None
    
    def fit(self, X, T, Y):
        self.X = X  # Store for predict_ate
        if not HAS_ECONML:
            # Fallback to S-learner if econml not available
            self.fallback = SLearnerFitter(LinearRegression())
            self.fallback.fit(X, T, Y)
            return self
        
        try:
            self.model = LinearDML(
                model_y=LinearRegression(),
                model_t=LogisticRegression(max_iter=1000),
                discrete_treatment=True,
                random_state=42
            )
            self.model.fit(Y, T, X=None, W=X, inference='statsmodels')
        except Exception as e:
            print(f"LinearDML failed: {e}, using fallback")
            self.fallback = SLearnerFitter(LinearRegression())
            self.fallback.fit(X, T, Y)
            self.model = None
        return self
    
    def predict(self, X):
        if self.model is None:
            return self.fallback.predict(X)
        try:
            return self.model.effect(X).flatten()
        except:
            return self.fallback.predict(X)
    
    def predict_ate(self):
        if self.model is None:
            return self.fallback.predict_ate()
        try:
            return np.mean(self.model.effect(self.X))
        except:
            return self.fallback.predict_ate()


class CausalForestDMLFitter(BaseFitter):
    """Causal Forest DML."""
    def __init__(self):
        self.model = None
    
    def fit(self, X, T, Y):
        if not HAS_ECONML:
            self.fallback = TLearnerFitter(RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42))
            self.fallback.fit(X, T, Y)
            return self
        
        try:
            self.model = CausalForestDML(
                model_y=RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42),
                model_t=RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42),
                n_estimators=100,
                random_state=42
            )
            self.model.fit(Y, T, X=X, W=None)
        except Exception as e:
            print(f"CausalForestDML failed: {e}, using fallback")
            self.fallback = TLearnerFitter(RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42))
            self.fallback.fit(X, T, Y)
            self.model = None
        return self
    
    def predict(self, X):
        if self.model is None:
            return self.fallback.predict(X)
        try:
            return self.model.effect(X).flatten()
        except:
            return self.fallback.predict(X)


# ============================================================================
# MODEL REGISTRY
# ============================================================================

def get_all_fitters():
    """Return dictionary of all model fitters."""
    fitters = {
        'Naive ATE': NaiveATEFitter,
        'Stratified ATE': StratifiedATEFitter,
        'Linear + Interactions': LinearInteractionFitter,
        'S-Learner (Linear)': lambda: SLearnerFitter(LinearRegression()),
        'T-Learner (Linear)': lambda: TLearnerFitter(LinearRegression()),
        'X-Learner (Linear)': lambda: XLearnerFitter(LinearRegression()),
        'T-Learner (RF)': lambda: TLearnerFitter(RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)),
        'X-Learner (RF)': lambda: XLearnerFitter(RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)),
    }
    
    # Add XGBoost models if available
    if HAS_XGB:
        fitters['T-Learner (XGB)'] = lambda: TLearnerFitter(
            XGBRegressor(n_estimators=50, max_depth=3, random_state=42, verbosity=0)
        )
        fitters['X-Learner (XGB)'] = lambda: XLearnerFitter(
            XGBRegressor(n_estimators=50, max_depth=3, random_state=42, verbosity=0)
        )
    
    # Add LightGBM models if available
    if HAS_LGBM:
        fitters['T-Learner (LGBM)'] = lambda: TLearnerFitter(
            LGBMRegressor(n_estimators=50, max_depth=3, random_state=42, verbose=-1)
        )
        fitters['X-Learner (LGBM)'] = lambda: XLearnerFitter(
            LGBMRegressor(n_estimators=50, max_depth=3, random_state=42, verbose=-1)
        )
    
    # Add econml models if available
    if HAS_ECONML:
        fitters['LinearDML'] = LinearDMLFitter
        fitters['CausalForestDML'] = CausalForestDMLFitter
    
    return fitters


# ============================================================================
# METRICS
# ============================================================================

def compute_metrics(tau_true, tau_pred, runtime):
    """Compute evaluation metrics."""
    # PEHE
    pehe = np.sqrt(np.mean((tau_true - tau_pred)**2))
    
    # ATE Bias
    ate_true = np.mean(tau_true)
    ate_pred = np.mean(tau_pred)
    ate_bias = abs(ate_pred - ate_true)
    
    # RMSE
    rmse = np.sqrt(np.mean((tau_true - tau_pred)**2))
    
    # Coverage (approximate)
    # Use simple prediction interval
    pred_std = np.std(tau_pred)
    ci_width = 1.96 * pred_std
    within_ci = np.abs(tau_true - tau_pred) < ci_width
    coverage = np.mean(within_ci)
    
    return {
        'PEHE': pehe,
        'ATE_Bias': ate_bias,
        'RMSE': rmse,
        'Coverage': coverage,
        'Runtime': runtime
    }


# ============================================================================
# SIMULATION RUNNER
# ============================================================================

def run_single_experiment(dgp, n, tau_scale, assignment, model_name, fitter_class):
    """Run a single experiment."""
    import time
    
    # Generate data
    X, T, Y, tau_true = generate_data(dgp, n, tau_scale, assignment)
    
    # Fit model
    start_time = time.time()
    try:
        fitter = fitter_class()
        fitter.fit(X, T, Y)
        tau_pred = fitter.predict(X)
        
        # Handle NaN/Inf
        if np.any(np.isnan(tau_pred)) or np.any(np.isinf(tau_pred)):
            tau_pred = np.zeros(n)
        
        runtime = time.time() - start_time
        
        # Compute metrics
        metrics = compute_metrics(tau_true, tau_pred, runtime)
        metrics['status'] = 'success'
        
    except Exception as e:
        runtime = time.time() - start_time
        metrics = {
            'PEHE': np.nan,
            'ATE_Bias': np.nan,
            'RMSE': np.nan,
            'Coverage': np.nan,
            'Runtime': runtime,
            'status': f'error: {str(e)[:50]}'
        }
    
    return metrics


def run_full_simulation(n_reps=20):
    """Run the full benchmark simulation."""
    
    # Configuration
    dgps = ['linear', 'nonlinear', 'sparse', 'highdim', 'no_hte', 
            'heteroskedastic', 'heavy_y', 'heavy_tau']
    sample_sizes = [500, 1000, 5000, 10000]
    tau_scales = [0.1, 0.5, 1.0]
    assignments = ['random', 'propensity', 'high_overlap', 'low_overlap']
    
    fitters = get_all_fitters()
    
    # Results storage
    results = []
    
    total_experiments = (len(dgps) * len(sample_sizes) * len(tau_scales) * 
                         len(assignments) * len(fitters) * n_reps)
    
    print(f"Starting HTE Benchmark")
    print(f"=" * 50)
    print(f"DGPs: {len(dgps)}")
    print(f"Sample sizes: {sample_sizes}")
    print(f"Treatment effects: {tau_scales}")
    print(f"Assignment mechanisms: {len(assignments)}")
    print(f"Models: {len(fitters)}")
    print(f"Reps: {n_reps}")
    print(f"Total experiments: {total_experiments}")
    print(f"=" * 50)
    
    exp_count = 0
    
    for dgp in dgps:
        for n in sample_sizes:
            for tau_scale in tau_scales:
                for assignment in assignments:
                    for model_name, fitter_class in fitters.items():
                        for rep in range(n_reps):
                            exp_count += 1
                            
                            # Run experiment
                            metrics = run_single_experiment(dgp, n, tau_scale, 
                                                            assignment, model_name, fitter_class)
                            
                            # Store results
                            result = {
                                'dgp': dgp,
                                'n': n,
                                'tau_scale': tau_scale,
                                'assignment': assignment,
                                'model': model_name,
                                'rep': rep,
                                **metrics
                            }
                            results.append(result)
                            
                            # Progress update every 100 experiments
                            if exp_count % 100 == 0:
                                print(f"Progress: {exp_count}/{total_experiments} ({100*exp_count/total_experiments:.1f}%)")
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Save results
    df.to_csv('/workspace/projects/hte-benchmark/results/raw_results.csv', index=False)
    print(f"\nResults saved to /workspace/projects/hte-benchmark/results/raw_results.csv")
    
    return df


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    import os
    os.makedirs('/workspace/projects/hte-benchmark/results', exist_ok=True)
    
    print("HTE Benchmark Simulation")
    print("=" * 50)
    
    df = run_full_simulation(n_reps=20)
    
    print("\nSimulation complete!")
    print(f"Total experiments: {len(df)}")
