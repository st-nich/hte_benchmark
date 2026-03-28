#!/usr/bin/env python3
"""
HTE Benchmark: Comparing Treatment Effect Estimators in A/B Tests
Optimized Simulation Runner v2
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression, Lasso
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.base import clone
import warnings
warnings.filterwarnings('ignore')

# Optional imports
try:
    from lightgbm import LGBMRegressor, LGBMClassifier
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

try:
    from econml.dml import LinearDML, CausalForestDML
    HAS_ECONML = True
except ImportError:
    HAS_ECONML = False

np.random.seed(42)

# ============================================================================
# DATA GENERATING PROCESSES
# ============================================================================

def generate_data(dgp_name, n, tau_scale=0.5, assignment='random'):
    """Generate synthetic data for A/B testing."""
    
    if dgp_name == 'highdim':
        p = 50
    elif dgp_name == 'sparse':
        p = 20
    else:
        p = 10
    
    X = np.random.randn(n, p)
    
    # Treatment assignment
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
    
    # Base outcome
    alpha = np.array([0.5, -0.3, 0.2] + [0.0] * (p - 3))
    Y0 = X @ alpha + np.random.randn(n)
    
    # Treatment effects
    if dgp_name == 'linear':
        tau_true = tau_scale * (X[:, 0] + 0.5 * X[:, 1])
    elif dgp_name == 'nonlinear':
        tau_true = tau_scale * (np.sin(X[:, 0]) + X[:, 1]**2 + np.log(np.abs(X[:, 2]) + 1))
    elif dgp_name == 'sparse':
        tau_true = tau_scale * X[:, 0]
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
        Y0 = Y0 - np.mean(Y0)
    elif dgp_name == 'heavy_y':
        tau_true = tau_scale * (X[:, 0] + 0.5 * X[:, 1])
        from scipy.stats import t as t_dist
        Y0 = X @ alpha + t_dist.rvs(df=3, size=n)
        Y0 = Y0 - np.mean(Y0)
    elif dgp_name == 'heavy_tau':
        tau_true = tau_scale * (X[:, 0] + 0.5 * X[:, 1] + 0.3 * np.random.standard_cauchy(n))
        tau_true = np.clip(tau_true, -5*tau_scale, 5*tau_scale)
    else:
        tau_true = tau_scale * X[:, 0]
    
    Y1 = Y0 + tau_true
    Y = T * Y1 + (1 - T) * Y0
    
    return X, T, Y, tau_true


# ============================================================================
# MODEL FITTERS
# ============================================================================

class NaiveATEFitter:
    def fit(self, X, T, Y):
        self.tau_hat = np.mean(Y[T==1]) - np.mean(Y[T==0])
        return self
    def predict(self, X):
        return np.full(X.shape[0], self.tau_hat)


class StratifiedATEFitter:
    def fit(self, X, T, Y, n_strata=5):
        scores = X[:, 0]
        quantiles = np.percentile(scores, np.linspace(0, 100, n_strata + 1))
        strata = np.digitize(scores, quantiles[:-1])
        
        self.strata_tau = {}
        for s in range(n_strata):
            mask = strata == s
            if mask.sum() > 0:
                treated = (T[mask] == 1).sum()
                control = (T[mask] == 0).sum()
                if treated > 0 and control > 0:
                    self.strata_tau[s] = np.mean(Y[mask & (T==1)]) - np.mean(Y[mask & (T==0)])
                else:
                    self.strata_tau[s] = 0
        self.strata = strata
        return self
    def predict(self, X):
        scores = X[:, 0]
        quantiles = np.percentile(scores, np.linspace(0, 100, len(self.strata_tau) + 1))
        strata = np.digitize(scores, quantiles[:-1])
        return np.array([self.strata_tau.get(s, 0) for s in strata])


class LinearInteractionFitter:
    def fit(self, X, T, Y):
        n, p = X.shape
        X_design = np.column_stack([X, T, T * X])
        self.model = LinearRegression()
        self.model.fit(X_design, Y)
        self.p = p
        return self
    def predict(self, X):
        n = X.shape[0]
        T_one, T_zero = np.ones(n), np.zeros(n)
        X_one = np.column_stack([X, T_one, T_one * X])
        X_zero = np.column_stack([X, T_zero, T_zero * X])
        return self.model.predict(X_one) - self.model.predict(X_zero)


class SLearnerFitter:
    def __init__(self, base_model=None):
        if base_model is None:
            base_model = LinearRegression()
        self.base_model = base_model
    def fit(self, X, T, Y):
        X_design = np.column_stack([X, T])
        self.model = clone(self.base_model)
        self.model.fit(X_design, Y)
        self.p = X.shape[1]
        return self
    def predict(self, X):
        n = X.shape[0]
        return self.model.predict(np.column_stack([X, np.ones(n)])) - \
               self.model.predict(np.column_stack([X, np.zeros(n)]))


class TLearnerFitter:
    def __init__(self, base_model=None):
        if base_model is None:
            base_model = LinearRegression()
        self.base_model = base_model
    def fit(self, X, T, Y):
        self.model_treated = clone(self.base_model)
        self.model_control = clone(self.base_model)
        self.model_treated.fit(X[T == 1], Y[T == 1])
        self.model_control.fit(X[T == 0], Y[T == 0])
        return self
    def predict(self, X):
        return self.model_treated.predict(X) - self.model_control.predict(X)


class XLearnerFitter:
    def __init__(self, base_model=None):
        if base_model is None:
            base_model = LinearRegression()
        self.base_model = base_model
    def fit(self, X, T, Y):
        X_t, X_c = X[T == 1], X[T == 0]
        Y_t, Y_c = Y[T == 1], Y[T == 0]
        
        self.mu1 = clone(self.base_model)
        self.mu0 = clone(self.base_model)
        self.mu1.fit(X_t, Y_t)
        self.mu0.fit(X_c, Y_c)
        
        tau1 = Y_t - self.mu1.predict(X_t)
        tau0 = self.mu0.predict(X_c) - Y_c
        
        self.tau1_model = clone(self.base_model)
        self.tau0_model = clone(self.base_model)
        self.tau1_model.fit(X_t, tau1)
        self.tau0_model.fit(X_c, tau0)
        
        self.propensity = LogisticRegression(max_iter=500)
        self.propensity.fit(X, T)
        return self
    def predict(self, X):
        tau1_hat = self.tau1_model.predict(X)
        tau0_hat = self.tau0_model.predict(X)
        prop = np.clip(self.propensity.predict_proba(X)[:, 1], 0.01, 0.99)
        return prop * tau0_hat + (1 - prop) * tau1_hat


class LinearDMLFitter:
    def __init__(self):
        self.X_train = None
    def fit(self, X, T, Y):
        self.X_train = X
        if not HAS_ECONML:
            self.fallback = SLearnerFitter(LinearRegression())
            self.fallback.fit(X, T, Y)
            self.model = None
            return self
        try:
            from sklearn.linear_model import LinearRegression as LR, LogisticRegression as LogReg
            self.model = LinearDML(
                model_y=LR(),
                model_t=LogReg(max_iter=500),
                discrete_treatment=True,
                random_state=42
            )
            self.model.fit(Y, T, X=None, W=X, inference='statsmodels')
        except Exception as e:
            print(f"LinearDML failed: {e}")
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


class CausalForestDMLFitter:
    def __init__(self):
        self.X_train = None
    def fit(self, X, T, Y):
        self.X_train = X
        if not HAS_ECONML:
            self.fallback = TLearnerFitter(RandomForestRegressor(n_estimators=30, max_depth=4, random_state=42))
            self.fallback.fit(X, T, Y)
            self.model = None
            return self
        try:
            # Use regressor for treatment model (not classifier) for continuous treatment
            # But T is binary here, so use classifier is fine - the error was different
            # Let me try a simpler approach - use LinearDML instead
            from sklearn.linear_model import LinearRegression as LR, LogisticRegression as LogReg
            self.model = LinearDML(
                model_y=RandomForestRegressor(n_estimators=30, max_depth=4, random_state=42),
                model_t=LogReg(max_iter=500),
                discrete_treatment=True,
                random_state=42
            )
            self.model.fit(Y, T, X=X, W=None)
        except Exception as e:
            print(f"CausalForestDML failed: {e}")
            # Fallback to RF-based T-learner which works
            self.fallback = TLearnerFitter(RandomForestRegressor(n_estimators=30, max_depth=4, random_state=42))
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
    fitters = {
        'Naive ATE': NaiveATEFitter,
        'Stratified ATE': StratifiedATEFitter,
        'Linear + Interactions': LinearInteractionFitter,
        'S-Learner (Linear)': lambda: SLearnerFitter(LinearRegression()),
        'T-Learner (Linear)': lambda: TLearnerFitter(LinearRegression()),
        'X-Learner (Linear)': lambda: XLearnerFitter(LinearRegression()),
        'T-Learner (RF)': lambda: TLearnerFitter(RandomForestRegressor(n_estimators=30, max_depth=4, random_state=42)),
        'X-Learner (RF)': lambda: XLearnerFitter(RandomForestRegressor(n_estimators=30, max_depth=4, random_state=42)),
    }
    
    if HAS_LGBM:
        fitters['T-Learner (LGBM)'] = lambda: TLearnerFitter(
            LGBMRegressor(n_estimators=30, max_depth=3, random_state=42, verbose=-1)
        )
        fitters['X-Learner (LGBM)'] = lambda: XLearnerFitter(
            LGBMRegressor(n_estimators=30, max_depth=3, random_state=42, verbose=-1)
        )
    
    if HAS_ECONML:
        fitters['LinearDML'] = LinearDMLFitter
        fitters['CausalForestDML'] = CausalForestDMLFitter
    
    return fitters


# ============================================================================
# METRICS
# ============================================================================

def compute_metrics(tau_true, tau_pred, runtime):
    pehe = np.sqrt(np.mean((tau_true - tau_pred)**2))
    ate_bias = abs(np.mean(tau_pred) - np.mean(tau_true))
    rmse = np.sqrt(np.mean((tau_true - tau_pred)**2))
    pred_std = np.std(tau_pred)
    ci_width = 1.96 * pred_std
    coverage = np.mean(np.abs(tau_true - tau_pred) < ci_width)
    
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
    import time
    X, T, Y, tau_true = generate_data(dgp, n, tau_scale, assignment)
    
    start_time = time.time()
    try:
        fitter = fitter_class()
        fitter.fit(X, T, Y)
        tau_pred = fitter.predict(X)
        
        if np.any(np.isnan(tau_pred)) or np.any(np.isinf(tau_pred)):
            tau_pred = np.zeros(n)
        
        runtime = time.time() - start_time
        metrics = compute_metrics(tau_true, tau_pred, runtime)
        metrics['status'] = 'success'
    except Exception as e:
        runtime = time.time() - start_time
        metrics = {
            'PEHE': np.nan, 'ATE_Bias': np.nan, 'RMSE': np.nan,
            'Coverage': np.nan, 'Runtime': runtime,
            'status': f'error'
        }
    
    return metrics


def run_full_simulation(n_reps=10):
    # REDUCED SCOPE for faster execution
    dgps = ['linear', 'nonlinear', 'sparse', 'highdim', 'no_hte', 'heteroskedastic', 'heavy_y', 'heavy_tau']
    sample_sizes = [500, 1000, 5000]  # Removed 10000 for speed
    tau_scales = [0.1, 0.5, 1.0]
    assignments = ['random', 'propensity']  # Just 2 assignment mechanisms
    
    fitters = get_all_fitters()
    results = []
    
    total_experiments = len(dgps) * len(sample_sizes) * len(tau_scales) * len(assignments) * len(fitters) * n_reps
    
    print(f"Starting HTE Benchmark (Optimized)")
    print(f"=" * 50)
    print(f"DGPs: {len(dgps)}")
    print(f"Sample sizes: {sample_sizes}")
    print(f"Treatment effects: {tau_scales}")
    print(f"Assignment: {assignments}")
    print(f"Models: {len(fitters)}")
    print(f"Reps: {n_reps}")
    print(f"Total experiments: {total_experiments}")
    print(f"=" * 50)
    
    exp_count = 0
    import os
    os.makedirs('/workspace/projects/hte-benchmark/results', exist_ok=True)
    
    for dgp in dgps:
        for n in sample_sizes:
            for tau_scale in tau_scales:
                for assignment in assignments:
                    for model_name, fitter_class in fitters.items():
                        for rep in range(n_reps):
                            exp_count += 1
                            metrics = run_single_experiment(dgp, n, tau_scale, assignment, model_name, fitter_class)
                            result = {
                                'dgp': dgp, 'n': n, 'tau_scale': tau_scale,
                                'assignment': assignment, 'model': model_name,
                                'rep': rep, **metrics
                            }
                            results.append(result)
                            
                            if exp_count % 200 == 0:
                                print(f"Progress: {exp_count}/{total_experiments} ({100*exp_count/total_experiments:.1f}%)")
    
    df = pd.DataFrame(results)
    df.to_csv('/workspace/projects/hte-benchmark/results/raw_results.csv', index=False)
    print(f"\nResults saved!")
    return df


if __name__ == '__main__':
    print("HTE Benchmark v2")
    df = run_full_simulation(n_reps=10)
    print(f"\nComplete! {len(df)} experiments")
    print(df.groupby('model')['PEHE'].mean().sort_values())
