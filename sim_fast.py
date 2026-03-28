#!/usr/bin/env python3
"""
HTE Benchmark: Fast version for quick results
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.base import clone
import warnings
warnings.filterwarnings('ignore')

try:
    from lightgbm import LGBMRegressor
    HAS_LGBM = True
except:
    HAS_LGBM = False

np.random.seed(42)

# ============================================================================
# SIMPLIFIED DGP
# ============================================================================

def generate_data(dgp_name, n, tau_scale=0.5):
    p = 10
    X = np.random.randn(n, p)
    T = np.random.binomial(1, 0.5, n)
    
    alpha = np.array([0.5, -0.3, 0.2] + [0.0] * (p - 3))
    Y0 = X @ alpha + np.random.randn(n)
    
    if dgp_name == 'linear':
        tau_true = tau_scale * (X[:, 0] + 0.5 * X[:, 1])
    elif dgp_name == 'nonlinear':
        tau_true = tau_scale * (np.sin(X[:, 0]) + X[:, 1]**2)
    elif dgp_name == 'sparse':
        tau_true = tau_scale * X[:, 0]
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
    else:
        tau_true = tau_scale * X[:, 0]
    
    Y = T * (Y0 + tau_true) + (1 - T) * Y0
    return X, T, Y, tau_true

# ============================================================================
# MODELS
# ============================================================================

class NaiveATE:
    def fit(self, X, T, Y):
        self.tau = np.mean(Y[T==1]) - np.mean(Y[T==0])
        return self
    def predict(self, X):
        return np.full(X.shape[0], self.tau)

class SLearner:
    def __init__(self, base=None):
        if base is None:
            base = LinearRegression()
        self.base = base
    def fit(self, X, T, Y):
        self.model = clone(self.base)
        self.model.fit(np.column_stack([X, T]), Y)
        self.p = X.shape[1]
        return self
    def predict(self, X):
        return (self.model.predict(np.column_stack([X, np.ones(X.shape[0])])) - 
                self.model.predict(np.column_stack([X, np.zeros(X.shape[0])])))

class TLearner:
    def __init__(self, base=None):
        if base is None:
            base = LinearRegression()
        self.base = base
    def fit(self, X, T, Y):
        self.m1 = clone(self.base)
        self.m0 = clone(self.base)
        self.m1.fit(X[T==1], Y[T==1])
        self.m0.fit(X[T==0], Y[T==0])
        return self
    def predict(self, X):
        return self.m1.predict(X) - self.m0.predict(X)

class XLearner:
    def __init__(self, base=None):
        if base is None:
            base = LinearRegression()
        self.base = base
    def fit(self, X, T, Y):
        X1, X0 = X[T==1], X[T==0]
        Y1, Y0 = Y[T==1], Y[T==0]
        
        self.mu1 = clone(self.base)
        self.mu0 = clone(self.base)
        self.mu1.fit(X1, Y1)
        self.mu0.fit(X0, Y0)
        
        tau1 = Y1 - self.mu1.predict(X1)
        tau0 = self.mu0.predict(X0) - Y0
        
        self.t1 = clone(self.base)
        self.t0 = clone(self.base)
        self.t1.fit(X1, tau1)
        self.t0.fit(X0, tau0)
        
        self.prop = LogisticRegression(max_iter=200)
        self.prop.fit(X, T)
        return self
    def predict(self, X):
        p = np.clip(self.prop.predict_proba(X)[:, 1], 0.01, 0.99)
        return p * self.t0.predict(X) + (1 - p) * self.t1.predict(X)

# ============================================================================
# RUN
# ============================================================================

MODELS = {
    'Naive ATE': lambda: NaiveATE(),
    'S-Learner (Linear)': lambda: SLearner(LinearRegression()),
    'T-Learner (Linear)': lambda: TLearner(LinearRegression()),
    'X-Learner (Linear)': lambda: XLearner(LinearRegression()),
    'T-Learner (RF)': lambda: TLearner(RandomForestRegressor(n_estimators=20, max_depth=3, random_state=42)),
    'X-Learner (RF)': lambda: XLearner(RandomForestRegressor(n_estimators=20, max_depth=3, random_state=42)),
}

if HAS_LGBM:
    MODELS['T-Learner (LGBM)'] = lambda: TLearner(LGBMRegressor(n_estimators=20, max_depth=3, random_state=42, verbose=-1))
    MODELS['X-Learner (LGBM)'] = lambda: XLearner(LGBMRegressor(n_estimators=20, max_depth=3, random_state=42, verbose=-1))

DGPS = ['linear', 'nonlinear', 'sparse', 'no_hte', 'heteroskedastic', 'heavy_y']
SIZES = [500, 1000, 5000]
N_REPS = 10

results = []
total = len(DGPS) * len(SIZES) * len(MODELS) * N_REPS
print(f"Running {total} experiments...")
print(f"DGPs: {DGPS}")
print(f"Sample sizes: {SIZES}")
print(f"Models: {list(MODELS.keys())}")
print(f"Reps: {N_REPS}")
print("-" * 40)

count = 0
for dgp in DGPS:
    for n in SIZES:
        for name, model_fn in MODELS.items():
            for rep in range(N_REPS):
                count += 1
                
                X, T, Y, tau_true = generate_data(dgp, n, 0.5)
                
                try:
                    m = model_fn()
                    m.fit(X, T, Y)
                    pred = m.predict(X)
                    
                    if np.any(np.isnan(pred)) or np.any(np.isinf(pred)):
                        pred = np.zeros(n)
                    
                    pehe = np.sqrt(np.mean((tau_true - pred)**2))
                    ate_bias = abs(np.mean(pred) - np.mean(tau_true))
                    status = 'success'
                except Exception as e:
                    pehe = np.nan
                    ate_bias = np.nan
                    status = str(e)[:30]
                
                results.append({
                    'dgp': dgp, 'n': n, 'model': name, 'rep': rep,
                    'PEHE': pehe, 'ATE_Bias': ate_bias, 'status': status
                })
                
                if count % 100 == 0:
                    print(f"Progress: {count}/{total} ({100*count/total:.0f}%)")

df = pd.DataFrame(results)
df.to_csv('/workspace/projects/hte-benchmark/results/full_results.csv', index=False)
print(f"\nDone! Saved {len(df)} results")

# ============================================================================
# ANALYSIS
# ============================================================================

print("\n" + "="*60)
print("PEHE BY MODEL (lower = better)")
print("="*60)
pehe = df.groupby('model')['PEHE'].agg(['mean', 'std']).sort_values('mean')
print(pehe.round(4))

print("\n" + "="*60)
print("IMPROVEMENT OVER NAIVE")
print("="*60)
naive_pehe = pehe.loc['Naive ATE', 'mean']
for m, row in pehe.iterrows():
    pct = 100 * (naive_pehe - row['mean']) / naive_pehe
    print(f"{m}: {pct:+.1f}% vs Naive")

print("\n" + "="*60)
print("PEHE BY DGP")
print("="*60)
print(df.groupby('dgp')['PEHE'].mean().round(4))

print("\n" + "="*60)
print("PEHE BY SAMPLE SIZE")
print("="*60)
print(df.groupby('n')['PEHE'].mean().round(4))
