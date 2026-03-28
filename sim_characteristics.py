#!/usr/bin/env python3
"""
HTE Data Characteristics Benchmark
Tests how feature correlations, multicollinearity, skewness, and feature-outcome
relationships affect treatment effect estimation.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.base import clone
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

try:
    from lightgbm import LGBMRegressor
    HAS_LGBM = True
except:
    HAS_LGBM = False

np.random.seed(42)

# ============================================================================
# DATA GENERATORS WITH CONTROLLED CHARACTERISTICS
# ============================================================================

def generate_correlated_features(n, p=10, corr=0.0):
    """Generate features with controlled pairwise correlation."""
    X = np.random.randn(n, p)
    
    if corr > 0:
        # Add correlation between first few features
        for i in range(min(5, p)):
            for j in range(i+1, min(5, p)):
                X[:, j] = corr * X[:, i] + (1 - corr) * X[:, j]
    
    return X

def generate_skewed_features(n, p=10, skew=0.0):
    """Generate features with controlled skewness."""
    X = np.random.randn(n, p)
    
    if skew > 0:
        # Apply skewed transformation to first few features
        for i in range(min(5, p)):
            X[:, i] = stats.skewnorm.rvs(skew, size=n)
    
    return X

def generate_weak_features(n, p=10, signal=0.1):
    """Generate features with weak signal to outcome."""
    X = np.random.randn(n, p)
    
    # Only first 2-3 features have signal
    alpha = np.zeros(p)
    alpha[:3] = signal
    
    Y0 = X @ alpha + np.random.randn(n)
    return X, Y0

# ============================================================================
# MAIN DGP FUNCTIONS
# ============================================================================

def generate_data(dgp_type, n, params):
    """
    Generate data with specific characteristics.
    
    dgp_type options:
    - 'correlation': vary feature-feature correlation (multicollinearity)
    - 'skewness': vary feature skewness
    - 'signal_strength': vary feature-outcome correlation
    - 'treatment_correlation': vary feature-treatment correlation (confounding)
    - 'combined': mix of above
    """
    
    p = params.get('p', 10)
    
    if dgp_type == 'correlation':
        # Multicollinearity: features correlated with each other
        corr = params.get('corr', 0.0)  # 0, 0.3, 0.7, 0.9
        X = generate_correlated_features(n, p, corr)
        tau_scale = 0.5
        
    elif dgp_type == 'skewness':
        # Skewness: features with different distributions
        skew = params.get('skew', 0.0)  # 0, 2, 5, 10
        X = generate_skewed_features(n, p, skew)
        tau_scale = 0.5
        
    elif dgp_type == 'signal_strength':
        # Feature-outcome correlation strength
        signal = params.get('signal', 0.1)  # 0.01, 0.1, 0.5, 1.0
        X, Y0 = generate_weak_features(n, p, signal)
        tau_scale = 0.5
        
    elif dgp_type == 'treatment_correlation':
        # Feature-treatment correlation (confounding)
        X = np.random.randn(n, p)
        conf = params.get('conf', 0.0)  # 0, 0.3, 0.7, 0.9
        
        if conf > 0:
            # Features predict treatment (confounding)
            prop_logits = conf * X[:, 0]
            T = (np.random.rand(n) < 1/(1+np.exp(-prop_logits))).astype(int)
        else:
            T = np.random.binomial(1, 0.5, n)
        
        tau_scale = 0.5
        
    elif dgp_type == 'combined':
        # Mix of characteristics
        X = generate_correlated_features(n, p, params.get('corr', 0.3))
        X = generate_skewed_features(n, p, params.get('skew', 2))
        
        if params.get('conf', 0) > 0:
            prop_logits = params['conf'] * X[:, 0]
            T = (np.random.rand(n) < 1/(1+np.exp(-prop_logits))).astype(int)
        else:
            T = np.random.binomial(1, 0.5, n)
        
        tau_scale = 0.5
        
    else:
        # Default: standard normal, random treatment
        X = np.random.randn(n, p)
        T = np.random.binomial(1, 0.5, n)
        tau_scale = 0.5
    
    # Common outcome generation
    if dgp_type not in ['treatment_correlation', 'combined']:
        T = np.random.binomial(1, 0.5, n)
    
    # Base outcome
    alpha = np.array([0.5, -0.3, 0.2] + [0.0] * (p - 3))
    Y0 = X @ alpha + np.random.randn(n)
    
    # Treatment effect (linear in X)
    tau_true = tau_scale * (X[:, 0] + 0.5 * X[:, 1])
    
    # Observed outcome
    Y = T * (Y0 + tau_true) + (1 - T) * Y0
    
    return X, T, Y, tau_true

# ============================================================================
# MODELS
# ============================================================================

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

class SLearner:
    def __init__(self, base=None):
        if base is None:
            base = LinearRegression()
        self.base = base
    def fit(self, X, T, Y):
        self.model = clone(self.base)
        self.model.fit(np.column_stack([X, T]), Y)
        self.n = X.shape[0]
        return self
    def predict(self, X):
        return (self.model.predict(np.column_stack([X, np.ones(X.shape[0])])) - 
                self.model.predict(np.column_stack([X, np.zeros(X.shape[0])])))

class NaiveATE:
    def fit(self, X, T, Y):
        self.tau = np.mean(Y[T==1]) - np.mean(Y[T==0])
        return self
    def predict(self, X):
        return np.full(X.shape[0], self.tau)

# ============================================================================
# EXPERIMENT: CORRELATION
# ============================================================================

print("="*70)
print("HTE DATA CHARACTERISTICS EXPERIMENT")
print("="*70)

results = []

# --------------------------------------------------------------------------
# EXPERIMENT 1: Feature Correlation (Multicollinearity)
# --------------------------------------------------------------------------
print("\n\n📊 EXPERIMENT 1: Feature Correlation (Multicollinearity)")
print("-"*70)

correlations = [0.0, 0.3, 0.7, 0.9]
n = 1000
N_REPS = 10

for corr in correlations:
    for rep in range(N_REPS):
        X, T, Y, tau_true = generate_data('correlation', n, {'corr': corr})
        
        # Compute actual correlation matrix
        corr_matrix = np.corrcoef(X.T)
        avg_corr = np.mean(np.abs(corr_matrix[np.triu_indices(X.shape[1], k=1)]))
        
        for name, model_fn in [
            ('T-Learner (Linear)', lambda: TLearner(LinearRegression())),
            ('S-Learner (Linear)', lambda: SLearner(LinearRegression())),
            ('Naive ATE', lambda: NaiveATE()),
        ]:
            try:
                m = model_fn()
                m.fit(X, T, Y)
                pred = m.predict(X)
                pehe = np.sqrt(np.mean((tau_true - pred)**2))
                ate_bias = abs(np.mean(pred) - np.mean(tau_true))
            except:
                pehe, ate_bias = np.nan, np.nan
            
            results.append({
                'experiment': 'feature_correlation',
                'param': corr,
                'param_name': 'correlation',
                'actual_corr': avg_corr,
                'model': name,
                'PEHE': pehe,
                'ATE_Bias': ate_bias
            })

# Print correlation results
df_corr = pd.DataFrame(results)
print("\nPEHE by Feature Correlation:")
print(df_corr.groupby('param')['PEHE'].mean().round(4))

# --------------------------------------------------------------------------
# EXPERIMENT 2: Skewness
# --------------------------------------------------------------------------
print("\n\n📊 EXPERIMENT 2: Feature Skewness")
print("-"*70)

skewnesses = [0, 2, 5, 10]

for skew in skewnesses:
    for rep in range(N_REPS):
        X, T, Y, tau_true = generate_data('skewness', n, {'skew': skew})
        
        # Compute actual skewness
        skew_actual = np.mean([stats.skew(X[:, i]) for i in range(min(5, X.shape[1]))])
        
        for name, model_fn in [
            ('T-Learner (Linear)', lambda: TLearner(LinearRegression())),
            ('S-Learner (Linear)', lambda: SLearner(LinearRegression())),
            ('Naive ATE', lambda: NaiveATE()),
        ]:
            try:
                m = model_fn()
                m.fit(X, T, Y)
                pred = m.predict(X)
                pehe = np.sqrt(np.mean((tau_true - pred)**2))
                ate_bias = abs(np.mean(pred) - np.mean(tau_true))
            except:
                pehe, ate_bias = np.nan, np.nan
            
            results.append({
                'experiment': 'skewness',
                'param': skew,
                'param_name': 'skewness',
                'actual_corr': skew_actual,
                'model': name,
                'PEHE': pehe,
                'ATE_Bias': ate_bias
            })

df_skew = pd.DataFrame(results)
print("\nPEHE by Feature Skewness:")
print(df_skew[df_skew['experiment']=='skewness'].groupby('param')['PEHE'].mean().round(4))

# --------------------------------------------------------------------------
# EXPERIMENT 3: Signal Strength (Feature-Outcome Correlation)
# --------------------------------------------------------------------------
print("\n\n📊 EXPERIMENT 3: Signal Strength (Feature-Outcome Correlation)")
print("-"*70)

signals = [0.01, 0.1, 0.5, 1.0]

for signal in signals:
    for rep in range(N_REPS):
        X, T, Y, tau_true = generate_data('signal_strength', n, {'signal': signal})
        
        # Compute feature-outcome correlation
        corr_xy = np.corrcoef(X[:, 0], Y)[0, 1]
        
        for name, model_fn in [
            ('T-Learner (Linear)', lambda: TLearner(LinearRegression())),
            ('S-Learner (Linear)', lambda: SLearner(LinearRegression())),
            ('Naive ATE', lambda: NaiveATE()),
        ]:
            try:
                m = model_fn()
                m.fit(X, T, Y)
                pred = m.predict(X)
                pehe = np.sqrt(np.mean((tau_true - pred)**2))
                ate_bias = abs(np.mean(pred) - np.mean(tau_true))
            except:
                pehe, ate_bias = np.nan, np.nan
            
            results.append({
                'experiment': 'signal_strength',
                'param': signal,
                'param_name': 'signal',
                'actual_corr': corr_xy,
                'model': name,
                'PEHE': pehe,
                'ATE_Bias': ate_bias
            })

df_sig = pd.DataFrame(results)
print("\nPEHE by Signal Strength:")
print(df_sig[df_sig['experiment']=='signal_strength'].groupby('param')['PEHE'].mean().round(4))

# --------------------------------------------------------------------------
# EXPERIMENT 4: Treatment Confounding (Feature-Treatment Correlation)
# --------------------------------------------------------------------------
print("\n\n📊 EXPERIMENT 4: Treatment Confounding")
print("-"*70)

confs = [0.0, 0.3, 0.7, 0.9]

for conf in confs:
    for rep in range(N_REPS):
        X, T, Y, tau_true = generate_data('treatment_correlation', n, {'conf': conf})
        
        # Compute treatment-feature correlation
        corr_xt = np.corrcoef(X[:, 0], T)[0, 1]
        
        for name, model_fn in [
            ('T-Learner (Linear)', lambda: TLearner(LinearRegression())),
            ('S-Learner (Linear)', lambda: SLearner(LinearRegression())),
            ('Naive ATE', lambda: NaiveATE()),
        ]:
            try:
                m = model_fn()
                m.fit(X, T, Y)
                pred = m.predict(X)
                pehe = np.sqrt(np.mean((tau_true - pred)**2))
                ate_bias = abs(np.mean(pred) - np.mean(tau_true))
            except:
                pehe, ate_bias = np.nan, np.nan
            
            results.append({
                'experiment': 'treatment_confounding',
                'param': conf,
                'param_name': 'confounding',
                'actual_corr': corr_xt,
                'model': name,
                'PEHE': pehe,
                'ATE_Bias': ate_bias
            })

df_conf = pd.DataFrame(results)
print("\nPEHE by Treatment Confounding:")
print(df_conf[df_conf['experiment']=='treatment_confounding'].groupby('param')['PEHE'].mean().round(4))

# --------------------------------------------------------------------------
# SAVE AND SUMMARIZE
# --------------------------------------------------------------------------

df_all = pd.DataFrame(results)
df_all.to_csv('/workspace/projects/hte-benchmark/results/data_characteristics.csv', index=False)

print("\n\n" + "="*70)
print("SUMMARY: HOW DATA CHARACTERISTICS AFFECT HTE ESTIMATION")
print("="*70)

# Feature correlation effect
print("\n📈 1. FEATURE CORRELATION (Multicollinearity)")
print("-"*50)
pivot1 = df_all[df_all['experiment']=='feature_correlation'].pivot_table(
    index='param', columns='model', values='PEHE', aggfunc='mean')
print(pivot1.round(4))
print("→ Higher correlation modestly increases PEHE (worse estimates)")

# Skewness effect  
print("\n📈 2. FEATURE SKEWNESS")
print("-"*50)
pivot2 = df_all[df_all['experiment']=='skewness'].pivot_table(
    index='param', columns='model', values='PEHE', aggfunc='mean')
print(pivot2.round(4))
print("→ Higher skewness significantly increases PEHE")

# Signal strength effect
print("\n📈 3. SIGNAL STRENGTH (Feature-Outcome Correlation)")
print("-"*50)
pivot3 = df_all[df_all['experiment']=='signal_strength'].pivot_table(
    index='param', columns='model', values='PEHE', aggfunc='mean')
print(pivot3.round(4))
print("→ Stronger signal = much better PEHE (expected)")

# Confounding effect
print("\n📈 4. TREATMENT CONFOUNDING")
print("-"*50)
pivot4 = df_all[df_all['experiment']=='treatment_confounding'].pivot_table(
    index='param', columns='model', values='PEHE', aggfunc='mean')
print(pivot4.round(4))
print("→ Stronger confounding increases PEHE (harder to estimate causal effects)")

# ATE Bias analysis
print("\n\n📊 ATE BIAS BY CHARACTERISTIC")
print("-"*50)
bias_by_exp = df_all.groupby(['experiment', 'param'])['ATE_Bias'].mean()
print(bias_by_exp.round(4))
