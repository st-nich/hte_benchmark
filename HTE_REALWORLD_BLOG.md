# HTE Benchmark on Real-World Data

## Executive Summary

This post validates our synthetic benchmarks on **real-world causal inference datasets**:
- Lalonde (job training program impact on earnings)
- IHDP (infant health intervention impact on IQ)

We test whether our findings (T-Learner wins, X-Learner rarely wins, etc.) hold on real data with actual confounding.

---

## Datasets

### Lalonde (1986)

**Source:** National Supported Work (NSW) Demonstration

**What:** Does a job training program increase earnings?

| Aspect | Value |
|--------|-------|
| n | 1,615 |
| Treatment | Job training (vs. no training) |
| Outcome | Real earnings in 1978 (re78) |
| Features | Age, education, race, marital status, pre-treatment earnings |

**Key challenge:** Non-random selection into treatment (observational)

---

### IHDP (Infant Health and Development Program)

**Source:** Randomized controlled trial

**What:** Does early intervention improve outcomes for premature infants?

| Aspect | Value |
|--------|-------|
| n | 985 |
| Treatment | Intensive early intervention vs. standard care |
| Outcome | IQ score at age 3 |
| Features | Birth weight, gestational age, mother's education, income |

**Key challenge:** Selective attrition (observational after initial randomization)

---

## Methods

### Models Tested

1. **Naive ATE** — Simple difference in means
2. **S-Learner** — Single model with treatment indicator
3. **T-Learner** — Separate models for treated/control
4. **X-Learner** — Imputation + weighting
5. **LinearDML** — Doubly machine learning
6. **T-Learner + Lasso** — With L1 regularization

### Evaluation

Since real data doesn't have ground-truth CATE:

1. **Predictive accuracy:** Hold-out RMSE of outcome prediction
2. **Policy learning:** If we target top X%, what's the actual uplift?
3. **Comparison to benchmarks:** Compare to published econml results

---

## Expected Findings

Based on our synthetic experiments:

| Finding | Expected on Real Data? |
|---------|----------------------|
| T-Learner wins overall | ✅ Likely |
| Confounding hurts Naive | ✅ Should be stronger effect |
| X-Learner rarely wins | ✅ Expected |
| Lasso helps in high-dim | Need to verify |

---

## Code to Run

```python
"""
HTE Benchmark on Real-World Data
Run this locally with: pip install econml causaldata pandas numpy scikit-learn
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.base import clone
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# DATA LOADING
# ============================================================================

def load_lalonde():
    """Load Lalonde dataset."""
    from causaldata import load_lalonde
    df = load_lalonde().load_pandas().data
    
    # Define features
    feature_cols = ['age', 'educ', 'black', 'hisp', 'married', 'nodegree', 
                    're74', 're75', 'u74', 'u75']
    
    X = df[feature_cols].fillna(0).values
    T = df['treat'].values  # Treatment indicator
    Y = df['re78'].values    # Outcome (earnings)
    
    return X, T, Y, feature_cols


def load_ihdp():
    """Load IHDP dataset."""
    from causaldata import load_ihdp
    df = load_ihdp().load_pandas().data
    
    # Define features  
    feature_cols = ['bwg', 'female', 'black', 'hisp', 'bired', 
                    'married', 'hsed', 'age', 'workyear']
    
    X = df[feature_cols].fillna(0).values
    T = df['treat'].values  # Treatment indicator
    Y = df['kidiq'].values   # Outcome (IQ score)
    
    return X, T, Y, feature_cols


# ============================================================================
# MODELS
# ============================================================================

class TLearner:
    """T-Learner with configurable base model."""
    def __init__(self, base=None):
        self.base = base or LinearRegression()
    
    def fit(self, X, T, Y):
        self.m1 = clone(self.base)
        self.m0 = clone(self.base)
        self.m1.fit(X[T==1], Y[T==1])
        self.m0.fit(X[T==0], Y[T==0])
        return self
    
    def predict(self, X):
        return self.m1.predict(X) - self.m0.predict(X)
    
    def predict_ate(self):
        return np.mean(self.predict(self.X))


class SLearner:
    """S-Learner with treatment indicator."""
    def __init__(self, base=None):
        self.base = base or LinearRegression()
    
    def fit(self, X, T, Y):
        X_design = np.column_stack([X, T])
        self.model = clone(self.base)
        self.model.fit(X_design, Y)
        self.n = X.shape[0]
        return self
    
    def predict(self, X):
        return (self.model.predict(np.column_stack([X, np.ones(self.n)])) - 
                self.model.predict(np.column_stack([X, np.zeros(self.n)])))


class XLearner:
    """X-Learner with imputation and weighting."""
    def __init__(self, base=None):
        self.base = base or LinearRegression()
    
    def fit(self, X, T, Y):
        X1, X0 = X[T==1], X[T==0]
        Y1, Y0 = Y[T==1], Y[T==0]
        
        # Stage 1: Estimate mu
        self.mu1 = clone(self.base)
        self.mu0 = clone(self.base)
        self.mu1.fit(X1, Y1)
        self.mu0.fit(X0, Y0)
        
        # Impute treatment effects
        tau1 = Y1 - self.mu1.predict(X1)
        tau0 = self.mu0.predict(X0) - Y0
        
        # Stage 2: Model imputed effects
        self.tau1 = clone(self.base)
        self.tau0 = clone(self.base)
        self.tau1.fit(X1, tau1)
        self.tau0.fit(X0, tau0)
        
        # Propensity
        self.prop = LogisticRegression(max_iter=500)
        self.prop.fit(X, T)
        
        return self
    
    def predict(self, X):
        p = np.clip(self.prop.predict_proba(X)[:, 1], 0.01, 0.99)
        return p * self.tau0.predict(X) + (1-p) * self.tau1.predict(X)


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_models(X, T, Y, dataset_name, n_reps=10):
    """Evaluate all models on a dataset."""
    from sklearn.metrics import mean_squared_error
    
    results = []
    
    models = {
        'Naive ATE': lambda: None,  # Special case
        'S-Learner': lambda: SLearner(LinearRegression()),
        'T-Learner': lambda: TLearner(LinearRegression()),
        'T-Learner (Lasso)': lambda: TLearner(Lasso(alpha=1.0)),
        'T-Learner (RF)': lambda: TLearner(RandomForestRegressor(n_estimators=50, max_depth=5)),
        'X-Learner': lambda: XLearner(RandomForestRegressor(n_estimators=50, max_depth=5)),
    }
    
    for name, model_fn in models.items():
        rmse_list = []
        
        for rep in range(n_reps):
            # Train-test split
            X_train, X_test, T_train, T_test, Y_train, Y_test = train_test_split(
                X, T, Y, test_size=0.3, random_state=rep
            )
            
            if name == 'Naive ATE':
                # Just predict average treatment effect
                tau_hat = Y_train[T_train==1].mean() - Y_train[T_train==0].mean()
                pred = np.full(len(X_test), tau_hat)
            else:
                # Fit model
                model = model_fn()
                model.fit(X_train, T_train, Y_train)
                pred = model.predict(X_test)
            
            # RMSE of outcome prediction
            # (We predict Y(1) - Y(0) = CATE, so compute outcome RMSE)
            Y_pred = pred * T_test + (Y_train.mean() - pred.mean() * T_train.mean())
            rmse = np.sqrt(mean_squared_error(Y_test, Y_pred))
            rmse_list.append(rmse)
        
        results.append({
            'dataset': dataset_name,
            'model': name,
            'rmse': np.mean(rmse_list),
            'rmse_std': np.std(rmse_list)
        })
    
    return pd.DataFrame(results)


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    print("Loading datasets...")
    
    # Lalonde
    print("\n" + "="*60)
    print("LALONDE DATASET")
    print("="*60)
    try:
        X, T, Y, features = load_lalonde()
        print(f"n = {len(Y)}, features = {X.shape[1]}")
        print(f"Treatment: {T.sum()} ({100*T.mean():.1f}%)")
        print(f"Outcome mean: {Y.mean():.2f}, std: {Y.std():.2f}")
        
        results_lalonde = evaluate_models(X, T, Y, 'Lalonde')
        print("\nResults:")
        print(results_lalonde.sort_values('rmse'))
    except Exception as e:
        print(f"Error loading Lalonde: {e}")
    
    # IHDP
    print("\n" + "="*60)
    print("IHDP DATASET")
    print("="*60)
    try:
        X, T, Y, features = load_ihdp()
        print(f"n = {len(Y)}, features = {X.shape[1]}")
        print(f"Treatment: {T.sum()} ({100*T.mean():.1f}%)")
        print(f"Outcome mean: {Y.mean():.2f}, std: {Y.std():.2f}")
        
        results_ihdp = evaluate_models(X, T, Y, 'IHDP')
        print("\nResults:")
        print(results_ihdp.sort_values('rmse'))
    except Exception as e:
        print(f"Error loading IHDP: {e}")
```

---

## Expected Results (from literature)

### Lalonde

| Model | Expected RMSE |
|-------|--------------|
| T-Learner | Baseline |
| S-Learner | Similar |
| X-Learner | May help with selection |
| LinearDML | Good with propensity |

### IHDP

| Model | Expected RMSE |
|-------|--------------|
| T-Learner | Good |
| LinearDML | Best (handles selection) |
| With Lasso | Better with few samples |

---

## Key Questions to Answer

1. Do our synthetic findings hold on real data?
2. Does confounding (in Lalonde) change which model wins?
3. Does Lasso help with small sample sizes (IHDP n~500)?

---

## Running This

```bash
# Install dependencies
pip install econml causaldata pandas numpy scikit-learn

# Run benchmark
python hte_realworld_benchmark.py
```

---

*Coming soon: Results from running this benchmark on real data.*
