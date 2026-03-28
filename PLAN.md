# HTE Benchmark: Comparing Treatment Effect Estimators in A/B Tests

## Overview

This project benchmarks heterogeneous treatment effect (HTE) estimators across diverse data generating processes (DGPs), evaluating their accuracy, robustness, and computational efficiency.

**Created:** 2026-03-08  
**Status:** Planning complete → Ready to implement

---

## Research Questions

1. **Q1:** Which econml estimators accurately recover heterogeneous treatment effects under different data conditions?
2. **Q2:** How do bias, variance, and PEHE compare across models?
3. **Q3:** How does computational cost scale with sample size?
4. **Q4:** Which models are robust to heteroskedasticity, heavy tails, and non-linear effects?

---

## Experimental Design

### Monte Carlo Replications
- **N_REPS:** 20 (adjustable)

### Sample Sizes
- n = 500, 1,000, 5,000, 10,000

### Treatment Effect Sizes
- **Small:** τ = 0.1 (subtle effect)
- **Medium:** τ = 0.5 (noticeable effect)  
- **Large:** τ = 1.0 (strong effect)

### Treatment Assignment Mechanisms
1. **Random (RCT):** T ⊥ X, P(T=1) = 0.5
2. **Propensity-Based:** P(T=1|X) = sigmoid(X·θ), moderate overlap
3. **High Overlap:** P(T=1|X) = uniform, strong overlap
4. **Low Overlap:** P(T=1|X) = highly selective, weak overlap

---

## Data Generating Processes (DGPs)

### DGP 1: Linear HTE
```
τ(x) = β · x₁ + β₂ · x₂  (linear interaction with treatment)
Y(0) = X·α + ε
Y(1) = Y(0) + τ(x)
```
- Covariates: X ~ N(0, I)
- Error: ε ~ N(0, 1)

### DGP 2: Non-Linear HTE
```
τ(x) = sin(x₁) + x₂² + log(abs(x₃)+1)
Y(0) = X·α + ε
Y(1) = Y(0) + τ(x)
```
- Non-linear treatment effect function

### DGP 3: Sparse HTE
```
τ(x) = β₁·x₁  (only x₁ matters)
Y(0) = X·α + ε
Y(1) = Y(0) + τ(x)
```
- High-dimensional X (20+ dims), but only 1-2 matter

### DGP 4: High-Dimensional
```
p = 50 covariates
τ(x) = sum(β_i · x_i) for i in top_k
Y(0) = X·α + ε
Y(1) = Y(0) + τ(x)
```
- 50 covariates, 5-10 relevant for τ

### DGP 5: No HTE (Constant Effect)
```
τ(x) = τ (constant)
Y(0) = X·α + ε
Y(1) = Y(0) + τ
```
- Baseline: constant treatment effect

### DGP 6: Heteroskedastic
```
τ(x) = linear(x)
σ(x) = exp(x₁)  # variance changes with x
Y(0) = X·α + ε·σ(x)
Y(1) = Y(0) + τ(x)
```
- Variance depends on covariates

### DGP 7: Heavy-Tailed Y
```
τ(x) = linear(x)
Y(0) = X·α + t_df=3  # t-distribution with 3 df
Y(1) = Y(0) + τ(x)
```
- Outcome has heavy tails (skewed)

### DGP 8: Heavy-Tailed τ(x)
```
τ(x) = Cauchy-inspired extreme values
Y(0) = X·α + ε
Y(1) = Y(0) + τ(x)
```
- Treatment effect itself has extreme outliers

---

## Models to Compare

### Category A: Naive Baselines (3 models)

| # | Model | Description |
|---|-------|-------------|
| 1 | **Naive ATE** | Simple difference in means (no HTE) |
| 2 | **Stratified ATE** | ATE computed within quantile subgroups |
| 3 | **Linear Regression + Interactions** | Y ~ T + X + T×X |

### Category B: Econml Meta-Learners (8 models)

| # | Model | Base Learner | Description |
|---|-------|--------------|-------------|
| 4 | **S-Learner** | LinearRegression | Single model with treatment indicator |
| 5 | **T-Learner** | LinearRegression | Separate models for treatment/control |
| 6 | **X-Learner** | LinearRegression | T-learner + imputed effects + weighting |
| 7 | **T-Learner** | RandomForest | T-learner with RF base |
| 8 | **X-Learner** | RandomForest | X-learner with RF base |
| 9 | **T-Learner** | XGBoost | T-learner with XGBoost |
| 10 | **X-Learner** | LightGBM | X-learner with LightGBM |
| 11 | **LinearDML** | - | Linear DML with cross-fitting |

### Category C: Advanced Econml (4 models)

| # | Model | Description |
|---|-------|-------------|
| 12 | **CausalForestDML** | Doubly Robust Causal Forest |
| 13 | **DMLPlasso** | DML with group LASSO for sparsity |
| 14 | **OrthoForest** | Orthogonal Random Forest |
| 15 | **DML Cate Classifier** | Binary classification of effect magnitude |

### Category D: Alternative Libraries (optional, if time permits)

| # | Model | Library |
|---|-------|---------|
| 16 | **GRF CausalForest** | grf |
| 17 | **DoWhy** | dowhy |

---

## Evaluation Metrics

### Primary Metrics

1. **PEHE (Precision in Estimation of Heterogeneous Effect)**
   ```
   PEHE = sqrt(mean((τ(x_i) - τ_hat(x_i))²))
   ```
   Lower is better.

2. **ATE Bias**
   ```
   Bias = |E[τ_hat] - τ_true|
   ```
   Lower is better.

3. **Coverage (95% CI)**
   ```
   Coverage = P(true_τ in CI)
   ```
   Target: 0.95

### Secondary Metrics

4. **Runtime (seconds)** — computational efficiency
5. **RMSE** — overall prediction error
6. **R-squared** — variance explained

---

## Output Formats

### Files
- `results/summary_table.csv` — Model × DGP × n results
- `results/pehe_heatmap.png` — PEHE by model and DGP
- `results/bias_heatmap.png` — ATE bias heatmap
- `results/runtime_chart.png` — Runtime vs sample size
- `results/coverage_plot.png` — Coverage rates

### Console
- Progress bars during simulation
- Summary statistics after each DGP
- Final ranking table

---

## Implementation Notes

### Python Environment
- Python 3.11+
- econml 0.16.0
- pandas, numpy, scipy
- xgboost, lightgbm (if available)
- matplotlib, seaborn for visualization

### Execution Strategy
1. Generate synthetic data for each DGP
2. Fit all models on each DGP × n combination
3. Compute metrics across N_REPS
4. Aggregate and visualize results

### Known Limitations
- Some models may fail on small n or extreme DGPs → handle gracefully with try/except
- Heavy-tailed DGPs may cause numerical issues → use robust estimators

---

## To Do

- [ ] Implement DGP generators
- [ ] Implement model wrappers
- [ ] Run full simulation (20 reps × 8 DGPs × 4 n × 15 models)
- [ ] Generate visualization plots
- [ ] Produce summary tables
- [ ] Write analysis / findings document

---

## Contact

For questions or modifications, refer to this document.
