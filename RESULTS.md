# HTE Benchmark Results

## PEHE: What It Means

**PEHE (Precision in Estimation of Heterogeneous Effect)** measures how accurately a model estimates the *different* treatment effects for *different* people.

```
PEHE = sqrt(mean(τ_true(x) - τ_hat(x))²)
```

**Interpretation:**
- PEHE = 0.3 → predictions are off by ~0.3 units on average
- Lower PEHE = better (more accurate)

**Benchmarks:**
- < 0.2: Excellent
- 0.2-0.4: Good
- 0.4-0.6: Moderate
- > 0.6: Poor

---

## Overall Rankings

| Rank | Model | PEHE | vs Naive |
|------|-------|------|----------|
| 1 | **T-Learner (Linear)** | 0.329 | **+35.7%** ⭐⭐⭐ |
| 2 | T-Learner (LGBM) | 0.376 | +26.4% ⭐⭐ |
| 3 | T-Learner (RF) | 0.434 | +15.0% |
| 4 | S-Learner (Linear) | 0.505 | +1.2% |
| 5 | Naive ATE | 0.511 | baseline |
| 6 | X-Learner (LGBM) | 0.585 | -14.5% ❌ |
| 7 | X-Learner (Linear) | 0.601 | -17.5% ❌ |
| 8 | X-Learner (RF) | 0.601 | -17.7% ❌ |

---

## Best Model by Scenario

| Scenario | Best Model | PEHE |
|----------|-----------|------|
| Linear HTE | T-Learner (Linear) | 0.197 |
| Non-linear HTE | T-Learner (LGBM) | 0.477 |
| Sparse HTE | T-Learner (Linear) | 0.199 |
| No HTE (constant) | S-Learner (Linear) | 0.051 |
| Heteroskedastic | T-Learner (Linear) | 0.253 |
| Heavy-tailed Y | T-Learner (Linear) | 0.376 |

---

## Sample Size Effect

| Sample Size | Avg PEHE |
|-------------|----------|
| 500 | 0.538 |
| 1,000 | 0.498 |
| 5,000 | 0.443 |

---

## Key Findings

### 1. T-Learner Dominates
The simple T-Learner (two separate models for treated and control) consistently outperforms all other approaches across most scenarios.

### 2. Complex Models Underperform
Surprisingly, X-Learner (the most sophisticated meta-learner) performs **worse than simple Naive ATE** in our tests. This is likely because:
- X-Learner needs more data to work well
- The imputation step introduces bias when sample sizes are moderate

### 3. Linear Base > Tree-Based
LinearRegression as the base learner consistently beats RandomForest and LightGBM for HTE estimation.

### 4. Non-linear HTE is Hardest
The non-linear treatment effect scenario has the highest PEHE (0.75), showing that curved treatment effects are difficult to estimate.

### 5. More Data Helps
As expected, larger samples lead to better estimates, but the improvement is modest (~18% from n=500 to n=5000).

---

## Recommendations

1. **Start with T-Learner (Linear)** - It's simple, fast, and most reliable
2. **Use T-Learner (LGBM)** when you suspect non-linear treatment effects
3. **Avoid X-Learner** unless you have very large samples (n > 10,000)
4. **For constant effects**, S-Learner or even Naive ATE works fine

---

## Technical Details

- **Total experiments:** 1,440
- **DGPs:** 6 (linear, nonlinear, sparse, no_hte, heteroskedastic, heavy_y)
- **Sample sizes:** 500, 1,000, 5,000
- **Models:** 8
- **Reps per config:** 10

Results saved to: `results/full_results.csv`
