# Heterogeneous Treatment Effects in A/B Testing

A comprehensive benchmark comparing causal inference methods for detecting who actually responds to your experiments.

## Overview

When running A/B tests, we often report the *average* treatment effect — but this masks important variation. Some users may love your changes, others may hate them, and some may not care at all.

**Heterogeneous Treatment Effects (HTE)** aim to answer: *Who* benefits from this treatment?

This repository benchmarks multiple HTE estimation methods across diverse data conditions to find what works in practice.

## Key Findings

| Rank | Method | RMSE | vs Naive |
|------|--------|------|----------|
| 🥇 | **T-Learner (Linear)** | 0.33 | +36% |
| 🥈 | T-Learner (LGBM) | 0.38 | +26% |
| 🥉 | T-Learner (RF) | 0.43 | +15% |
| 4 | S-Learner | 0.51 | +1% |
| 5 | Naive ATE | 0.51 | baseline |
| 6-8 | X-Learner | 0.59-0.60 | **-18%** ❌ |

### The Bottom Line

- **T-Learner wins** — Simple two-model approach beats sophisticated alternatives
- **X-Learner underperforms** — Often worse than doing nothing fancy
- **Confounding is the killer** — Increases RMSE by 400%+ for Naive ATE
- **Bidirectional effects are dangerous** — When some benefit and some are harmed, Naive ATE says "no effect"
- **Sample size alone doesn't help** — Model choice matters more than data size

## Contents

### Blog Posts

- **[HTE_RESEARCH_BLOG.md](./HTE_RESEARCH_BLOG.md)** — Main benchmark with full results
- **[HTE_REALWORLD_BLOG.md](./HTE_REALWORLD_BLOG.md)** — Validation on real-world datasets (Lalonde, IHDP)

### Code

| File | Description |
|------|-------------|
| `sim_fast.py` | Fast benchmark runner |
| `sim_characteristics.py` | Data characteristics analysis |
| `simulate.py` | Full simulation (slower) |
| `visualizations.py` | Matplotlib plots (run locally) |

### Results

- `results/full_results.csv` — All experiment results
- `results/data_characteristics.csv` — Confounding/correlations analysis

## Quick Start

```python
# Install dependencies
pip install pandas numpy scikit-learn lightgbm econml

# Run benchmark
python sim_fast.py
```

## Methods Tested

### Simple Baselines
- **Naive ATE** — Difference in means (no heterogeneity)
- **Stratified ATE** — ATE within subgroups

### Meta-Learners
- **S-Learner** — Single model: Y ~ T + X + T×X
- **T-Learner** — Separate models for treated/control
- **X-Learner** — Imputation + weighting

### ML-Based
- **T-Learner (RF)** — T-Learner with Random Forest
- **T-Learner (LGBM)** — T-Learner with LightGBM
- **LinearDML** — Doubly Machine Learning

## Data Scenarios Tested

| Scenario | Description |
|----------|-------------|
| Linear HTE | τ(x) = β·x |
| Non-linear | τ(x) = sin(x₁) + x₂² |
| Sparse | Only a few features matter |
| High-dimensional | 50 features, 5 relevant |
| No HTE | Constant effect |
| Heteroskedastic | Variance changes with X |
| Heavy-tailed | Outliers in outcome |
| Bidirectional | Some benefit, some harmed |

## Key Takeaways

1. **Start with T-Learner** — Simple, fast, reliable
2. **Use Lasso for high-dimensional** — 2.5x better RMSE
3. **Avoid X-Learner** unless n > 10K with specific conditions
4. **Randomization is crucial** — Without it, HTE estimates are biased

## Citation

If you use this work, please cite:

```bibtex
@software{hte-benchmark,
  title = {HTE Benchmark for A/B Testing},
  author = {Research Team},
  year = {2026},
  url = {https://github.com/your-repo/hte-benchmark}
}
```

## License

MIT

---

*For questions or contributions, open an issue or PR.*
