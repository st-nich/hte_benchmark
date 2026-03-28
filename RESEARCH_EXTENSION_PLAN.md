# HTE Research Extension Plan

## Current Status

### Completed
- ✅ Baseline model comparisons (T-Learner, S-Learner, X-Learner, LinearDML)
- ✅ Data characteristics (multicollinearity, skewness, confounding)
- ✅ Sparse effects analysis
- ✅ Power analysis (effect size × noise × responder %)
- ✅ Feature selection with Lasso
- ✅ Large sample size analysis
- ✅ Bidirectional effects

---

## Phase 1: Immediate Extensions (Quick Wins)

### 1.1 More Realistic DGPs
- [ ] **Covariate shift**: Training on different distribution than test
- [ ] **Temporal effects**: Treatment effects that change over time
- [ ] **Network effects**: Users who interact (SUTVA violation)
- [ ] **Longitudinal**: Repeated measurements on same users

### 1.2 Model Extensions
- [ ] **CausalForest** from econml (proper implementation)
- [ ] **DoWhy** integration for comparison
- [ ] **Doubly Robust learners** (AIPW)
- [ ] **OrthoForest** for nonparametric HTE

### 1.3 Evaluation Metrics
- [ ] **Policy risk**: If we target top X%, what's the actual uplift?
- [ ] **CATE accuracy** at different percentiles
- [ ] **Coverage** of confidence intervals
- [ ] **Heterogeneity detection** (can we correctly identify who responds?)

---

## Phase 2: Real-World Validation

### 2.1 Observational Data
- [ ] **Propensity score matching** for non-randomized data
- [ ] **Instrumental variables** for unmeasured confounding
- [ ] **Difference-in-differences** for before/after designs
- [ ] **Regression discontinuity** for threshold-based assignments

### 2.2 Industry Benchmarks
- [ ] Replicate on **public datasets**:
  - [ ] Lalonde (job training) - **IN PROGRESS** (code written)
  - [ ] IHDP (infant health)
  - [ ] ACIC benchmarks
- [ ] **A/B test data** from industry partners (if available)

---

## Phase 3: Advanced Methods

### 3.1 Continuous Treatments
- [ ] Dosing effects (not just treated vs control)
- [ ] Price elasticity estimation
- [ ] Treatment dosage optimization

### 3.2 Multiple Treatments
- [ ] A/B/n tests (more than 2 variants)
- [ ] Treatment combinations
- [ ] Dynamic treatment regimes

### 3.3 Longitudinal HTE
- [ ] Panel data with repeated measurements
- [ ] Time-varying treatment effects
- [ ] Sustained vs temporary effects

### 3.4 Network Effects
- [ ] Interference between users
- [ ] Viral/spread effects
- [ ] Social network HTE

---

## Phase 4: Production Readiness

### 4.1 Inference & Uncertainty
- [ ] **Confidence intervals** for CATE estimates
- [ ] **Uncertainty quantification** (Bayesian methods)
- [ ] **Statistical significance** testing for heterogeneity
- [ ] **False discovery rate** control

### 4.2 Deployment Considerations
- [ ] **Online learning**: Updating models as data streams
- [ ] **Monitoring**: Detecting model drift
- [ ] **A/B test design**: How to power experiments for HTE
- [ ] **Sample size calculators** for HTE

### 4.3 Interpretability
- [ ] Feature importance for treatment effects
- [ ] Counterfactual explanations
- [ ] Visualization tools

---

## Phase 5: Novel Contributions

### 5.1 New Methods
- [ ] **Adaptive HTE**: Targeting where it's hardest to estimate
- [ ] **Transfer learning**: Using historical experiments
- [ ] **Multi-task learning**: Sharing info across experiments
- [ ] **Bandit-based**: Exploration-exploitation in HTE

### 5.2 Theory
- [ ] **Minimax rates** for HTE under different assumptions
- [ ] **Optimal design** for HTE experiments
- [ ] **Bounds** for unmeasured confounding

### 5.3 Software
- [ ] Open-source Python package
- [ ] AutoML for HTE (auto-select best method)
- [ ] Integration with experiment platforms

---

## Priority Ranking

| Priority | Topic | Impact | Feasibility |
|----------|-------|--------|--------------|
| 1 | CATE uncertainty/intervals | High | Medium |
| 2 | Real-world datasets | High | High |
| 3 | Continuous treatments | High | Medium |
| 4 | Propensity matching | High | High |
| 5 | Policy risk metric | High | High |
| 6 | Longitudinal effects | Medium | Medium |
| 7 | Network effects | High | Low |
| 8 | New methods | High | Low |

---

## Resources Needed

### Data
- Public causal inference benchmarks
- Industry A/B test datasets (partnerships)
- Synthetic data generators

### Compute
- ~100 GPU hours for full grid search
- ~50 CPU hours for baseline experiments

### People
- 1-2 ML engineers for implementation
- 1 researcher for methodology
- 1 engineer for visualization

---

## Timeline Estimate

| Phase | Effort | Time |
|-------|--------|------|
| Phase 1 | 20% | 1-2 weeks |
| Phase 2 | 30% | 2-3 weeks |
| Phase 3 | 25% | 2-3 weeks |
| Phase 4 | 15% | 1-2 weeks |
| Phase 5 | 10% | Ongoing |

---

## Key Research Questions to Answer

1. **When does HTE actually work in practice?**
2. **What's the minimum sample size for reliable HTE?**
3. **How do we quantify uncertainty in HTE estimates?**
4. **Can we transfer learning across experiments?**
5. **What's the economic value of accurate HTE?**

---

## Next Steps

1. **Immediate**: Add CATE confidence intervals to current benchmark
2. **This month**: Test on public datasets (Lalonde, IHDP)
3. **Next month**: Implement continuous treatment methods
4. **Ongoing**: Build out production-ready pipeline

---

*Last updated: March 2026*
