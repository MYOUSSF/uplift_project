# Uplift Modelling with Meta-Learners
### Causal Data Science

> **Core question:** *Which customers will actually change their behaviour because of a marketing email — and which would have converted anyway?*

Uplift modelling (also called **heterogeneous treatment effect estimation** or **CATE estimation**) answers exactly this. Instead of predicting *who converts*, it estimates *by how much* an intervention changes each individual's probability of converting. This distinction is the heart of causal data science.

---

## Table of Contents

- [Problem Statement](#problem-statement)
- [Dataset](#dataset)
- [Methods](#methods)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Results](#results)
- [Key Visualisations](#key-visualisations)
- [Causal Framework](#causal-framework)
- [Limitations & Assumptions](#limitations--assumptions)
- [References](#references)

---

## Problem Statement

A retailer sends promotional emails to a random sample of 64,000 customers. Some receive a men's catalogue, some a women's catalogue, and some receive nothing. Two weeks later, we observe website visits, conversions, and revenue.

**The naive ML approach** trains a model on treated customers and ranks by predicted conversion probability. This fails because it confuses *responders* with *always-buyers* — customers who would have converted without any email.

**Uplift modelling** estimates the **Individual Treatment Effect (ITE)**:

```
τ(x) = E[Y(1) | X=x] − E[Y(0) | X=x]
```

where `Y(1)` is potential outcome *with* treatment and `Y(0)` without. We target customers with high `τ(x)` — the genuine persuadables.

The four customer archetypes:

| Archetype | Without email | With email | Target? |
|-----------|:---:|:---:|:---:|
| **Persuadable** | No conversion | Converts | ✅ Yes |
| **Sure thing** | Converts | Converts | ❌ Wasted budget |
| **Lost cause** | No conversion | No conversion | ❌ Wasted budget |
| **Do-not-disturb** | Converts | No conversion | ❌ Harmful! |

---

## Dataset

**Hillstrom MineThatData E-Mail Analytics Challenge (2008)**

| Property | Value |
|---|---|
| Source | [minethatdata.com](https://www.minethatdata.com/Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv) |
| Rows | 64,000 customers |
| Treatment arms | No E-Mail / Mens E-Mail / Womens E-Mail |
| Assignment | Randomised (true A/B/C experiment) |
| Outcomes | `visit` (binary), `conversion` (binary), `spend` (continuous) |

**Key features:**

| Feature | Description |
|---|---|
| `recency` | Months since last purchase |
| `history` | Dollar value of past purchases |
| `mens` / `womens` | 1 if purchased that merchandise category |
| `newbie` | New customer (<12 months) |
| `zip_code` | Rural / Suburban / Urban |
| `channel` | Phone / Web / Multichannel |

Because the treatment is **randomised**, we can use it as a ground truth for causal identification. No instrumental variables or propensity weighting needed — the experiment gives us valid causal estimates directly.

---

## Methods

Three meta-learner architectures are implemented from scratch using XGBoost as the base learner:

### S-Learner (Single model)

Trains **one** outcome model with treatment as a feature:

```
μ(x, t) = E[Y | X=x, T=t]
τ(x)    = μ(x, 1) − μ(x, 0)
```

Simple but may underweight the treatment indicator in regularised models.

### T-Learner (Two models)

Trains **separate** models for treated and control groups:

```
μ₀(x) = E[Y | X=x, T=0]
μ₁(x) = E[Y | X=x, T=1]
τ(x)  = μ₁(x) − μ₀(x)
```

Better than S-Learner when treatment groups are large. Suffers with unbalanced arms.

### X-Learner (Cross-fitting)

Extends T-Learner with imputed treatment effects and propensity blending (Künzel et al. 2019):

```
Stage 1: fit μ₀, μ₁ (same as T-Learner)
Stage 2: D₁ = Y₁ − μ₀(X₁)      fit τ₁(x) on D₁
         D₀ = μ₁(X₀) − Y₀      fit τ₀(x) on D₀
Stage 3: τ(x) = g(x)·τ₀(x) + (1−g(x))·τ₁(x)
```

where `g(x)` is the propensity score. Outperforms T-Learner when treatment groups are unequal in size or signal quality.

---

## Project Structure

```
uplift-modelling/
├── main.py                      # End-to-end pipeline
├── requirements.txt
├── README.md
│
├── src/
│   ├── data_loader.py           # Download, cache & preprocess Hillstrom data
│   ├── uplift_models.py         # S/T/X-Learner implementations
│   ├── evaluation.py            # Qini curve, AUQC, cumulative gain
│   ├── targeting_policy.py      # Budget optimisation & policy logic
│   └── visualizations.py        # All 8 matplotlib charts
│
├── tests/
│   └── test_pipeline.py         # 25 unit & integration tests (pytest)
│
├── data/                        # Auto-created; gitignored
│   ├── hillstrom_raw.csv
│   └── hillstrom_processed.csv
│
└── outputs/
    ├── model_comparison.csv
    ├── budget_sweep.csv
    └── figures/
        ├── 01_eda_overview.png
        ├── 02_cate_distributions.png
        ├── 03_qini_curves.png
        ├── 04_cumulative_gain.png
        ├── 05_feature_importance.png
        ├── 06_budget_curve.png
        ├── 07_cate_heatmap.png
        └── 08_model_comparison.png
```

---

## Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/MYOUSSF/uplift_project
cd uplift_project

# 2. Create environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the full pipeline
python main.py

# 5. Run tests
pytest tests/ -v
```

**Options:**

```bash
python main.py --skip-eda          # Skip EDA charts (faster)
python main.py --budget 0.20       # Evaluate policy at 20% budget
```

The pipeline downloads the Hillstrom dataset automatically. If the remote source is unavailable, a synthetic replica matching the original schema and marginal distributions is generated instead.

---

## Results

| Model | AUQC | Notes |
|---|---|---|
| **S-Learner** | Best | Simple; treatment feature well-used by XGBoost |
| T-Learner | — | Strong; slightly higher variance |
| X-Learner | — | Most theoretically sound; minimal gains here due to balanced arms |

> **Note:** With balanced treatment arms (~33% each), the X-Learner's propensity-blending advantage is smaller. X-Learner typically dominates when arms are heavily imbalanced (e.g. 5% treated, 95% control).

### Policy outcome (30% budget)

Targeting the top 30% of customers by predicted uplift score:

| Metric | Value |
|---|---|
| Customers targeted | ~19,200 |
| Email cost | $2,880 |
| Incremental conversions | ~155 |
| Revenue lift | $7,734 |
| **Profit lift** | **$4,854** |
| **ROI** | **168.5%** |

---

## Key Visualisations

### 1. EDA Overview
Conversion rates, spend distributions, and treatment arm balance across all three campaign arms.

### 2. CATE Distributions
Histogram of individual-level predicted uplift per model. The proportion of customers with positive predicted uplift quantifies the size of the persuadable segment.

### 3. Qini Curves
The primary evaluation chart for uplift models. A model outperforming the diagonal random baseline is correctly identifying the persuadable segment.

### 4. Budget Allocation Curve
Profit lift and ROI as a function of targeting fraction. Identifies the optimal budget fraction that maximises incremental profit.

### 5. CATE Heatmap
Mean predicted treatment effect segmented by recency and historical spend. Reveals which customer segments are most receptive to the campaign.

---

## Causal Framework

This project operates under the **Rubin Causal Model** (potential outcomes framework):

**Assumptions:**

1. **SUTVA** (Stable Unit Treatment Value Assumption) — one customer's treatment does not affect another's outcome. Plausible: email marketing has negligible spillover effects.

2. **Ignorability** (unconfounded assignment) — `Y(0), Y(1) ⊥ T | X`. Satisfied here because the Hillstrom experiment is **randomised** — treatment assignment is independent of all covariates.

3. **Overlap** (positivity) — `0 < P(T=1 | X=x) < 1` for all x. Verified: balanced randomisation ensures every subgroup has non-zero probability of treatment.

Because randomisation satisfies ignorability by design, we do not need propensity score weighting or instrumental variables for identification. The experiment gives us direct access to the ATE:

```
ATE = E[Y(1) − Y(0)] = E[Y | T=1] − E[Y | T=0]
```

The meta-learners go further — they estimate the **CATE** (Conditional ATE) at the individual level, which is what enables personalised targeting.

---

## Limitations & Assumptions

**Data limitations:**
- The Hillstrom dataset is from 2008 and reflects one retailer's customer base. Conclusions may not generalise to other industries.
- Observed features may not capture all heterogeneity drivers (e.g. customer intent, browsing history).

**Modelling limitations:**
- Meta-learners estimate CATE but cannot guarantee individual-level causal claims — only distributional ones.
- XGBoost may overfit on small subgroups. Cross-validation is not fully implemented for the T-Learner's two-model setup.
- The financial projections assume constant revenue per conversion ($50) and cost per email ($0.50) — real campaigns require dynamic estimates.

**Evaluation limitations:**
- AUQC depends on the random baseline construction. With imbalanced treatment arms, the baseline shifts and scores become less comparable across datasets.
- Without a holdout A/B test on the policy itself, financial projections remain estimates.

---

## References

1. Künzel, S. R., Sekhon, J. S., Wager, S., & Yu, B. (2019). [Metalearners for estimating heterogeneous treatment effects using machine learning.](https://www.pnas.org/doi/10.1073/pnas.1804597116) *PNAS*, 116(10), 4156–4165.

2. Radcliffe, N. J. (2007). [Using control groups to target on predicted lift.](https://www.stochasticsolutions.com/pdf/sig-based-up-modelling.pdf) *Direct Marketing Analytics Journal*, 1, 14–21.

3. Hillstrom, K. (2008). [MineThatData E-Mail Analytics and Data Mining Challenge.](https://blog.minethatdata.com/2008/03/minethatdata-e-mail-analytics-and-data.html)

4. Gutierrez, P., & Gérardy, J. Y. (2017). [Causal inference and uplift modelling: A review of the literature.](http://proceedings.mlr.press/v67/gutierrez17a/gutierrez17a.pdf) *PMLR*, 67, 1–13.

5. Rubin, D. B. (1974). [Estimating causal effects of treatments in randomized and nonrandomized studies.](https://psycnet.apa.org/doi/10.1037/h0037350) *Journal of Educational Psychology*, 66(5), 688–701.

---

## Licence

MIT — free to use, adapt, and share with attribution.

---
