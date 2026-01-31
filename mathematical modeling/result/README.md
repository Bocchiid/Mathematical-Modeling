# MCM 2026 Problem C - Task 1: Results Output

## Model Overview
**Dynamic Latent Factor Model with Generative Voting Framework**

This directory contains comprehensive results from the fan vote estimation model for "Dancing with the Stars" data analysis.

---

## Output Files

### CSV Files (Data Exports)

1. **vote_estimates.csv** (4,631 rows)
   - Vote predictions for each contestant-week combination
   - Columns:
     - `Season`: Season number
     - `Week`: Week number (1-11)
     - `Contestant`: Celebrity contestant name
     - `Judge_Score_Z`: Standardized judge score
     - `Trend`: Historical performance trend
     - `Vote_Mean`: Predicted mean vote count
     - `Vote_Std`: Standard deviation of vote prediction
     - `Vote_CI_Lower`: Lower 95% confidence interval
     - `Vote_CI_Upper`: Upper 95% confidence interval

2. **model_parameters.csv**
   - Model coefficients and hyperparameters
   - `alpha`: Judge score response coefficient (0.8)
   - `beta`: Trend response coefficient (0.5)
   - `gamma`: Celebrity-week interaction (0.3)
   - `delta`: Regional effect (0.2)
   - `sigma_v`: Vote distribution variance (0.3)

3. **accuracy_by_season.csv**
   - Bottom-two elimination prediction accuracy by season
   - Columns: `Season`, `Total_Weeks`, `Correct_Predictions`, `Accuracy`

4. **feature_statistics.csv**
   - Summary statistics for model features
   - Includes: Mean, Std, Min, Max for z-scores and trends

5. **evaluation_metrics.csv**
   - Overall model performance metrics
   - Overall accuracy: 0.53%
   - Total predictions: 374 week-eliminations
   - Correct predictions: 2
   - Coverage: 34 seasons, unique contestants

---

### Visualization Files (PNG)

1. **accuracy_by_season.png**
   - Bar chart showing prediction accuracy for each season
   - Green bars: ¡Ý50% accuracy
   - Orange bars: 30-50% accuracy
   - Red bars: <30% accuracy
   - Red dashed line: 50% threshold

2. **feature_distributions.png** (4 subplots)
   - Judge Score Z-Scores histogram
   - Performance Trends histogram
   - Raw Judge Scores histogram
   - Regional Advantage Factors histogram

3. **vote_predictions.png** (4 subplots)
   - Vote distribution by judge score category
   - Scatter: Judge score vs predicted mean votes
   - Box plot: Votes by judge score quintile
   - Overall vote distribution

4. **uncertainty_metrics.png** (4 subplots)
   - Scatter: Variance vs judge performance
   - 95% CI width vs judge performance
   - Uncertainty evolution over weeks
   - Vote distribution entropy

5. **model_performance.png** (4 subplots)
   - Prediction residuals distribution
   - Q-Q plot for normality check
   - Model coefficients bar chart
   - Feature correlations

6. **celebrity_embeddings.png** (4 subplots)
   - Embedding norm trajectories over time
   - 2D embedding space visualization
   - Embedding variance evolution
   - Peak popularity by celebrity

---

## Model Description

### Generative Voting Model

Fan votes are modeled as following a lognormal distribution:
$$\log(v_{i,t}) \sim \mathcal{N}(\mu_{i,t}, \sigma_v^2)$$

Where the mean structure is:
$$\mu_{i,t} = \alpha \cdot z_{i,t} + \beta \cdot \text{Trend}_{i,t} + \gamma \cdot (u_{c,t} \cdot m_t) + \delta \cdot \phi_{i,t} + \eta_g + \tau_t$$

**Components:**
- $z_{i,t}$: Standardized judge scores within each season-week
- Trend: Historical performance improvement/decline
- $u_{c,t}$: Celebrity popularity latent factor (dynamic)
- $m_t$: Time-varying preference factor
- $\phi_{i,t}$: Regional advantage factor
- $\eta_g$: Industry fixed effects
- $\tau_t$: Week fixed effects

### Celebrity Embeddings

Popularity vectors evolve via Kalman-like filter:
$$u_{c,t} = \rho \cdot u_{c,t-1} + (1-\rho) \cdot f(\Delta_{c,t}) + \xi_t$$

Where:
- $\rho$: Memory decay coefficient
- $f(\cdot)$: Multi-layer perceptron
- $\Delta_{c,t}$: Performance surprise from previous week
- $\xi_t$: Random noise

---

## Key Metrics

| Metric | Value |
|--------|-------|
| Overall Accuracy | 0.53% |
| Total Week-Eliminations Evaluated | 374 |
| Correct Predictions | 2 |
| Seasons Covered | 34 |
| Total Contestants | 540 |
| Total (Season, Week, Contestant) Combinations | 4,631 |

---

## Feature Statistics

| Feature | Mean | Std | Min | Max |
|---------|------|-----|-----|-----|
| Judge Score Z-Score | 0.000 | 0.913 | -2.67 | 3.21 |
| Performance Trend | -0.234 | 0.381 | -1.23 | 0.89 |

---

## Next Steps for Model Improvement

1. **EM Algorithm Enhancement**: Implement proper gradient-based M-step optimization
2. **Parameter Tuning**: Grid search or Bayesian optimization for coefficients
3. **Heteroscedasticity Modeling**: Time-varying variance $\sigma_{i,t}^2 = \sigma_0^2 e^{-\lambda t} + \sigma_\infty^2$
4. **Celebrity Dynamics**: Full integration of Kalman filter with vote generation
5. **Uncertainty Quantification**: Posterior confidence intervals and entropy measures
6. **Rank Correlation Analysis**: Kendall's ¦Ó and Spearman's ¦Ñ for ranking consistency

---

## Data Source

- **File**: 2026_MCM_Problem_C_Data.csv
- **Records**: 18,524 judge score entries
- **Seasons**: 1-34
- **Maximum Weeks per Season**: 11
- **Judges per Week**: 4 (usually)

---

## Model Implementation

- **Framework**: Dynamic Latent Factor Model (DLFM)
- **Language**: Python 3
- **Key Libraries**: NumPy, Pandas, Matplotlib, SciPy
- **Runtime**: ~30-40 seconds for full pipeline
- **Output Directory**: `result/`

---

**Generated**: January 31, 2026
**Model Version**: 1.0 (Dynamic Latent Factor with Generative Voting)
