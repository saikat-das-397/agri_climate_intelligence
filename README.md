# AgroClimate Intelligence Platform
### Global Agriculture & Climate Resilience Analytics
**Powered by Python, Dash & Plotly • Dataset: `climate_change_impact_on_agriculture_2024.csv` (10,000 Records)**

---

## 🌟 Overview
An interactive analytical platform designed to evaluate and model the impact of climate change on global agriculture, crop yields, and economic resilience.

---

## 🌾 Module & Workflow Structure

### 1. **Crop Baselines & Climate Spread** *(Descriptive Analytics & EDA)*
- **Benchmark Yields & Central Trends**: Evaluates central yield metrics (Mean, Median, Mode) across diverse crop distributions with outlier diagnostics.
- **Climate Dispersion & Anomaly Bands**: Analyzes temperature and precipitation dispersion, standard deviation spreads, and the **Empirical Rule (68-95-99.7%)**.
- **Global Farm Distributions**: Histograms with Kernel Density Estimation (KDE), crop yield comparisons with error bars, and country-level breakdowns.

### 2. **Climate Risk & Weather Shock Modeling** *(Probability & Distributions)*
- **Drought Early Warning & Bayes' System**: Interactive Bayesian updating model calculating posterior drought risk given sensor trigger alerts.
- **Precipitation & Shock Distribution Sandbox**: Models rainfall patterns, heatwave anomalies, and extreme weather occurrences using **Binomial**, **Poisson**, **Normal**, **Exponential**, and **Uniform** distributions.
- **Expected Loss Convergence & Law of Large Numbers (LLN)**: Simulates trial-based convergence of long-term economic damage expectations.

### 3. **Yield Impact & Hypothesis Lab** *(Inferential Analytics)*
- **Regional Farm Sampling (CLT)**: Central Limit Theorem simulation demonstrating Gaussian normality from skewed meteorological parent populations.
- **Yield Benchmark Confidence Intervals**: 100-sample replication simulation verifying parameter capture rates across 90%, 95%, and 99% confidence levels.
- **Climate Shift Significance Testing**: One-sample and two-sample t-tests and z-tests evaluating agricultural yield shifts under climate stress.
- **Adaptation Independence & Risk Analysis**: Chi-Square test of independence examining associations between adaptation adoption and yield resilience.

### 4. **Agro-Meteorological Visual Intelligence** *(Data Visualization)*
- **Data Science Graphics Ecosystem**: Procedural vs. statistical vs. reactive analytical engines (Matplotlib, Seaborn, Plotly/Dash).
- **Multidimensional Agro-Climatic Gallery**: Boxplots with violin overlays, multi-variable correlation heatmaps, 3D temperature-precipitation-yield scatters, and density contours.
- **Intelligent Chart Selection Guide**: Dynamic recommendation engine guiding visual analytics by variable types and research objectives.

### 5. **Global Adaptation & Policy Synthesis** *(Capstone Empirical Study)*
- **Part 1 — Global Farm Profiling & Filtering**: Sub-region filtering, Tukey's IQR outlier trimming on `Crop_Yield_MT_per_HA`, and an 8-variable parametric/non-parametric summary table.
- **Part 2 — Multivariate Climate Impact Regression**: Ordinary Least Squares (**OLS**) multi-predictor regression ($R^2$, F-statistic p-value, t-statistics, 95% CIs) and One-Way **ANOVA** across adaptation strategies.
- **Part 3 — Actionable Policy & Executive Report**: Core findings on climate resilience, precision irrigation mitigation, and a downloadable **Executive Capstone Report (.txt)**.

---

## 🚀 How to Run the Application

```bash
cd /Users/saikatdas/agriculture_climate_change
python3 app.py
```

Open your browser at:
👉 **`http://127.0.0.1:8050`**
