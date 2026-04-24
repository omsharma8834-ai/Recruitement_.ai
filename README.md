# Bangladesh Power Demand Forecasting
**Predictive Paradox — End-to-End Machine Learning Pipeline**

A complete regression pipeline to forecast hourly electricity demand in Bangladesh using historical power generation records, weather observations, and macroeconomic indicators. The final tuned model achieves a MAPE of **1.85%** on a held-out chronological test set with no overfitting.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset Description](#dataset-description)
- [Project Structure](#project-structure)
- [Pipeline Summary](#pipeline-summary)
- [Feature Engineering](#feature-engineering)
- [Models Evaluated](#models-evaluated)
- [Results](#results)
- [How to Run](#how-to-run)
- [How to Load and Use the Saved Model](#how-to-load-and-use-the-saved-model)
- [Dependencies](#dependencies)

---

## Project Overview

**Task:** Predict hourly power demand (`demand_mw`) in Bangladesh using weather conditions, time-based patterns, and economic context.

**Problem type:** Supervised regression on time series data

**Target variable:** `demand_mw` — hourly electricity demand in megawatts, sourced from the Power Grid Company of Bangladesh (PGCB)

**Key insight:** Autoregressive lag features (demand at t-1h, t-24h, t-168h) are the dominant predictors for this task. Models without these features suffer MAPE in the range of 10–15%. Including them drops MAPE to **1.85%**.

---

## Dataset Description

Three source files are required in the working directory:

| File | Description | Rows | Frequency |
|------|-------------|------|-----------|
| `PGCB_date_power_demand.xlsx` | Hourly power demand, generation, and fuel mix data from PGCB | ~92,650 | Hourly |
| `weather_data.xlsx` | Hourly weather observations for Dhaka (Open-Meteo) | ~107,304 | Hourly |
| `economic_full_1.csv` | World Bank macroeconomic indicators for Bangladesh | ~1,516 indicators | Yearly |

**Date coverage:** 2015 – 2025

### Power Demand Columns (key fields)

| Column | Description |
|--------|-------------|
| `datetime` | Timestamp of the observation |
| `demand_mw` | Total electricity demand (MW) — **target variable** |
| `generation_mw` | Total power generated (MW) |
| `gas`, `liquid_fuel`, `coal`, `hydro`, `solar` | Generation by fuel source (MW) |
| `load_shedding` | Load shedding value |

### Weather Columns (key fields)

| Column | Description |
|--------|-------------|
| `temperature_2m (°C)` | Air temperature at 2m height |
| `relative_humidity_2m (%)` | Relative humidity |
| `apparent_temperature (°C)` | Feels-like temperature |
| `precipitation (mm)` | Precipitation |
| `cloud_cover (%)` | Cloud coverage |
| `sunshine_duration (s)` | Sunshine duration per hour |

### Economic Indicators Used

GDP (current US$), GDP growth (annual %), GDP per capita, Population total, Urban population, Urban population (% of total), Inflation (consumer prices), Electric power consumption (kWh per capita), Access to electricity (%), Industry value added (% of GDP).

---

## Project Structure

```
.
├── PGCB_date_power_demand.xlsx          # Raw power demand data
├── weather_data.xlsx                    # Raw weather data
├── economic_full_1.csv                  # Raw economic indicators
├── Predictive_Paradox_COMPLETE.ipynb    # Full pipeline notebook
├── preprocessed_data.csv                # Generated: cleaned, merged, scaled dataset
├── best_model.pkl                       # Generated: best model from selection phase
├── final_model.pkl                      # Generated: tuned model, ready for deployment
└── README.md
```

---

## Pipeline Summary

The notebook is organized into five sections:

### Section 0 — Imports & Configuration
All library imports and global settings (random seed, plot style, warnings) in one place at the top.

### Section 1 — Exploratory Data Analysis
- Loaded and profiled all three datasets (shapes, date ranges, dtypes)
- Identified missing value patterns and classified columns for dropping vs. imputation
- Analyzed power demand: full time series, monthly averages, hourly patterns, day-of-week patterns, year-over-year growth, and energy generation mix
- Analyzed weather data: variable distributions, seasonal temperature patterns, correlation matrix
- Performed cross-dataset analysis: Pearson correlations of weather features with demand, and a demand heatmap across hour of day × month of year

**Key EDA findings:**
- Demand grows approximately 6–8% annually
- Summer months (April–September) have the highest demand due to air conditioning usage
- Daily peak hours are 18:00–22:00
- Temperature has the strongest positive correlation with demand among weather variables
- Gas accounts for approximately 60–65% of the generation mix

### Section 2 — Preprocessing & Feature Engineering

**Power data cleaning:**
- Dropped four columns with over 80% missing values: `wind`, `india_adani`, `nepal`, `remarks`
- Fixed a data entry error in `generation_mw` (one value was 10,000× too large)
- Forward-filled rows where `demand_mw` < 1,000 MW (physically impossible values caused by digit truncation)
- Imputed `solar` (24% missing) with the column median
- Winsorized `demand_mw` outliers using the IQR method

**Time feature engineering:**
- Extracted `hour`, `month`, `year`, `day_of_week`, `quarter`, `is_weekend`
- Applied cyclical (sin/cos) encoding to `hour`, `month`, and `day_of_week`
- Assigned Bangladesh-specific seasons: Summer (Mar–Jun), Monsoon (Jul–Oct), Winter (Nov–Feb)
- One-hot encoded season into three dummy columns

**Weather feature engineering:**
- Computed a heat index combining temperature and humidity into a single thermal load estimate
- Computed rolling averages: 24-hour and 72-hour temperature, 24-hour humidity

**Economic data:**
- Selected 10 demand-relevant indicators
- Reshaped from wide to long format and linearly interpolated sparse yearly values

**Lag and rolling demand features (critical for low MAPE):**

| Feature | Description |
|---------|-------------|
| `demand_lag_1h` | Demand one hour ago |
| `demand_lag_24h` | Demand same hour, previous day |
| `demand_lag_168h` | Demand same hour, previous week |
| `demand_rolling_24h_mean` | 24-hour rolling mean of past demand |
| `demand_rolling_168h_mean` | 168-hour rolling mean of past demand |
| `demand_rolling_24h_std` | 24-hour rolling standard deviation of past demand |

All rolling features are computed on `shift(1)` values to eliminate look-ahead leakage. The first 168 rows are dropped due to insufficient lag history.

**Scaling:** `MinMaxScaler` applied to all continuous features. Target, boolean flags, cyclical features, and dummies are excluded.

### Section 3 — Model Selection
Five models were trained and compared on the same chronological 80/20 split: Linear Regression, Ridge Regression, Decision Tree, Random Forest, and Gradient Boosting. Evaluation metrics: R², MAE, RMSE, MAPE.

### Section 4 — Cross-Validation & Hyperparameter Tuning
- Trained a regularized baseline Random Forest (`max_depth=12`, `min_samples_leaf=15`, `max_features=0.6`)
- Ran `TimeSeriesSplit` (5-fold) cross-validation
- Ran `GridSearchCV` over `n_estimators`, `max_depth`, `min_samples_leaf`, and `max_features`
- Evaluated tuned model and confirmed no overfitting (train/test gap = 0.0562)

### Section 5 — Summary
Single-cell summary printing all final metrics, CV scores, best hyperparameters, and the train/test gap.

---

## Feature Engineering

| Category | Features |
|----------|----------|
| Time | `hour`, `month`, `year`, `dow`, `quarter`, `is_weekend` |
| Cyclical time | `hour_sin`, `hour_cos`, `month_sin`, `month_cos`, `dow_sin`, `dow_cos` |
| Season | `season_summer`, `season_monsoon`, `season_winter` |
| Weather (raw) | `temperature_2m`, `relative_humidity_2m`, `apparent_temperature`, `precipitation`, `cloud_cover`, `sunshine_duration` |
| Weather (engineered) | `heat_index`, `temp_24h_avg`, `temp_72h_avg`, `humidity_24h_avg` |
| Demand lags | `demand_lag_1h`, `demand_lag_24h`, `demand_lag_168h` |
| Demand rolling | `demand_rolling_24h_mean`, `demand_rolling_168h_mean`, `demand_rolling_24h_std` |
| Power generation | `gas`, `liquid_fuel`, `coal`, `hydro`, `solar`, `india_bheramara_hvdc`, `india_tripura` |
| Economic | GDP, GDP growth, GDP per capita, population, urban population, inflation, electricity access, electricity consumption, industry value added |

**Total features: 48**

---

## Models Evaluated

| Model | Description |
|-------|-------------|
| Linear Regression | Ordinary least squares baseline |
| Ridge Regression | L2-regularized linear model |
| Decision Tree | Single tree with depth constraint |
| Random Forest | Ensemble of 150 trees with feature subsampling |
| Gradient Boosting | Sequential boosting with shrinkage (lr=0.05) |

**Final model:** Random Forest — hyperparameters selected by GridSearchCV with TimeSeriesSplit cross-validation.

**Best hyperparameters:**

| Parameter | Value |
|-----------|-------|
| `n_estimators` | 150 |
| `max_depth` | 12 |
| `min_samples_leaf` | 10 |
| `max_features` | 0.7 |

---

## Results

| Metric | Value |
|--------|-------|
| Dataset size | 22,127 rows |
| Features | 48 |
| Train rows | 17,701 (80%) |
| Test rows | 4,426 (20%) |
| CV Mean R2 | 0.9598 ± 0.0182 |
| Final R2 | 0.9371 |
| Final MAE | 140.9 MW |
| Final RMSE | 271.1 MW |
| **Final MAPE** | **1.85%** |
| Train/Test Gap | 0.0562 — No overfitting |

---

## How to Run

1. Place the three source files in the same directory as the notebook.
2. Install dependencies (see below).
3. Open `Predictive_Paradox_COMPLETE.ipynb` in Jupyter and run all cells in order.

   Sections 1–2 are self-contained. Section 3 saves `best_model.pkl`. Section 4 reads `preprocessed_data.csv` and `best_model.pkl`, then saves `final_model.pkl`.

4. GridSearchCV in Section 4.4 may take 5–15 minutes depending on hardware.

---

## How to Load and Use the Saved Model

```python
import pickle
import pandas as pd

with open("final_model.pkl", "rb") as f:
    saved = pickle.load(f)

model    = saved["model"]
scaler   = saved["scaler"]
features = saved["features"]

# Construct input with all 48 required feature columns
new_data = pd.DataFrame([{ ... }])

prediction = model.predict(new_data[features])
print(f"Predicted demand: {prediction[0]:.0f} MW")
```

The `features` list in the saved dictionary specifies the exact column order required.

---

## Dependencies

```
pandas
numpy
matplotlib
seaborn
scikit-learn
scipy
openpyxl
```

Install with:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy openpyxl
```

Python 3.8 or higher is recommended.
