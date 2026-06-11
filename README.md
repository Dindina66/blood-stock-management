# Blood Demand Forecasting with ARIMA and Reinforcement Learning

## Kerangka Solusi

```
Permasalahan Ketidakpastian Permintaan Darah
                    ↓
        Data Historis Permintaan Darah
                    ↓
               Model ARIMA
                    ↓
          Forecast Kebutuhan Darah
                    ↓
         State Reinforcement Learning
                    ↓
                Agent RL
                    ↓
         Keputusan Pengelolaan Stok
                    ↓
            Optimasi Stok Darah
```

## Project Overview

Sistem prediksi dan optimasi kebutuhan darah untuk PMI Jakarta Barat menggunakan hybrid approach:
- **ARIMA**: Time series forecasting untuk prediksi demand jangka pendek
- **Reinforcement Learning**: Decision making untuk pengelolaan stok optimal

### Tujuan
- Mengurangi waste rate dari 12% menjadi <5%
- Mengurangi stockout incidents
- Meningkatkan efisiensi ke level WHO (85%)
- Forecast accuracy RMSE < 10%

## Project Structure

```
blood-stock-management/
├── data/
│   ├── raw/                    # Data historis mentah
│   └── processed/              # Data siap training
├── src/
│   ├── 01_data_preparation.py  # EDA, cleaning, decomposition
│   ├── 02_arima_forecasting.py # ARIMA model training & forecast
│   ├── 03_rl_environment.py    # RL environment setup
│   ├── 04_rl_agent.py          # RL agent training
│   └── 05_hybrid_system.py     # Integration ARIMA + RL
├── notebooks/                   # Jupyter notebooks
├── results/                      # Models, predictions, metrics
├── requirements.txt
└── README.md
```

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Data Preparation
```bash
python src/01_data_preparation.py
```
Output: train/test split, EDA visualizations, time series decomposition

### 2. ARIMA Forecasting
```bash
python src/02_arima_forecasting.py
```
Output: Trained ARIMA models, forecast, evaluation metrics

### 3. RL Environment Setup
```bash
python src/03_rl_environment.py
```
Output: Gymnasium environment with state/action/reward design

### 4. RL Agent Training
```bash
python src/04_rl_agent.py
```
Output: Trained RL agent, learning curves

### 5. Hybrid System Integration
```bash
python src/05_hybrid_system.py
```
Output: Final optimization policy

## Key Technologies

- **Time Series**: Statsmodels (ARIMA)
- **RL**: Gymnasium + Stable-Baselines3
- **ML**: Scikit-learn, TensorFlow/PyTorch
- **Data**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn, Plotly

## Status

🚀 Development Phase
- [x] Project structure
- [ ] Data preparation
- [ ] ARIMA model
- [ ] RL environment
- [ ] RL agent
- [ ] Hybrid integration
- [ ] Evaluation & deployment

---

**Last Updated**: 2026-06-11
