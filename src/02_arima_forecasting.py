"""ARIMA Forecasting untuk Blood Demand Prediction

Tahap ini meliputi:
1. Identifikasi parameter ARIMA (p, d, q)
2. Training ARIMA model per golongan darah
3. Forecast kebutuhan darah
4. Evaluasi model dengan MAE & RMSE
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
import json

from sklearn.metrics import mean_absolute_error, mean_squared_error

# ============================================================================
# STEP 1: Stationarity Testing (ADF Test)
# ============================================================================

def test_stationarity(timeseries, blood_type, output_dir='results/stationarity'):
    """
    Perform Augmented Dickey-Fuller test untuk memeriksa stationarity
    
    Parameters:
    -----------
    timeseries : pd.Series
        Time series data
    blood_type : str
        Golongan darah
    output_dir : str
        Direktori untuk simpan hasil
    
    Returns:
    --------
    dict : Test results
    """
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # ADF Test
    result = adfuller(timeseries, autolag='AIC')
    
    test_results = {
        'blood_type': blood_type,
        'ADF_statistic': float(result[0]),
        'p_value': float(result[1]),
        'n_lags_used': int(result[2]),
        'n_observations': int(result[3]),
        'critical_values': {k: float(v) for k, v in result[4].items()},
        'is_stationary': result[1] < 0.05  # Reject H0 jika p < 0.05
    }
    
    return test_results


# ============================================================================
# STEP 2: ACF/PACF Analysis untuk Parameter Selection
# ============================================================================

def analyze_acf_pacf(timeseries, blood_type, output_dir='results/acf_pacf'):
    """
    Visualisasi ACF dan PACF untuk identifikasi parameter p, d, q
    
    Parameters:
    -----------
    timeseries : pd.Series
        Time series data
    blood_type : str
        Golongan darah
    output_dir : str
        Direktori untuk simpan hasil
    """
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'ACF & PACF Analysis - {blood_type}', fontsize=14, fontweight='bold')
    
    # ACF
    plot_acf(timeseries, lags=40, ax=axes[0])
    axes[0].set_title('Autocorrelation Function (ACF)')
    
    # PACF
    plot_pacf(timeseries, lags=40, ax=axes[1], method='ywm')
    axes[1].set_title('Partial Autocorrelation Function (PACF)')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/acf_pacf_{blood_type}.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved ACF/PACF: {output_dir}/acf_pacf_{blood_type}.png")
    plt.close()


# ============================================================================
# STEP 3: ARIMA Parameter Grid Search
# ============================================================================

def find_optimal_arima_params(train_data, blood_type, 
                              p_range=range(0, 6), 
                              d_range=range(0, 3), 
                              q_range=range(0, 6)):
    """
    Grid search untuk menemukan parameter ARIMA optimal
    Berdasarkan AIC score
    
    Parameters:
    -----------
    train_data : pd.Series
        Training time series
    blood_type : str
        Golongan darah
    p_range : range
        Range untuk parameter p (AR)
    d_range : range
        Range untuk parameter d (differencing)
    q_range : range
        Range untuk parameter q (MA)
    
    Returns:
    --------
    dict : Optimal parameters dan metrics
    """
    
    print(f"\n  Searching optimal ARIMA parameters for {blood_type}...")
    
    best_aic = np.inf
    best_params = None
    
    for p in p_range:
        for d in d_range:
            for q in q_range:
                try:
                    model = ARIMA(train_data, order=(p, d, q))
                    fitted_model = model.fit()
                    
                    if fitted_model.aic < best_aic:
                        best_aic = fitted_model.aic
                        best_params = (p, d, q)
                        best_model = fitted_model
                
                except:
                    continue
    
    print(f"    ✓ Best params: ARIMA{best_params}, AIC={best_aic:.2f}")
    
    return {
        'blood_type': blood_type,
        'optimal_params': best_params,
        'aic': float(best_aic),
        'model': best_model
    }


# ============================================================================
# STEP 4: Train ARIMA Model
# ============================================================================

def train_arima_models(train_df, blood_types, output_dir='results/arima_models'):
    """
    Train ARIMA model untuk setiap golongan darah
    
    Parameters:
    -----------
    train_df : pd.DataFrame
        Training data
    blood_types : list
        Daftar golongan darah
    output_dir : str
        Direktori untuk simpan models
    
    Returns:
    --------
    dict : Trained models dan parameters per blood type
    """
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*80)
    print("TRAINING ARIMA MODELS")
    print("="*80)
    
    models_dict = {}
    params_dict = {}
    stationarity_results = {}
    
    for blood_type in blood_types:
        print(f"\n[{blood_type}]")
        
        # Get time series for this blood type
        ts = train_df[train_df['blood_type'] == blood_type].set_index('date')['demand']
        ts = ts.sort_index()
        
        # Test stationarity
        stat_result = test_stationarity(ts, blood_type)
        stationarity_results[blood_type] = stat_result
        
        if stat_result['is_stationary']:
            print(f"  ✓ Time series is stationary (p={stat_result['p_value']:.4f})")
            d_optimal = 0
        else:
            print(f"  ✗ Time series is NOT stationary (p={stat_result['p_value']:.4f})")
            print(f"    Will use d=1 for differencing")
            d_optimal = 1
        
        # Analyze ACF/PACF
        analyze_acf_pacf(ts, blood_type)
        
        # Find optimal parameters (with d constraint based on stationarity test)
        d_range = range(d_optimal, min(d_optimal + 2, 3))
        optimal_result = find_optimal_arima_params(ts, blood_type, 
                                                   p_range=range(0, 6),
                                                   d_range=d_range,
                                                   q_range=range(0, 6))
        
        models_dict[blood_type] = optimal_result['model']
        params_dict[blood_type] = optimal_result['optimal_params']
        
        # Save model summary
        print(f"\n  Model Summary:")
        print(f"    Parameters: ARIMA{optimal_result['optimal_params']}")
        print(f"    AIC: {optimal_result['aic']:.2f}")
    
    # Save parameters to JSON
    params_json = {k: list(v) for k, v in params_dict.items()}
    with open(f'{output_dir}/optimal_parameters.json', 'w') as f:
        json.dump(params_json, f, indent=2)
    
    print(f"\n✓ Saved: {output_dir}/optimal_parameters.json")
    
    return models_dict, params_dict


# ============================================================================
# STEP 5: Forecast
# ============================================================================

def forecast_blood_demand(models_dict, test_df, forecast_periods=[1, 7, 14, 30]):
    """
    Generate forecast menggunakan trained ARIMA models
    
    Parameters:
    -----------
    models_dict : dict
        Dictionary of trained ARIMA models per blood type
    test_df : pd.DataFrame
        Test data
    forecast_periods : list
        Periode forecast dalam hari [1, 7, 14, 30]
    
    Returns:
    --------
    dict : Forecasts per blood type dan periode
    """
    
    print("\n" + "="*80)
    print("GENERATING FORECASTS")
    print("="*80)
    
    all_forecasts = {}
    
    for blood_type, model in models_dict.items():
        print(f"\n[{blood_type}] Forecasting...")
        
        forecasts_by_period = {}
        
        for period in forecast_periods:
            # Forecast
            forecast_result = model.get_forecast(steps=period)
            forecast_values = forecast_result.predicted_mean
            forecast_ci = forecast_result.conf_int()
            
            forecasts_by_period[period] = {
                'forecast': forecast_values.tolist(),
                'lower_ci': forecast_ci.iloc[:, 0].tolist(),
                'upper_ci': forecast_ci.iloc[:, 1].tolist()
            }
            
            print(f"  ✓ {period}-day forecast: {forecast_values.mean():.2f} ± {(forecast_ci.iloc[:, 1] - forecast_ci.iloc[:, 0]).mean():.2f}")
        
        all_forecasts[blood_type] = forecasts_by_period
    
    return all_forecasts


# ============================================================================
# STEP 6: Evaluation
# ============================================================================

def evaluate_arima_models(models_dict, test_df, output_dir='results/arima_evaluation'):
    """
    Evaluate model performance pada test set
    Metrics: MAE, RMSE, MAPE
    
    Parameters:
    -----------
    models_dict : dict
        Dictionary of trained ARIMA models
    test_df : pd.DataFrame
        Test data
    output_dir : str
        Direktori untuk simpan hasil evaluasi
    
    Returns:
    --------
    dict : Evaluation metrics
    """
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*80)
    print("MODEL EVALUATION")
    print("="*80)
    
    evaluation_results = {}
    
    for blood_type, model in models_dict.items():
        # Get test data
        test_data = test_df[test_df['blood_type'] == blood_type].set_index('date')['demand']
        test_data = test_data.sort_index()
        
        # Make predictions
        predictions = model.get_forecast(steps=len(test_data))
        pred_values = predictions.predicted_mean.values
        
        # Ensure same length
        if len(pred_values) > len(test_data):
            pred_values = pred_values[:len(test_data)]
        elif len(pred_values) < len(test_data):
            test_data = test_data[:len(pred_values)]
        
        # Calculate metrics
        mae = mean_absolute_error(test_data.values, pred_values)
        rmse = np.sqrt(mean_squared_error(test_data.values, pred_values))
        mape = np.mean(np.abs((test_data.values - pred_values) / test_data.values)) * 100
        
        evaluation_results[blood_type] = {
            'MAE': float(mae),
            'RMSE': float(rmse),
            'MAPE': float(mape),
            'predictions': pred_values.tolist(),
            'actual': test_data.values.tolist()
        }
        
        print(f"\n[{blood_type}]")
        print(f"  MAE:  {mae:.2f}")
        print(f"  RMSE: {rmse:.2f}")
        print(f"  MAPE: {mape:.2f}%")
    
    # Visualization
    fig, axes = plt.subplots(2, 4, figsize=(18, 10))
    fig.suptitle('ARIMA Model Evaluation - Actual vs Predicted', fontsize=14, fontweight='bold')
    
    blood_types = list(evaluation_results.keys())
    for idx, blood_type in enumerate(blood_types):
        row = idx // 4
        col = idx % 4
        ax = axes[row, col]
        
        results = evaluation_results[blood_type]
        ax.plot(results['actual'], label='Actual', marker='o', alpha=0.7)
        ax.plot(results['predictions'], label='Predicted', marker='s', alpha=0.7)
        ax.set_title(f"{blood_type} (RMSE: {results['RMSE']:.2f})")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylabel('Demand')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/evaluation_actual_vs_predicted.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved: {output_dir}/evaluation_actual_vs_predicted.png")
    plt.close()
    
    # Save evaluation results
    with open(f'{output_dir}/evaluation_metrics.json', 'w') as f:
        json.dump(evaluation_results, f, indent=2)
    
    print(f"✓ Saved: {output_dir}/evaluation_metrics.json")
    
    return evaluation_results


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("\n" + "#"*80)
    print("# ARIMA FORECASTING FOR BLOOD DEMAND PREDICTION")
    print("#"*80)
    
    # Load prepared data
    print("\n[STEP 1] Loading prepared data...")
    train_df = pd.read_csv('data/processed/train.csv')
    test_df = pd.read_csv('data/processed/test.csv')
    train_df['date'] = pd.to_datetime(train_df['date'])
    test_df['date'] = pd.to_datetime(test_df['date'])
    
    blood_types = train_df['blood_type'].unique()
    print(f"✓ Data loaded: {len(blood_types)} blood types")
    print(f"  Blood types: {', '.join(sorted(blood_types))}")
    
    # Step 2: Train ARIMA models
    print("\n[STEP 2] Training ARIMA models...")
    models_dict, params_dict = train_arima_models(train_df, blood_types)
    print(f"✓ {len(models_dict)} ARIMA models trained")
    
    # Step 3: Forecast
    print("\n[STEP 3] Generating forecasts...")
    forecasts = forecast_blood_demand(models_dict, test_df)
    print(f"✓ Forecasts generated for {len(forecasts)} blood types")
    
    # Step 4: Evaluate
    print("\n[STEP 4] Evaluating models...")
    evaluation = evaluate_arima_models(models_dict, test_df)
    print(f"✓ Models evaluated")
    
    print("\n" + "#"*80)
    print("# ARIMA FORECASTING COMPLETED ✓")
    print("#"*80)
    print(f"\nNext step: Run 'python src/03_rl_environment.py'")
    print("#"*80 + "\n")
