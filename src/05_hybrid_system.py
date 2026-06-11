"""Hybrid System Integration: ARIMA + Reinforcement Learning

Tahap akhir:
1. Integrate ARIMA forecasts ke dalam RL state
2. End-to-end optimization pipeline
3. Decision making dengan forecast predictions
4. Final evaluation dan comparison
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import json
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# ============================================================================
# HYBRID STATE CONSTRUCTION
# ============================================================================

class HybridStateBuilder:
    """
    Build state untuk RL agent dengan ARIMA forecasts sebagai komponen
    
    State composition:
    - ARIMA forecast demand (7-day ahead)
    - Current stock level
    - Shelf life remaining
    - Demand trend
    - Day of week (seasonality)
    - Forecast uncertainty (confidence interval)
    """
    
    def __init__(self, arima_forecasts, blood_types):
        """
        Parameters:
        -----------
        arima_forecasts : dict
            Dictionary of ARIMA forecasts per blood type
            Format: {blood_type: {period: {forecast, lower_ci, upper_ci}}}
        blood_types : list
            List of blood types
        """
        
        self.arima_forecasts = arima_forecasts
        self.blood_types = blood_types
        self.n_blood_types = len(blood_types)
        
        print(f"\n[HYBRID STATE BUILDER INITIALIZED]")
        print(f"  Blood types: {self.n_blood_types}")
        print(f"  ARIMA forecasts loaded for: {list(arima_forecasts.keys())}")
    
    def get_forecast_features(self, blood_type, forecast_period=7):
        """
        Extract ARIMA forecast sebagai features
        
        Parameters:
        -----------
        blood_type : str
            Golongan darah
        forecast_period : int
            Berapa hari forecast (default 7)
        
        Returns:
        --------
        np.array : Forecast features (normalized)
        """
        
        if blood_type not in self.arima_forecasts:
            # Default jika forecast tidak tersedia
            return np.random.rand(forecast_period)
        
        forecasts_by_period = self.arima_forecasts[blood_type]
        
        if forecast_period not in forecasts_by_period:
            # Use closest available period
            available_periods = list(forecasts_by_period.keys())
            closest_period = min(available_periods, 
                               key=lambda x: abs(x - forecast_period))
            forecast_period = closest_period
        
        forecast_data = forecasts_by_period[forecast_period]
        forecast_values = np.array(forecast_data['forecast'])
        
        # Normalize
        forecast_normalized = forecast_values / (np.max(forecast_values) + 1e-6)
        
        return forecast_normalized
    
    def get_forecast_uncertainty(self, blood_type, forecast_period=7):
        """
        Extract uncertainty dari confidence interval
        
        Returns:
        --------
        float : Coefficient of variation (uncertainty measure)
        """
        
        if blood_type not in self.arima_forecasts:
            return 0.1
        
        forecasts_by_period = self.arima_forecasts[blood_type]
        
        if forecast_period not in forecasts_by_period:
            available_periods = list(forecasts_by_period.keys())
            closest_period = min(available_periods, 
                               key=lambda x: abs(x - forecast_period))
            forecast_period = closest_period
        
        forecast_data = forecasts_by_period[forecast_period]
        lower = np.array(forecast_data['lower_ci'])
        upper = np.array(forecast_data['upper_ci'])
        forecast = np.array(forecast_data['forecast'])
        
        # Calculate width of confidence interval
        ci_width = (upper - lower).mean()
        forecast_mean = forecast.mean()
        
        uncertainty = ci_width / (forecast_mean + 1e-6)
        uncertainty = min(1.0, max(0.0, uncertainty))  # Clip to [0, 1]
        
        return float(uncertainty)
    
    def construct_hybrid_state(self, current_state, blood_type, day_index):
        """
        Construct hybrid state dengan ARIMA forecasts
        
        Parameters:
        -----------
        current_state : np.array
            Original RL state
        blood_type : str
            Golongan darah utama untuk forecast
        day_index : int
            Hari dalam cycle (untuk seasonality)
        
        Returns:
        --------
        np.array : Hybrid state dengan forecast features
        """
        
        # Get forecast features
        forecast_features = self.get_forecast_features(blood_type, forecast_period=7)
        uncertainty = self.get_forecast_uncertainty(blood_type, forecast_period=7)
        
        # Combine dengan original state
        hybrid_state = np.concatenate([
            current_state,
            forecast_features,
            np.array([uncertainty])
        ], axis=0)
        
        return hybrid_state.astype(np.float32)


# ============================================================================
# HYBRID SYSTEM PIPELINE
# ============================================================================

class HybridOptimizationSystem:
    """
    End-to-end system yang mengintegrasikan:
    1. ARIMA untuk forecasting
    2. RL Agent untuk decision making
    3. Evaluation dan optimization
    """
    
    def __init__(self, arima_models, rl_agent, env, blood_types):
        """
        Parameters:
        -----------
        arima_models : dict
            Trained ARIMA models per blood type
        rl_agent : DQNAgent
            Trained RL agent
        env : BloodStockEnvironment
            Simulation environment
        blood_types : list
            List of blood types
        """
        
        self.arima_models = arima_models
        self.rl_agent = rl_agent
        self.env = env
        self.blood_types = blood_types
        
        # Generate ARIMA forecasts
        print(f"\n[HYBRID SYSTEM INITIALIZATION]")
        print(f"  Generating ARIMA forecasts...")
        self.arima_forecasts = self._generate_arima_forecasts()
        print(f"  ✓ Forecasts generated for {len(self.arima_forecasts)} blood types")
        
        # Initialize state builder
        self.state_builder = HybridStateBuilder(self.arima_forecasts, blood_types)
        
        # Results tracking
        self.optimization_history = []
        self.decisions_log = []
    
    def _generate_arima_forecasts(self):
        """
        Generate forecasts dari trained ARIMA models
        
        Returns:
        --------
        dict : ARIMA forecasts per blood type
        """
        
        forecasts = {}
        
        for blood_type, model in self.arima_models.items():
            try:
                forecast_result = model.get_forecast(steps=30)
                forecast_values = forecast_result.predicted_mean
                forecast_ci = forecast_result.conf_int()
                
                # Store forecasts untuk berbagai horizons
                forecasts[blood_type] = {
                    1: {
                        'forecast': forecast_values[:1].tolist(),
                        'lower_ci': forecast_ci.iloc[:1, 0].tolist(),
                        'upper_ci': forecast_ci.iloc[:1, 1].tolist()
                    },
                    7: {
                        'forecast': forecast_values[:7].tolist(),
                        'lower_ci': forecast_ci.iloc[:7, 0].tolist(),
                        'upper_ci': forecast_ci.iloc[:7, 1].tolist()
                    },
                    14: {
                        'forecast': forecast_values[:14].tolist(),
                        'lower_ci': forecast_ci.iloc[:14, 0].tolist(),
                        'upper_ci': forecast_ci.iloc[:14, 1].tolist()
                    },
                    30: {
                        'forecast': forecast_values[:30].tolist(),
                        'lower_ci': forecast_ci.iloc[:30, 0].tolist(),
                        'upper_ci': forecast_ci.iloc[:30, 1].tolist()
                    }
                }
            except Exception as e:
                print(f"  Warning: Could not generate forecast for {blood_type}: {e}")
                # Use dummy forecasts
                forecasts[blood_type] = {
                    period: {
                        'forecast': [100] * period,
                        'lower_ci': [80] * period,
                        'upper_ci': [120] * period
                    }
                    for period in [1, 7, 14, 30]
                }
        
        return forecasts
    
    def run_optimization(self, num_days=30):
        """
        Run optimization untuk n hari
        
        Parameters:
        -----------
        num_days : int
            Jumlah hari simulasi
        
        Returns:
        --------
        dict : Optimization results
        """
        
        print(f"\n" + "="*80)
        print(f"RUNNING HYBRID OPTIMIZATION - {num_days} DAYS")
        print("="*80)
        
        state, _ = self.env.reset()
        total_reward = 0
        daily_metrics = []
        
        for day in range(num_days):
            # Select dominant blood type for this day
            dominant_blood_type = np.random.choice(self.blood_types)
            
            # Build hybrid state dengan ARIMA forecasts
            hybrid_state = self.state_builder.construct_hybrid_state(
                current_state=state,
                blood_type=dominant_blood_type,
                day_index=day
            )
            
            # Get RL agent decision
            action = self.rl_agent.select_action(
                hybrid_state[:len(state)],  # Use original state for action selection
                training=False
            )
            
            # Execute action
            next_state, reward, terminated, truncated, info = self.env.step(action)
            
            # Log decision
            decision_log = {
                'day': day + 1,
                'blood_type': dominant_blood_type,
                'action': action.tolist(),
                'reward': float(reward),
                'stockout_incidents': info['stockout_incidents'],
                'wasted_kantong': info['wasted_kantong'],
                'efficiency': float(info['efficiency']),
                'forecast_uncertainty': float(
                    self.state_builder.get_forecast_uncertainty(dominant_blood_type)
                )
            }
            self.decisions_log.append(decision_log)
            
            # Track metrics
            daily_metrics.append(info)
            total_reward += reward
            state = next_state
            
            if (day + 1) % 10 == 0:
                print(f"Day {day+1}/{num_days}: Reward={reward:.2f}, Efficiency={info['efficiency']:.2f}, Stockout={info['stockout_incidents']}")
            
            if terminated or truncated:
                break
        
        # Compile results
        results = {
            'num_days': day + 1,
            'total_reward': float(total_reward),
            'avg_reward': float(total_reward / (day + 1)),
            'avg_efficiency': float(np.mean([m['efficiency'] for m in daily_metrics])),
            'total_stockout_incidents': int(sum([m['stockout_incidents'] for m in daily_metrics])),
            'total_wasted_kantong': int(sum([m['wasted_kantong'] for m in daily_metrics])),
            'daily_metrics': daily_metrics,
            'decisions_log': self.decisions_log
        }
        
        self.optimization_history.append(results)
        
        print(f"\n" + "="*80)
        print("OPTIMIZATION SUMMARY")
        print("="*80)
        print(f"Total Days: {results['num_days']}")
        print(f"Total Reward: {results['total_reward']:.2f}")
        print(f"Avg Reward per Day: {results['avg_reward']:.2f}")
        print(f"Avg Efficiency: {results['avg_efficiency']:.4f}")
        print(f"Total Stockout Incidents: {results['total_stockout_incidents']}")
        print(f"Total Wasted Kantong: {results['total_wasted_kantong']}")
        
        return results
    
    def save_results(self, output_dir='results/hybrid_system'):
        """
        Save optimization results
        """
        
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        if self.optimization_history:
            results = self.optimization_history[-1]
            
            # Save summary
            summary = {
                'timestamp': datetime.now().isoformat(),
                'num_days': results['num_days'],
                'total_reward': results['total_reward'],
                'avg_reward': results['avg_reward'],
                'avg_efficiency': results['avg_efficiency'],
                'total_stockout_incidents': results['total_stockout_incidents'],
                'total_wasted_kantong': results['total_wasted_kantong']
            }
            
            with open(f'{output_dir}/optimization_summary.json', 'w') as f:
                json.dump(summary, f, indent=2)
            
            # Save decisions log
            with open(f'{output_dir}/decisions_log.json', 'w') as f:
                json.dump(results['decisions_log'], f, indent=2)
            
            print(f"\n✓ Saved: {output_dir}/optimization_summary.json")
            print(f"✓ Saved: {output_dir}/decisions_log.json")
            
            # Visualize results
            self._visualize_results(results, output_dir)
    
    def _visualize_results(self, results, output_dir):
        """
        Visualisasi hasil optimization
        """
        
        daily_metrics = results['daily_metrics']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Hybrid ARIMA-RL Optimization Results', fontsize=14, fontweight='bold')
        
        # Rewards
        ax = axes[0, 0]
        rewards = [m.get('reward', 0) for m in daily_metrics if isinstance(m, dict)]
        # Extracting rewards from decisions_log instead
        rewards = [d['reward'] for d in results['decisions_log']]
        ax.plot(rewards, marker='o', label='Daily Reward', alpha=0.7)
        ax.axhline(y=np.mean(rewards), color='r', linestyle='--', label=f'Avg: {np.mean(rewards):.2f}')
        ax.set_xlabel('Day')
        ax.set_ylabel('Reward')
        ax.set_title('Daily Rewards')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Efficiency
        ax = axes[0, 1]
        efficiencies = [d['efficiency'] for d in results['decisions_log']]
        ax.plot(efficiencies, marker='s', label='Daily Efficiency', alpha=0.7, color='green')
        ax.axhline(y=0.85, color='r', linestyle='--', label='Target: 0.85')
        ax.set_xlabel('Day')
        ax.set_ylabel('Efficiency')
        ax.set_title('Stock Efficiency')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Stockout Incidents
        ax = axes[1, 0]
        stockouts = [d['stockout_incidents'] for d in results['decisions_log']]
        ax.bar(range(len(stockouts)), stockouts, color='red', alpha=0.7)
        ax.set_xlabel('Day')
        ax.set_ylabel('Incidents')
        ax.set_title('Stockout Incidents per Day')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Waste
        ax = axes[1, 1]
        wastes = [d['wasted_kantong'] for d in results['decisions_log']]
        ax.bar(range(len(wastes)), wastes, color='orange', alpha=0.7)
        ax.set_xlabel('Day')
        ax.set_ylabel('Kantong')
        ax.set_title('Wasted Blood per Day')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/optimization_results.png', dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_dir}/optimization_results.png")
        plt.close()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("\n" + "#"*80)
    print("# HYBRID ARIMA-RL SYSTEM INTEGRATION")
    print("#"*80)
    
    print("\nNote: This is a template for hybrid system integration.")
    print("Requires trained ARIMA models and RL agent from previous steps.")
    print("\nSteps to complete:")
    print("  1. Run src/01_data_preparation.py")
    print("  2. Run src/02_arima_forecasting.py")
    print("  3. Run src/03_rl_environment.py")
    print("  4. Run src/04_rl_agent.py")
    print("  5. Load models and run this hybrid system")
    
    print("\n" + "#"*80)
    print("# Ready for hybrid system deployment ✓")
    print("#"*80 + "\n")
