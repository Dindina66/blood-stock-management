"""Reinforcement Learning Environment untuk Blood Stock Management

Tahap ini meliputi:
1. Design State Space (informasi yang diamati oleh agent)
2. Design Action Space (aksi yang dapat dilakukan agent)
3. Reward Function (mekanisme pembelajaran)
4. Custom Gymnasium Environment
"""

import numpy as np
import pandas as pd
from gymnasium import Env
from gymnasium.spaces import Box, Discrete, Dict
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# STATE SPACE DESIGN
# ============================================================================

class BloodStockState:
    """
    State Space untuk Blood Stock Management
    
    Komponen State:
    1. Forecast demand (h-step ahead dari ARIMA)
    2. Current stock level per blood type
    3. Shelf life remaining days
    4. Demand trend (naik/turun/stabil)
    5. Seasonality factor (hari minggu)
    """
    
    def __init__(self, blood_types=['O+', 'O-', 'A+', 'A-', 'B+', 'B-', 'AB+', 'AB-']):
        self.blood_types = blood_types
        self.n_blood_types = len(blood_types)
        
        # State dimensions
        self.forecast_horizon = 7  # 7-day forecast
        self.n_shelf_life_bins = 6  # Group shelf life: 0-7, 8-14, 15-21, 22-28, 29-35, 36-42
        
        # State size calculation
        self.state_dim = (
            self.n_blood_types * self.forecast_horizon +
            self.n_blood_types +
            self.n_blood_types * self.n_shelf_life_bins +
            self.n_blood_types +
            7 +
            1
        )
        
        print(f"\n[STATE SPACE DESIGN]")
        print(f"  Blood types: {self.n_blood_types}")
        print(f"  Forecast horizon: {self.forecast_horizon} days")
        print(f"  Total state dimension: {self.state_dim}")
    
    def create_state(self, forecast_demand, current_stock, shelf_life_remaining, 
                    demand_trend, day_of_week, seasonality_factor):
        """
        Combine components into single state vector
        """
        state_components = []
        
        # 1. Forecast demand (normalized)
        forecast_normalized = forecast_demand / (np.max(forecast_demand) + 1e-6)
        state_components.append(forecast_normalized.flatten())
        
        # 2. Current stock (normalized)
        stock_normalized = current_stock / 500.0
        state_components.append(stock_normalized)
        
        # 3. Shelf life distribution (binned)
        shelf_life_dist = np.zeros((self.n_blood_types, self.n_shelf_life_bins))
        for i, shelf_life in enumerate(shelf_life_remaining):
            bin_idx = min(int(shelf_life / 7), self.n_shelf_life_bins - 1)
            shelf_life_dist[i, bin_idx] = 1
        state_components.append(shelf_life_dist.flatten())
        
        # 4. Demand trend
        trend_normalized = (demand_trend + 1) / 2
        state_components.append(trend_normalized)
        
        # 5. Day of week (one-hot encoding)
        day_one_hot = np.zeros(7)
        day_one_hot[day_of_week] = 1
        state_components.append(day_one_hot)
        
        # 6. Seasonality factor
        seasonality_normalized = (seasonality_factor - 0.5) / 1.0
        state_components.append(np.array([seasonality_normalized]))
        
        state = np.concatenate(state_components, axis=None).astype(np.float32)
        return state


# ============================================================================
# ACTION SPACE DESIGN
# ============================================================================

class BloodStockActions:
    """
    Action Space untuk Blood Stock Management
    
    Agent dapat melakukan aksi:
    1. Procurement: Berapa banyak darah diambil dari donor (0-500 kantong per jenis)
    2. Distribution: Ke rumah sakit mana prioritas (top-k hospitals)
    """
    
    def __init__(self, blood_types=['O+', 'O-', 'A+', 'A-', 'B+', 'B-', 'AB+', 'AB-'],
                 n_hospitals=25, max_procurement=500):
        
        self.blood_types = blood_types
        self.n_blood_types = len(blood_types)
        self.n_hospitals = n_hospitals
        self.max_procurement = max_procurement
        self.action_dim = self.n_blood_types + self.n_hospitals
        
        print(f"\n[ACTION SPACE DESIGN]")
        print(f"  Blood types: {self.n_blood_types} (procurement decisions)")
        print(f"  Hospitals: {self.n_hospitals} (distribution decisions)")
        print(f"  Total action dimension: {self.action_dim}")
    
    def decode_action(self, action_vector):
        """
        Decode normalized action vector ke procurement dan distribution
        """
        procurement_normalized = action_vector[:self.n_blood_types]
        procurement = (procurement_normalized * self.max_procurement).astype(int)
        procurement_dict = {blood_type: procurement[i] 
                           for i, blood_type in enumerate(self.blood_types)}
        
        distribution_normalized = action_vector[self.n_blood_types:]
        distribution_normalized = np.maximum(distribution_normalized, 0)
        distribution_normalized = distribution_normalized / (distribution_normalized.sum() + 1e-6)
        distribution_dict = {f'hospital_{i}': distribution_normalized[i]
                            for i in range(self.n_hospitals)}
        
        return procurement_dict, distribution_dict


# ============================================================================
# REWARD FUNCTION
# ============================================================================

class BloodStockReward:
    """
    Reward Function untuk incentivize optimal stock management
    """
    
    def __init__(self, stockout_penalty=-100, waste_penalty=-0.5, 
                 cost_penalty=-0.01, efficiency_bonus=10):
        
        self.stockout_penalty = stockout_penalty
        self.waste_penalty = waste_penalty
        self.cost_penalty = cost_penalty
        self.efficiency_bonus = efficiency_bonus
        
        print(f"\n[REWARD FUNCTION DESIGN]")
        print(f"  Stockout penalty: {stockout_penalty}")
        print(f"  Waste penalty: {waste_penalty}")
        print(f"  Cost penalty: {cost_penalty}")
        print(f"  Efficiency bonus: {efficiency_bonus}")
    
    def calculate_reward(self, n_stockout, n_waste, procurement_cost, efficiency_score):
        """
        Calculate total reward
        """
        reward = 0
        reward += n_stockout * self.stockout_penalty
        reward += n_waste * self.waste_penalty
        reward += procurement_cost * self.cost_penalty
        
        if efficiency_score >= 0.85:
            reward += self.efficiency_bonus
        
        return reward


# ============================================================================
# CUSTOM GYMNASIUM ENVIRONMENT
# ============================================================================

class BloodStockEnvironment(Env):
    """
    Custom Gymnasium Environment untuk Blood Stock Management
    """
    
    def __init__(self, demand_data, forecast_model=None,
                 blood_types=['O+', 'O-', 'A+', 'A-', 'B+', 'B-', 'AB+', 'AB-'],
                 n_hospitals=25, max_stock_per_type=500, shelf_life_days=42,
                 min_stock_threshold=50, daily_collection=300):
        
        self.demand_data = demand_data
        self.forecast_model = forecast_model
        self.blood_types = blood_types
        self.n_blood_types = len(blood_types)
        self.n_hospitals = n_hospitals
        self.max_stock_per_type = max_stock_per_type
        self.shelf_life_days = shelf_life_days
        self.min_stock_threshold = min_stock_threshold
        self.daily_collection = daily_collection
        
        # Initialize components
        self.state_space = BloodStockState(blood_types)
        self.action_space = BloodStockActions(blood_types, n_hospitals)
        self.reward_fn = BloodStockReward()
        
        # Gymnasium space definitions
        self.observation_space = Box(low=-np.inf, high=np.inf, 
                                    shape=(self.state_space.state_dim,), 
                                    dtype=np.float32)
        
        self.action_space_gym = Box(low=0, high=1,
                                   shape=(self.action_space.action_dim,),
                                   dtype=np.float32)
        
        self.reset()
        
        print(f"\n[ENVIRONMENT INITIALIZED]")
        print(f"  Observation space: {self.observation_space.shape}")
        print(f"  Action space: {self.action_space_gym.shape}")
    
    def reset(self):
        """
        Reset environment ke initial state
        """
        self.current_stock = {bt: 100 for bt in self.blood_types}
        self.shelf_life_remaining = {bt: self.shelf_life_days for bt in self.blood_types}
        self.current_step = 0
        self.max_steps = 365
        self.stockout_incidents = 0
        self.wasted_kantong = 0
        self.total_procurement = 0
        
        obs = self._get_observation()
        return obs, {}
    
    def _get_observation(self):
        """
        Generate current observation (state)
        """
        forecast_demand = np.array([
            [np.random.randint(50, 150) for _ in range(7)]
            for _ in self.blood_types
        ]).astype(np.float32)
        
        current_stock = np.array([self.current_stock[bt] for bt in self.blood_types]).astype(np.float32)
        shelf_life = np.array([self.shelf_life_remaining[bt] for bt in self.blood_types]).astype(np.float32)
        trend = np.random.choice([-1, 0, 1], size=self.n_blood_types)
        day_of_week = self.current_step % 7
        seasonality = 1.2 if day_of_week < 5 else 0.8
        
        state = self.state_space.create_state(
            forecast_demand=forecast_demand,
            current_stock=current_stock,
            shelf_life_remaining=shelf_life,
            demand_trend=trend,
            day_of_week=day_of_week,
            seasonality_factor=seasonality
        )
        
        return state
    
    def step(self, action):
        """
        Execute one step in environment
        """
        procurement_dict, distribution_dict = self.action_space.decode_action(action)
        
        for bt in self.blood_types:
            self.current_stock[bt] += procurement_dict[bt]
            self.current_stock[bt] = min(self.current_stock[bt], self.max_stock_per_type)
            self.total_procurement += procurement_dict[bt]
        
        daily_demand = {bt: np.random.randint(50, 150) for bt in self.blood_types}
        
        for bt in self.blood_types:
            if self.current_stock[bt] < daily_demand[bt]:
                self.stockout_incidents += 1
            else:
                self.current_stock[bt] -= daily_demand[bt]
        
        for bt in self.blood_types:
            self.shelf_life_remaining[bt] -= 1
            if self.shelf_life_remaining[bt] <= 0:
                self.wasted_kantong += self.current_stock[bt]
                self.current_stock[bt] = 0
                self.shelf_life_remaining[bt] = self.shelf_life_days
        
        total_stock = sum(self.current_stock.values())
        target_stock = 500
        efficiency = 1 - abs(total_stock - target_stock) / target_stock
        efficiency = max(0, min(1, efficiency))
        
        reward = self.reward_fn.calculate_reward(
            n_stockout=1 if self.stockout_incidents > 0 else 0,
            n_waste=self.wasted_kantong,
            procurement_cost=sum(procurement_dict.values()),
            efficiency_score=efficiency
        )
        
        self.current_step += 1
        obs = self._get_observation()
        terminated = self.current_step >= self.max_steps
        
        info = {
            'stockout_incidents': self.stockout_incidents,
            'wasted_kantong': self.wasted_kantong,
            'total_procurement': self.total_procurement,
            'efficiency': efficiency
        }
        
        return obs, reward, terminated, False, info
    
    def render(self):
        pass


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("\n" + "#"*80)
    print("# REINFORCEMENT LEARNING ENVIRONMENT - BLOOD STOCK MANAGEMENT")
    print("#"*80)
    
    print("\n[STEP 1] Creating State Space...")
    state_space = BloodStockState()
    print(f"✓ State space created")
    
    print("\n[STEP 2] Creating Action Space...")
    action_space = BloodStockActions(n_hospitals=25)
    print(f"✓ Action space created")
    
    print("\n[STEP 3] Creating Reward Function...")
    reward_fn = BloodStockReward()
    print(f"✓ Reward function initialized")
    
    print("\n[STEP 4] Creating Custom Environment...")
    demand_data = pd.read_csv('data/processed/full_data.csv')
    demand_data['date'] = pd.to_datetime(demand_data['date'])
    
    env = BloodStockEnvironment(
        demand_data=demand_data,
        forecast_model={},
        blood_types=demand_data['blood_type'].unique(),
        n_hospitals=25
    )
    print(f"✓ Environment created")
    
    print("\n[STEP 5] Testing Environment...")
    obs, info = env.reset()
    print(f"  Initial observation shape: {obs.shape}")
    
    for step in range(5):
        action = env.action_space_gym.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"  Step {step+1}: reward={reward:.2f}, efficiency={info['efficiency']:.2f}")
    
    print("\n" + "#"*80)
    print("# RL ENVIRONMENT SETUP COMPLETED ✓")
    print("#"*80)
    print(f"\nNext step: Implement RL agent training")
    print("#"*80 + "\n")
