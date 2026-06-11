"""Reinforcement Learning Agent untuk Blood Stock Optimization

Tahap ini meliputi:
1. DQN Agent implementation
2. Training loop dengan experience replay
3. Policy learning dan evaluation
4. Integration dengan ARIMA forecasts
"""

import numpy as np
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import json
import matplotlib.pyplot as plt

# ============================================================================
# DEEP Q-NETWORK (DQN) IMPLEMENTATION
# ============================================================================

class DQNNetwork(nn.Module):
    """
    Deep Q-Network untuk Blood Stock Management
    
    Architecture:
    - Input: State (forecast + stock + shelf life + trend + seasonality)
    - Hidden: 2 dense layers dengan ReLU activation
    - Output: Q-values untuk setiap aksi
    """
    
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(DQNNetwork, self).__init__()
        
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        q_values = self.fc3(x)
        return q_values


# ============================================================================
# EXPERIENCE REPLAY BUFFER
# ============================================================================

class ReplayBuffer:
    """
    Experience Replay Buffer untuk DQN
    
    Menyimpan transisi (state, action, reward, next_state, done)
    untuk training yang lebih stable
    """
    
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """Simpan transisi ke buffer"""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """Sample random batch dari buffer"""
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        
        states = np.array([t[0] for t in batch])
        actions = np.array([t[1] for t in batch])
        rewards = np.array([t[2] for t in batch])
        next_states = np.array([t[3] for t in batch])
        dones = np.array([t[4] for t in batch])
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)


# ============================================================================
# DQN AGENT
# ============================================================================

class DQNAgent:
    """
    DQN Agent untuk Blood Stock Management
    
    Komponen:
    - Main Network: Predict Q-values
    - Target Network: Stable target values (update periodic)
    - Epsilon-Greedy: Exploration vs Exploitation
    """
    
    def __init__(self, state_dim, action_dim, learning_rate=1e-4, 
                 gamma=0.99, epsilon_start=1.0, epsilon_end=0.01, 
                 epsilon_decay=0.995, buffer_capacity=10000,
                 device='cpu'):
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.device = device
        
        # Networks
        self.q_network = DQNNetwork(state_dim, action_dim).to(device)
        self.target_network = DQNNetwork(state_dim, action_dim).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(capacity=buffer_capacity)
        
        # Training tracking
        self.training_rewards = []
        self.training_losses = []
        self.episodes = 0
        
        print(f"\n[DQN AGENT INITIALIZED]")
        print(f"  State dimension: {state_dim}")
        print(f"  Action dimension: {action_dim}")
        print(f"  Learning rate: {learning_rate}")
        print(f"  Gamma (discount): {gamma}")
        print(f"  Device: {device}")
    
    def select_action(self, state, training=True):
        """
        Select action dengan epsilon-greedy strategy
        
        Parameters:
        -----------
        state : np.array
            Current state
        training : bool
            If True, apply epsilon-greedy; else use greedy
        
        Returns:
        --------
        np.array : Action (continuous values in [0, 1])
        """
        
        if training and random.random() < self.epsilon:
            # Exploration: random action
            action = np.random.uniform(0, 1, self.action_dim).astype(np.float32)
        else:
            # Exploitation: greedy action
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.q_network(state_tensor)
                action_idx = q_values.argmax(dim=1).item()
                
                # Convert discrete action to continuous
                action = np.zeros(self.action_dim)
                action[action_idx % self.action_dim] = 1.0
        
        return action
    
    def train_step(self, batch_size=32):
        """
        Training step: update Q-network dari replay buffer
        
        Parameters:
        -----------
        batch_size : int
            Batch size untuk training
        
        Returns:
        --------
        float : Loss value
        """
        
        if len(self.replay_buffer) < batch_size:
            return None
        
        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Q-values dari main network
        q_values = self.q_network(states)
        q_selected = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Target Q-values dari target network
        with torch.no_grad():
            next_q_values = self.target_network(next_states)
            next_q_max = next_q_values.max(dim=1)[0]
            target_q = rewards + (1 - dones) * self.gamma * next_q_max
        
        # Calculate loss
        loss = self.criterion(q_selected, target_q)
        
        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        return loss.item()
    
    def update_target_network(self):
        """Update target network dengan weights dari main network"""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def update_epsilon(self):
        """Decay epsilon untuk exploration"""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def save_model(self, filepath):
        """Save model weights"""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.q_network.state_dict(), filepath)
        print(f"✓ Model saved: {filepath}")
    
    def load_model(self, filepath):
        """Load model weights"""
        self.q_network.load_state_dict(torch.load(filepath, map_location=self.device))
        self.target_network.load_state_dict(self.q_network.state_dict())
        print(f"✓ Model loaded: {filepath}")


# ============================================================================
# RL TRAINING LOOP
# ============================================================================

def train_rl_agent(env, agent, episodes=100, batch_size=32, 
                   target_update_freq=1000, output_dir='results/rl_training'):
    """
    Main training loop untuk RL agent
    
    Parameters:
    -----------
    env : BloodStockEnvironment
        Custom environment
    agent : DQNAgent
        RL agent
    episodes : int
        Number of episodes
    batch_size : int
        Batch size untuk training
    target_update_freq : int
        Frequency untuk update target network
    output_dir : str
        Direktori untuk simpan results
    """
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*80)
    print("TRAINING DQN AGENT FOR BLOOD STOCK MANAGEMENT")
    print("="*80)
    
    episode_rewards = []
    episode_losses = []
    total_steps = 0
    
    for episode in range(episodes):
        # Reset environment
        state, _ = env.reset()
        episode_reward = 0
        episode_loss = 0
        episode_steps = 0
        
        while True:
            # Select action
            action = agent.select_action(state, training=True)
            
            # Step environment
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Store transition
            agent.replay_buffer.push(state, np.argmax(action), reward, next_state, done)
            
            # Training step
            loss = agent.train_step(batch_size)
            if loss is not None:
                episode_loss += loss
            
            # Update target network
            if total_steps % target_update_freq == 0:
                agent.update_target_network()
            
            episode_reward += reward
            state = next_state
            total_steps += 1
            episode_steps += 1
            
            if done:
                break
        
        # End of episode
        agent.update_epsilon()
        agent.episodes += 1
        
        episode_rewards.append(episode_reward)
        if episode_steps > 0:
            episode_loss /= episode_steps
        episode_losses.append(episode_loss)
        
        # Logging
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            print(f"\nEpisode {episode+1}/{episodes}")
            print(f"  Episode Reward: {episode_reward:.2f}")
            print(f"  Avg Reward (last 10): {avg_reward:.2f}")
            print(f"  Epsilon: {agent.epsilon:.4f}")
            print(f"  Buffer Size: {len(agent.replay_buffer)}")
            print(f"  Info: Stockout={info['stockout_incidents']}, Waste={info['wasted_kantong']}, Efficiency={info['efficiency']:.2f}")
    
    print("\n" + "="*80)
    print(f"TRAINING COMPLETED - Episodes: {episodes}, Total Steps: {total_steps}")
    print("="*80)
    
    # Save training results
    training_results = {
        'episodes': episodes,
        'total_steps': total_steps,
        'episode_rewards': episode_rewards,
        'episode_losses': episode_losses,
        'final_epsilon': float(agent.epsilon),
        'avg_final_reward': float(np.mean(episode_rewards[-10:]))
    }
    
    with open(f'{output_dir}/training_results.json', 'w') as f:
        json.dump(training_results, f, indent=2)
    
    print(f"✓ Saved: {output_dir}/training_results.json")
    
    # Plot training curves
    plot_training_results(episode_rewards, episode_losses, output_dir)
    
    return episode_rewards, episode_losses


# ============================================================================
# EVALUATION & VISUALIZATION
# ============================================================================

def plot_training_results(episode_rewards, episode_losses, output_dir):
    """
    Visualisasi training curves
    """
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle('DQN Agent Training Results', fontsize=14, fontweight='bold')
    
    # Rewards
    ax = axes[0]
    ax.plot(episode_rewards, label='Episode Reward', alpha=0.6)
    # Moving average
    window = 10
    moving_avg = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
    ax.plot(range(window-1, len(episode_rewards)), moving_avg, label=f'Moving Avg (window={window})', linewidth=2)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward')
    ax.set_title('Episode Rewards')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Losses
    ax = axes[1]
    ax.plot(episode_losses, label='Episode Loss', alpha=0.6, color='orange')
    moving_avg_loss = np.convolve(episode_losses, np.ones(window)/window, mode='valid')
    ax.plot(range(window-1, len(episode_losses)), moving_avg_loss, label=f'Moving Avg (window={window})', linewidth=2)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/training_curves.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir}/training_curves.png")
    plt.close()


def evaluate_agent(env, agent, num_episodes=10, output_dir='results/rl_evaluation'):
    """
    Evaluate trained agent pada test episodes
    
    Parameters:
    -----------
    env : BloodStockEnvironment
        Custom environment
    agent : DQNAgent
        Trained RL agent
    num_episodes : int
        Number of evaluation episodes
    output_dir : str
        Direktori untuk simpan results
    """
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*80)
    print(f"EVALUATING AGENT - {num_episodes} EPISODES")
    print("="*80)
    
    eval_rewards = []
    eval_stockouts = []
    eval_wastes = []
    eval_efficiencies = []
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        episode_info = {'stockout_incidents': 0, 'wasted_kantong': 0, 'efficiency': 0}
        
        while True:
            # Greedy action (no exploration)
            action = agent.select_action(state, training=False)
            next_state, reward, terminated, truncated, info = env.step(action)
            
            episode_reward += reward
            episode_info = info
            state = next_state
            
            if terminated or truncated:
                break
        
        eval_rewards.append(episode_reward)
        eval_stockouts.append(episode_info['stockout_incidents'])
        eval_wastes.append(episode_info['wasted_kantong'])
        eval_efficiencies.append(episode_info['efficiency'])
        
        print(f"\nEpisode {episode+1}/{num_episodes}")
        print(f"  Reward: {episode_reward:.2f}")
        print(f"  Stockout Incidents: {episode_info['stockout_incidents']}")
        print(f"  Wasted Kantong: {episode_info['wasted_kantong']}")
        print(f"  Efficiency: {episode_info['efficiency']:.2f}")
    
    print("\n" + "="*80)
    print("EVALUATION SUMMARY")
    print("="*80)
    print(f"Avg Reward: {np.mean(eval_rewards):.2f} ± {np.std(eval_rewards):.2f}")
    print(f"Avg Stockout Incidents: {np.mean(eval_stockouts):.2f}")
    print(f"Avg Wasted Kantong: {np.mean(eval_wastes):.2f}")
    print(f"Avg Efficiency: {np.mean(eval_efficiencies):.4f}")
    
    # Save evaluation results
    eval_results = {
        'num_episodes': num_episodes,
        'avg_reward': float(np.mean(eval_rewards)),
        'std_reward': float(np.std(eval_rewards)),
        'avg_stockout_incidents': float(np.mean(eval_stockouts)),
        'avg_wasted_kantong': float(np.mean(eval_wastes)),
        'avg_efficiency': float(np.mean(eval_efficiencies)),
        'episode_rewards': eval_rewards,
        'episode_stockouts': eval_stockouts,
        'episode_wastes': eval_wastes,
        'episode_efficiencies': eval_efficiencies
    }
    
    with open(f'{output_dir}/evaluation_results.json', 'w') as f:
        json.dump(eval_results, f, indent=2)
    
    print(f"\n✓ Saved: {output_dir}/evaluation_results.json")
    
    return eval_results


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("\n" + "#"*80)
    print("# REINFORCEMENT LEARNING AGENT TRAINING")
    print("#"*80)
    
    # Import environment
    from src.src_03_rl_environment import BloodStockEnvironment
    
    # Load demand data
    print("\n[STEP 1] Loading data...")
    demand_data = pd.read_csv('data/processed/full_data.csv')
    demand_data['date'] = pd.to_datetime(demand_data['date'])
    print(f"✓ Data loaded: {len(demand_data)} records")
    
    # Create environment
    print("\n[STEP 2] Creating environment...")
    env = BloodStockEnvironment(
        demand_data=demand_data,
        blood_types=demand_data['blood_type'].unique(),
        n_hospitals=25
    )
    print(f"✓ Environment created")
    
    # Create agent
    print("\n[STEP 3] Creating DQN agent...")
    state_dim = env.observation_space.shape[0]
    action_dim = int(env.action_space_gym.shape[0])
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        learning_rate=1e-4,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995,
        buffer_capacity=10000,
        device=device
    )
    print(f"✓ Agent created (device: {device})")
    
    # Train agent
    print("\n[STEP 4] Training agent...")
    episode_rewards, episode_losses = train_rl_agent(
        env=env,
        agent=agent,
        episodes=100,
        batch_size=32,
        target_update_freq=1000
    )
    print(f"✓ Training completed")
    
    # Save trained model
    print("\n[STEP 5] Saving model...")
    agent.save_model('results/rl_training/dqn_agent.pt')
    print(f"✓ Model saved")
    
    # Evaluate agent
    print("\n[STEP 6] Evaluating agent...")
    eval_results = evaluate_agent(env, agent, num_episodes=10)
    print(f"✓ Evaluation completed")
    
    print("\n" + "#"*80)
    print("# RL AGENT TRAINING COMPLETED ✓")
    print("#"*80)
    print(f"\nNext step: Run 'python src/05_hybrid_system.py'")
    print("#"*80 + "\n")
