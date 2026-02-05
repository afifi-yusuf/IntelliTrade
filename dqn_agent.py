"""
Deep Q-Network (DQN) Agent for Trading

Implements:
- Neural network Q-function approximation
- Experience replay buffer
- Target network for stable learning
- Epsilon-greedy exploration
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque, namedtuple
import random
from typing import Tuple, List

# Experience tuple for replay buffer
Experience = namedtuple('Experience', 
                       ['state', 'action', 'reward', 'next_state', 'done'])


class DQNNetwork(nn.Module):
    """
    Deep Q-Network: Neural network that approximates Q(s,a)
    
    Input: State vector (narrative_embedding + technical_features + financial_state)
    Output: Q-values for each action [BUY, HOLD, SELL]
    """
    def __init__(self, state_dim: int, action_dim: int = 3, hidden_dims: List[int] = [256, 128, 64]):
        super(DQNNetwork, self).__init__()
        
        layers = []
        input_dim = state_dim
        
        # Build hidden layers
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))  # Prevent overfitting
            input_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(input_dim, action_dim))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass: state -> Q-values"""
        return self.network(state)


class ReplayBuffer:
    """
    Experience Replay Buffer
    
    Stores past experiences and samples random batches for training.
    This breaks correlation between consecutive samples.
    """
    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state: np.ndarray, action: int, reward: float, 
             next_state: np.ndarray, done: bool):
        """Store experience in buffer"""
        self.buffer.append(Experience(state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> List[Experience]:
        """Sample random batch of experiences"""
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))
    
    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    """
    Deep Q-Network Agent for Trading
    
    Uses a neural network to approximate Q-values instead of a Q-table.
    Handles continuous state spaces and learns complex patterns.
    """
    def __init__(
        self,
        state_dim: int,
        action_dim: int = 3,
        learning_rate: float = 0.001,
        discount_factor: float = 0.95,
        epsilon: float = 0.1,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.01,
        buffer_size: int = 10000,
        batch_size: int = 64,
        target_update_freq: int = 10,
        enable_learning: bool = True
    ):
        """
        Args:
            state_dim: Dimension of state vector
            action_dim: Number of actions (3: BUY, HOLD, SELL)
            learning_rate: Learning rate for optimizer
            discount_factor (γ): Discount future rewards
            epsilon (ε): Exploration rate
            epsilon_decay: Decay rate for epsilon
            epsilon_min: Minimum epsilon
            buffer_size: Size of replay buffer
            batch_size: Batch size for training
            target_update_freq: Update target network every N steps
            enable_learning: If True, train the network
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.enable_learning = enable_learning
        
        # Device (CPU or GPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Q-Network and Target Network
        self.q_network = DQNNetwork(state_dim, action_dim).to(self.device)
        self.target_network = DQNNetwork(state_dim, action_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()  # Target network in eval mode
        
        # Optimizer and loss
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.loss_fn = nn.SmoothL1Loss()  # Huber loss (more robust than MSE)
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size)
        
        # Action mapping
        self.actions = ["BUY", "HOLD", "SELL"]
        
        # Tracking
        self.training_steps = 0
        self.episode_rewards = []
        self.losses = []
        self.reward_history = []
        self.nan_skips = 0
        
        print(f"🧠 DQN Agent initialized on {self.device}")
        print(f"   State dim: {state_dim}, Action dim: {action_dim}")
        print(f"   Network: {sum(p.numel() for p in self.q_network.parameters())} parameters")
    
    def get_state_vector(self, narrative_vector: np.ndarray, 
                         stock_features: dict, 
                         narrative_state: int,
                         financial_state: int) -> np.ndarray:
        """
        Combine all state information into a single vector
        
        Returns:
            State vector: [narrative_embedding, technical_features, categorical_features]
        """
        # Ensure narrative_vector is 1-dimensional
        narrative_vector = np.atleast_1d(narrative_vector).astype(np.float32)
        if narrative_vector.ndim > 1:
            narrative_vector = narrative_vector.flatten()
        
        # Ensure consistent embedding dimension based on configured state_dim
        # Pad with zeros or truncate to match expected dimension
        expected_embedding_dim = max(1, self.state_dim - (7 + 3 + 2))
        if len(narrative_vector) < expected_embedding_dim:
            # Pad with zeros
            padding = np.zeros(expected_embedding_dim - len(narrative_vector), dtype=np.float32)
            narrative_vector = np.concatenate([narrative_vector, padding])
        elif len(narrative_vector) > expected_embedding_dim:
            # Truncate
            narrative_vector = narrative_vector[:expected_embedding_dim]
        
        # Normalize narrative embedding to stabilize training
        norm = np.linalg.norm(narrative_vector)
        if norm > 0:
            narrative_vector = narrative_vector / norm

        # Build state parts list
        state_parts = [narrative_vector]
        
        # Technical indicators (7 features)
        tech_features = np.array([
            float(stock_features.get('momentum', 0.0)),
            float(stock_features.get('volatility', 0.0)),
            float(stock_features.get('trend', 0.0)),
            float(stock_features.get('rsi', 50.0)) / 100.0,  # Normalize to 0-1
            float(stock_features.get('price_change_1d', 0.0)) / 100.0,
            float(stock_features.get('price_change_5d', 0.0)) / 100.0,
            float(stock_features.get('price_change_20d', 0.0)) / 100.0
        ], dtype=np.float32)
        state_parts.append(tech_features)
        
        # One-hot encode categorical states (3 + 2 = 5 features)
        narrative_onehot = np.zeros(3, dtype=np.float32)
        narrative_onehot[int(narrative_state)] = 1.0
        state_parts.append(narrative_onehot)
        
        financial_onehot = np.zeros(2, dtype=np.float32)
        financial_onehot[int(financial_state)] = 1.0
        state_parts.append(financial_onehot)
        
        # Concatenate all parts
        state_vector = np.concatenate(state_parts)
        
        return state_vector
    
    def get_action(self, state: np.ndarray, explore: bool = True) -> Tuple[int, np.ndarray]:
        """
        Select action using epsilon-greedy policy
        
        Args:
            state: State vector
            explore: If True, use epsilon-greedy; if False, always exploit
        
        Returns:
            action_idx: 0=BUY, 1=HOLD, 2=SELL
            q_values: Q-values for all actions
        """
        # Epsilon-greedy exploration
        if explore and self.enable_learning and random.random() < self.epsilon:
            # Explore: random action
            action_idx = random.randint(0, self.action_dim - 1)
            q_values = np.zeros(self.action_dim)
        else:
            # Exploit: best action from Q-network
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                q_values = self.q_network(state_tensor).cpu().numpy()[0]
            if not np.all(np.isfinite(q_values)):
                action_idx = random.randint(0, self.action_dim - 1)
                q_values = np.zeros(self.action_dim)
                return action_idx, q_values
            
            action_idx = np.argmax(q_values)
        
        return action_idx, q_values
    
    def store_experience(self, state: np.ndarray, action: int, reward: float,
                        next_state: np.ndarray, done: bool = False):
        """Store experience in replay buffer"""
        if self.enable_learning:
            self.replay_buffer.push(state, action, reward, next_state, done)
    
    def train_step(self):
        """
        Perform one training step using experience replay
        
        Updates Q-network using samples from replay buffer.
        """
        if not self.enable_learning or len(self.replay_buffer) < self.batch_size:
            return
        
        # Sample batch from replay buffer
        batch = self.replay_buffer.sample(self.batch_size)
        
        # Unpack batch
        states = torch.FloatTensor(np.array([e.state for e in batch])).to(self.device)
        actions = torch.LongTensor([e.action for e in batch]).to(self.device)
        rewards = torch.FloatTensor([e.reward for e in batch]).to(self.device)
        next_states = torch.FloatTensor(np.array([e.next_state for e in batch])).to(self.device)
        dones = torch.FloatTensor([e.done for e in batch]).to(self.device)
        
        if not (torch.isfinite(states).all() and torch.isfinite(next_states).all() and torch.isfinite(rewards).all()):
            self.nan_skips += 1
            return
        
        # Current Q-values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Target Q-values (using target network for stability)
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
            target_q_values = torch.clamp(target_q_values, -50.0, 50.0)
        
        # Compute loss
        loss = self.loss_fn(current_q_values, target_q_values)

        # Guard against NaNs/Infs
        if not torch.isfinite(loss):
            self.nan_skips += 1
            return
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)  # Gradient clipping
        self.optimizer.step()
        
        # Track loss
        self.losses.append(loss.item())
        self.training_steps += 1
        
        # Update target network periodically
        if self.training_steps % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def get_signal(self, narrative_vector: np.ndarray, price_history, 
                   financial_data: dict, narrative_state: int = None,
                   financial_state: int = None) -> dict:
        """
        Get trading signal (compatible with backtest interface)
        """
        # Import here to avoid circular dependency
        from rl_agent import MarkovRLAgent
        
        # Extract features using existing RL agent methods
        temp_agent = MarkovRLAgent()
        stock_features = temp_agent.extract_stock_features(price_history)
        
        if narrative_state is None:
            narrative_state = 1  # Default neutral
        if financial_state is None:
            financial_state = temp_agent.assess_financial_state(financial_data)
        
        # Get state vector
        state = self.get_state_vector(
            narrative_vector, stock_features, narrative_state, financial_state
        )
        
        # Get action
        action_idx, q_values = self.get_action(state, explore=True)
        action = self.actions[action_idx]
        
        # Store current state and action for learning
        self.current_state = state
        self.current_action_idx = action_idx
        
        # Build response
        return {
            "action": action,
            "confidence": float(np.max(np.abs(q_values))) if len(q_values) > 0 else 0.5,
            "narrative_state": narrative_state,
            "financial_state": financial_state,
            "stock_features": stock_features,
            "q_values": {
                "BUY": float(q_values[0]) if len(q_values) > 0 else 0.0,
                "HOLD": float(q_values[1]) if len(q_values) > 0 else 0.0,
                "SELL": float(q_values[2]) if len(q_values) > 0 else 0.0
            },
            "state_vector": state  # Store for learning
        }
    
    def update(self, current_portfolio_value: float, next_state_key: tuple = None, next_q_values: dict = None):
        """
        Update method (compatible with backtest interface)
        
        For DQN, we:
        1. Calculate reward from portfolio value change
        2. Store experience in replay buffer
        3. Train on batch from replay buffer
        """
        if not self.enable_learning:
            return
        
        if not hasattr(self, 'current_state') or not hasattr(self, 'current_action_idx'):
            return
        
        # Calculate reward
        if hasattr(self, 'last_portfolio_value') and hasattr(self, 'last_state') and hasattr(self, 'last_action_idx'):
            # Calculate basic return
            raw_reward = (current_portfolio_value - self.last_portfolio_value) / self.last_portfolio_value * 100
            
            # Risk-Adjustment (Volatility Penalty)
            # We use the standard deviation of recent rewards as a proxy for volatility
            self.reward_history.append(float(raw_reward))
            volatility = np.std(self.reward_history[-20:]) if len(self.reward_history) >= 20 else 0.0
            
            # Reward = Return - (Penalty_Factor * Volatility)
            # This penalizes erratic returns, encouraging stable growth (Sharpe-like behavior)
            reward = raw_reward - (0.1 * volatility)
            
            reward = np.clip(reward, -10, 10)  # Clip result
            
            # Store experience
            if hasattr(self, 'current_state'):
                self.store_experience(
                    self.last_state,
                    self.last_action_idx,
                    reward,
                    self.current_state,
                    done=False
                )
                
                # Train on experience
                self.train_step()
                
                # Print progress every 50 steps
                if self.training_steps % 50 == 0 and self.training_steps > 0:
                    stats = self.get_learning_stats()
                    print(f"   🧠 DQN Update #{self.training_steps} | "
                          f"Loss: {stats['avg_loss']:.4f} | "
                          f"ε: {stats['exploration_rate']:.3f} | "
                          f"Buffer: {stats['buffer_size']}")
        
        # Update last values for next iteration
        if not hasattr(self, 'last_state') or not hasattr(self, 'last_action_idx') or not hasattr(self, 'last_portfolio_value'):
            self.last_state = self.current_state
            self.last_action_idx = self.current_action_idx
            self.last_portfolio_value = current_portfolio_value
            return

        self.last_state = self.current_state
        self.last_action_idx = self.current_action_idx
        self.last_portfolio_value = current_portfolio_value
    
    def get_learning_stats(self) -> dict:
        """Get learning statistics (compatible with Q-learning interface)"""
        finite_losses = [v for v in self.losses if np.isfinite(v)]
        finite_rewards = [v for v in self.reward_history if np.isfinite(v)]
        return {
            'total_updates': self.training_steps,
            'avg_reward': float(np.mean(finite_rewards)) if finite_rewards else 0.0,
            'std_reward': float(np.std(finite_rewards)) if finite_rewards else 0.0,
            'avg_q_change': float(np.mean(finite_losses[-100:])) if finite_losses else 0.0,
            'q_table_size': self.training_steps,  # Use training steps as proxy
            'exploration_rate': self.epsilon,
            # DQN-specific stats
            'training_steps': self.training_steps,
            'buffer_size': len(self.replay_buffer),
            'avg_loss': float(np.mean(finite_losses[-100:])) if finite_losses else 0.0,
            'network_params': sum(p.numel() for p in self.q_network.parameters()),
            'nan_skips': self.nan_skips
        }
    
    def save_model(self, filepath: str):
        """Save DQN model"""
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'training_steps': self.training_steps,
            'epsilon': self.epsilon,
            'losses': self.losses
        }, filepath)
        print(f"💾 DQN model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load DQN model"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.training_steps = checkpoint['training_steps']
        self.epsilon = checkpoint['epsilon']
        self.losses = checkpoint['losses']
        print(f"📂 DQN model loaded from {filepath}")
