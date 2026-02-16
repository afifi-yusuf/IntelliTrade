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
from typing import Tuple, List, Optional

# Experience tuple for replay buffer
Experience = namedtuple('Experience', 
                       ['state', 'action', 'reward', 'next_state', 'done'])


class DQNNetwork(nn.Module):
    """
    Deep Q-Network: Neural network that approximates Q(s,a)
    
    Input: State vector (narrative_embedding + technical_features + financial_state)
    Output: Q-values for each action [BUY, HOLD, SELL]
    """
    def __init__(self, state_dim: int, embedding_dim: int, action_dim: int = 3):
        super(DQNNetwork, self).__init__()

        self.embedding_dim = embedding_dim
        self.tech_dim = 7
        self.narrative_cat_dim = 3
        self.financial_cat_dim = 2
        self.position_dim = 1
        self.structured_signal_dim = 6
        self.context_meta_dim = 5
        self.expected_state_dim = (
            self.embedding_dim
            + self.tech_dim
            + self.narrative_cat_dim
            + self.financial_cat_dim
            + self.position_dim
            + self.structured_signal_dim
            + self.context_meta_dim
        )

        self.price_input_dim = self.tech_dim + self.financial_cat_dim + self.position_dim
        self.context_input_dim = self.embedding_dim + self.narrative_cat_dim + self.structured_signal_dim + self.context_meta_dim

        # Learned attention over structured signals
        self.signal_attention = nn.Sequential(
            nn.Linear(self.structured_signal_dim, 16),
            nn.ReLU(),
            nn.Linear(16, self.structured_signal_dim),
            nn.Softmax(dim=-1)
        )

        # Price branch: technical + position + financial state
        self.price_branch = nn.Sequential(
            nn.Linear(self.price_input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
        )

        # Context branch: embeddings + narrative state + attended structured signals + reliability meta
        self.context_branch = nn.Sequential(
            nn.Linear(self.context_input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
        )

        # Context gate (alpha in [0,1]) controls how much context branch contributes.
        self.context_gate = nn.Sequential(
            nn.Linear(self.context_meta_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

        self.q_head = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, action_dim)
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass with gated context integration."""
        if state.dim() == 1:
            state = state.unsqueeze(0)

        if state.shape[-1] < self.expected_state_dim:
            pad = torch.zeros(
                state.shape[0],
                self.expected_state_dim - state.shape[-1],
                device=state.device,
                dtype=state.dtype
            )
            state = torch.cat([state, pad], dim=-1)
        elif state.shape[-1] > self.expected_state_dim:
            state = state[:, :self.expected_state_dim]

        idx = 0
        narrative_embedding = state[:, idx:idx + self.embedding_dim]
        idx += self.embedding_dim
        tech_features = state[:, idx:idx + self.tech_dim]
        idx += self.tech_dim
        narrative_onehot = state[:, idx:idx + self.narrative_cat_dim]
        idx += self.narrative_cat_dim
        financial_onehot = state[:, idx:idx + self.financial_cat_dim]
        idx += self.financial_cat_dim
        position_ratio = state[:, idx:idx + self.position_dim]
        idx += self.position_dim
        structured_signals = state[:, idx:idx + self.structured_signal_dim]
        idx += self.structured_signal_dim
        context_meta = state[:, idx:idx + self.context_meta_dim]

        attention_weights = self.signal_attention(structured_signals)
        weighted_signals = structured_signals * attention_weights

        price_input = torch.cat([tech_features, financial_onehot, position_ratio], dim=-1)
        context_input = torch.cat([narrative_embedding, narrative_onehot, weighted_signals, context_meta], dim=-1)

        price_latent = self.price_branch(price_input)
        context_latent = self.context_branch(context_input)

        alpha = self.context_gate(context_meta)
        fused_latent = price_latent + alpha * context_latent
        return self.q_head(fused_latent)


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
        embedding_dim: Optional[int] = None,
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
            embedding_dim: Narrative embedding dimension inside state vector
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
        self.tech_dim = 7
        self.narrative_cat_dim = 3
        self.financial_cat_dim = 2
        self.position_dim = 1
        self.structured_signal_dim = 6
        self.context_meta_dim = 5

        if embedding_dim is None:
            embedding_dim = max(1, state_dim - (
                self.tech_dim
                + self.narrative_cat_dim
                + self.financial_cat_dim
                + self.position_dim
                + self.structured_signal_dim
                + self.context_meta_dim
            ))
        self.embedding_dim = int(embedding_dim)
        self.state_dim = (
            self.embedding_dim
            + self.tech_dim
            + self.narrative_cat_dim
            + self.financial_cat_dim
            + self.position_dim
            + self.structured_signal_dim
            + self.context_meta_dim
        )
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
        self.q_network = DQNNetwork(self.state_dim, embedding_dim=self.embedding_dim, action_dim=action_dim).to(self.device)
        self.target_network = DQNNetwork(self.state_dim, embedding_dim=self.embedding_dim, action_dim=action_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()  # Target network in eval mode
        if self.enable_learning:
            self.q_network.train()
        else:
            self.q_network.eval()
        
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
        print(f"   State dim: {self.state_dim}, Action dim: {action_dim}")
        print(f"   Network: {sum(p.numel() for p in self.q_network.parameters())} parameters")
    
    def get_state_vector(self, narrative_vector: np.ndarray, 
                         stock_features: dict, 
                         narrative_state: int,
                         financial_state: int,
                         position_ratio: float = 0.0,
                         structured_signals: np.ndarray = None,
                         context_meta: np.ndarray = None) -> np.ndarray:
        """
        Combine all state information into a single vector
        
        Returns:
            State vector: [narrative_embedding, technical_features, categorical_features, position_ratio, structured_signals]
        """
        # Ensure narrative_vector is 1-dimensional
        narrative_vector = np.atleast_1d(narrative_vector).astype(np.float32)
        if narrative_vector.ndim > 1:
            narrative_vector = narrative_vector.flatten()
        
        # Ensure consistent embedding dimension based on configured state layout.
        expected_embedding_dim = self.embedding_dim
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
        
        # Position ratio (1 feature)
        state_parts.append(np.array([float(position_ratio)], dtype=np.float32))
        
        # Structured signals from multi-agent analysis (6 features)
        if structured_signals is not None and len(structured_signals) >= 6:
            state_parts.append(np.array(structured_signals[:6], dtype=np.float32))
        else:
            state_parts.append(np.zeros(6, dtype=np.float32))

        # Context quality/reliability metadata (5 features)
        if context_meta is not None and len(context_meta) >= self.context_meta_dim:
            state_parts.append(np.array(context_meta[:self.context_meta_dim], dtype=np.float32))
        else:
            state_parts.append(np.zeros(self.context_meta_dim, dtype=np.float32))
        
        # Concatenate all parts
        state_vector = np.concatenate(state_parts)
        if len(state_vector) < self.state_dim:
            pad = np.zeros(self.state_dim - len(state_vector), dtype=np.float32)
            state_vector = np.concatenate([state_vector, pad])
        elif len(state_vector) > self.state_dim:
            state_vector = state_vector[:self.state_dim]
        
        return state_vector
    
    def get_action(
        self,
        state: np.ndarray,
        explore: bool = True,
        position_state: str = "FLAT"
    ) -> Tuple[int, np.ndarray]:
        """
        Select action using epsilon-greedy policy
        
        Args:
            state: State vector
            explore: If True, use epsilon-greedy; if False, always exploit
        
        Returns:
            action_idx: 0=BUY, 1=HOLD, 2=SELL
            q_values: Q-values for all actions
        """
        position_state = (position_state or "FLAT").upper()
        if position_state not in {"FLAT", "LONG"}:
            position_state = "FLAT"

        # Atomic long-only constraints:
        # FLAT -> can BUY or HOLD
        # LONG -> can SELL or HOLD
        valid_actions = [0, 1] if position_state == "FLAT" else [1, 2]

        # Epsilon-greedy exploration
        if explore and self.enable_learning and random.random() < self.epsilon:
            # Explore only among valid actions
            action_idx = random.choice(valid_actions)
            q_values = np.zeros(self.action_dim)
        else:
            # Exploit: best action from Q-network
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                q_values = self.q_network(state_tensor).cpu().numpy()[0]
            if not np.all(np.isfinite(q_values)):
                action_idx = random.choice(valid_actions)
                q_values = np.zeros(self.action_dim)
                return action_idx, q_values

            masked_q_values = np.array(q_values, copy=True)
            if position_state == "FLAT":
                masked_q_values[2] = -np.inf  # Can't SELL while flat
            else:
                masked_q_values[0] = -np.inf  # Can't BUY while already long

            action_idx = int(np.argmax(masked_q_values))
        
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
                   financial_state: int = None,
                   position_ratio: float = 0.0,
                   structured_signals: np.ndarray = None,
                   context_meta: np.ndarray = None,
                   position_state: str = "FLAT",
                   explore: bool = False) -> dict:
        """
        Get trading signal (compatible with backtest interface)
        
        Args:
            explore: If True, use epsilon-greedy exploration. Default False for inference.
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
        
        # Get state vector (now includes position_ratio and structured_signals)
        state = self.get_state_vector(
            narrative_vector, stock_features, narrative_state, financial_state,
            position_ratio=position_ratio,
            structured_signals=structured_signals,
            context_meta=context_meta
        )
        
        # Get action — only explore during training
        action_idx, q_values = self.get_action(
            state,
            explore=explore,
            position_state=position_state
        )
        action = self.actions[action_idx]

        # Confidence as action softmax probability for bounded, comparable scores.
        if len(q_values) > 0 and np.all(np.isfinite(q_values)):
            q_shifted = q_values - np.max(q_values)
            exp_q = np.exp(np.clip(q_shifted, -50.0, 50.0))
            softmax_probs = exp_q / np.sum(exp_q)
            confidence = float(softmax_probs[action_idx])
        else:
            confidence = 0.5
        
        # Store current state and action for learning
        self.current_state = state
        self.current_action_idx = action_idx
        
        # Build response
        return {
            "action": action,
            "confidence": confidence,
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
            self.reward_history.append(float(raw_reward))
            volatility = np.std(self.reward_history[-20:]) if len(self.reward_history) >= 20 else 0.0
            
            # Base reward: return minus volatility penalty
            reward = raw_reward - (0.1 * volatility)
            
            # Holding penalty: small negative reward for HOLD when losing money
            last_action_name = self.actions[self.last_action_idx]
            if last_action_name == "HOLD" and raw_reward < -0.5:
                reward -= 0.3  # Penalize holding during losses
            
            # Profit-taking bonus: reward SELL that locks in gains
            if last_action_name == "SELL" and raw_reward > 0:
                reward += 0.2  # Bonus for profitable exit
            
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
            'losses': self.losses,
            'state_dim': self.state_dim,
            'embedding_dim': self.embedding_dim,
            'context_meta_dim': self.context_meta_dim
        }, filepath)
        print(f"💾 DQN model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load DQN model"""
        checkpoint = torch.load(filepath, map_location=self.device)

        def _partial_load(model: nn.Module, source_state: dict):
            target_state = model.state_dict()
            compatible = {
                key: value for key, value in source_state.items()
                if key in target_state and target_state[key].shape == value.shape
            }
            target_state.update(compatible)
            model.load_state_dict(target_state, strict=False)
            return len(compatible), len(source_state)

        try:
            self.q_network.load_state_dict(checkpoint['q_network'])
            self.target_network.load_state_dict(checkpoint['target_network'])
        except Exception as e:
            loaded_q, total_q = _partial_load(self.q_network, checkpoint.get('q_network', {}))
            loaded_t, total_t = _partial_load(self.target_network, checkpoint.get('target_network', {}))
            print(f"⚠️ Partial model load due to architecture mismatch: {e}")
            print(f"   q_network: {loaded_q}/{total_q} tensors loaded")
            print(f"   target_network: {loaded_t}/{total_t} tensors loaded")

        try:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
        except Exception:
            pass

        self.training_steps = checkpoint.get('training_steps', 0)
        self.epsilon = checkpoint.get('epsilon', self.epsilon)
        self.losses = checkpoint.get('losses', [])
        # Ensure deterministic inference path (dropout disabled) when learning is off.
        if self.enable_learning:
            self.q_network.train()
        else:
            self.q_network.eval()
        self.target_network.eval()
        print(f"📂 DQN model loaded from {filepath}")
