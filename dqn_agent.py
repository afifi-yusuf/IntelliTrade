"""
Deep Q-Network (DQN) Agent for Trading

Implements:
- Neural network Q-function approximation (single-branch MLP, 30-dim state)
- Experience replay buffer
- Target network for stable learning
- Double DQN (online net selects action, target net evaluates)
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

# State vector layout (30 dims total):
#   tech_features        7   momentum, volatility, trend, rsi, pc_1d, pc_5d, pc_20d
#   narrative_onehot     3   bullish / neutral / bearish
#   financial_onehot     2   good / weak
#   position_ratio       1   fraction of portfolio in stock
#   structured_signals   6   revenue_trend, margin_trend, peer_relative,
#                            historical_momentum, risk_level, overall_sentiment
#   signal_deltas        6   quarter-over-quarter changes in structured_signals
#   context_meta         5   days_since_report_norm, is_new_quarter,
#                            agent_agreement, evidence_strength, parse_success
STATE_DIM = 7 + 3 + 2 + 1 + 6 + 6 + 5  # = 30


class DQNNetwork(nn.Module):
    """
    Deep Q-Network: single-branch MLP that approximates Q(s,a).

    Input:  30-dim compact state vector
    Output: Q-values for each action [BUY, HOLD, SELL]
    """

    def __init__(self, state_dim: int = STATE_DIM, action_dim: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        if state.dim() == 1:
            state = state.unsqueeze(0)
        return self.net(state)


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
        self.buffer.append(Experience(state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> List[Experience]:
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    """
    Deep Q-Network Agent for Trading

    Uses a single-branch MLP over a 30-dim state to approximate Q-values.
    Employs Double DQN to reduce overestimation bias.
    """

    # State vector component sizes (must sum to STATE_DIM = 30)
    TECH_DIM = 7
    NARRATIVE_CAT_DIM = 3
    FINANCIAL_CAT_DIM = 2
    POSITION_DIM = 1
    STRUCTURED_SIGNAL_DIM = 6
    SIGNAL_DELTA_DIM = 6
    CONTEXT_META_DIM = 5

    def __init__(
        self,
        state_dim: int = STATE_DIM,
        action_dim: int = 3,
        learning_rate: float = 0.001,
        discount_factor: float = 0.95,
        epsilon: float = 0.1,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.01,
        buffer_size: int = 10000,
        batch_size: int = 64,
        target_update_freq: int = 10,
        enable_learning: bool = True,
        # Legacy parameter accepted but ignored (kept for call-site compatibility)
        embedding_dim: Optional[int] = None,
    ):
        self.state_dim = STATE_DIM  # always 30; ignore passed state_dim if wrong
        self.action_dim = action_dim
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.enable_learning = enable_learning

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.q_network = DQNNetwork(self.state_dim, action_dim).to(self.device)
        self.target_network = DQNNetwork(self.state_dim, action_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        if self.enable_learning:
            self.q_network.train()
        else:
            self.q_network.eval()

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.loss_fn = nn.SmoothL1Loss()

        self.replay_buffer = ReplayBuffer(buffer_size)
        self.actions = ["BUY", "HOLD", "SELL"]

        # Tracking
        self.training_steps = 0
        self.episode_rewards = []
        self.losses = []
        self.reward_history = []
        self.nan_skips = 0

        n_params = sum(p.numel() for p in self.q_network.parameters())
        print(f"🧠 DQN Agent initialized on {self.device}")
        print(f"   State dim: {self.state_dim}  |  Action dim: {action_dim}")
        print(f"   Network: {n_params} parameters (single-branch MLP, Double DQN)")

    # -------------------------------------------------------------------------
    # State construction
    # -------------------------------------------------------------------------

    def get_state_vector(
        self,
        stock_features: dict,
        narrative_state: int,
        financial_state: int,
        position_ratio: float = 0.0,
        structured_signals: np.ndarray = None,
        structured_signal_deltas: np.ndarray = None,
        context_meta: np.ndarray = None,
        # Legacy parameter accepted but ignored
        narrative_vector: np.ndarray = None,
    ) -> np.ndarray:
        """
        Build the 30-dim state vector from compact features.

        Parameters
        ----------
        stock_features : dict  Technical indicators from MarkovRLAgent.
        narrative_state : int  0=Bullish, 1=Neutral, 2=Bearish.
        financial_state : int  0=Good, 1=Weak.
        position_ratio  : float  Fraction of portfolio currently in the stock.
        structured_signals     : ndarray(6)  LLM-derived signals (may be decayed).
        structured_signal_deltas: ndarray(6) Quarter-over-quarter delta.
        context_meta           : ndarray(5)  Reliability/staleness meta-features.
        narrative_vector       : ignored (kept for backward compatibility).
        """
        state_parts = []

        # ---- Technical indicators (7) ----
        tech = np.array([
            float(stock_features.get('momentum', 0.0)),
            float(stock_features.get('volatility', 0.0)),
            float(stock_features.get('trend', 0.0)),
            float(stock_features.get('rsi', 50.0)) / 100.0,
            float(stock_features.get('price_change_1d', 0.0)) / 100.0,
            float(stock_features.get('price_change_5d', 0.0)) / 100.0,
            float(stock_features.get('price_change_20d', 0.0)) / 100.0,
        ], dtype=np.float32)
        state_parts.append(tech)

        # ---- Narrative one-hot (3) ----
        narrative_onehot = np.zeros(self.NARRATIVE_CAT_DIM, dtype=np.float32)
        narrative_onehot[int(np.clip(narrative_state, 0, 2))] = 1.0
        state_parts.append(narrative_onehot)

        # ---- Financial one-hot (2) ----
        financial_onehot = np.zeros(self.FINANCIAL_CAT_DIM, dtype=np.float32)
        financial_onehot[int(np.clip(financial_state, 0, 1))] = 1.0
        state_parts.append(financial_onehot)

        # ---- Position ratio (1) ----
        state_parts.append(np.array([float(position_ratio)], dtype=np.float32))

        # ---- Structured signals (6) ----
        if structured_signals is not None and len(structured_signals) >= self.STRUCTURED_SIGNAL_DIM:
            state_parts.append(np.array(structured_signals[:self.STRUCTURED_SIGNAL_DIM], dtype=np.float32))
        else:
            state_parts.append(np.zeros(self.STRUCTURED_SIGNAL_DIM, dtype=np.float32))

        # ---- Signal deltas (6) ----
        if structured_signal_deltas is not None and len(structured_signal_deltas) >= self.SIGNAL_DELTA_DIM:
            state_parts.append(np.array(structured_signal_deltas[:self.SIGNAL_DELTA_DIM], dtype=np.float32))
        else:
            state_parts.append(np.zeros(self.SIGNAL_DELTA_DIM, dtype=np.float32))

        # ---- Context meta (5) ----
        if context_meta is not None and len(context_meta) >= self.CONTEXT_META_DIM:
            state_parts.append(np.array(context_meta[:self.CONTEXT_META_DIM], dtype=np.float32))
        else:
            state_parts.append(np.zeros(self.CONTEXT_META_DIM, dtype=np.float32))

        state_vector = np.concatenate(state_parts)

        # Safety: pad/truncate to exact state_dim
        if len(state_vector) < self.state_dim:
            state_vector = np.concatenate([state_vector,
                                           np.zeros(self.state_dim - len(state_vector), dtype=np.float32)])
        elif len(state_vector) > self.state_dim:
            state_vector = state_vector[:self.state_dim]

        return state_vector

    # -------------------------------------------------------------------------
    # Action selection
    # -------------------------------------------------------------------------

    def get_action(
        self,
        state: np.ndarray,
        explore: bool = True,
        position_state: str = "FLAT",
    ) -> Tuple[int, np.ndarray]:
        """
        Select action using epsilon-greedy policy with long-only constraints.

        Returns
        -------
        action_idx : 0=BUY, 1=HOLD, 2=SELL
        q_values   : Q-values for all actions
        """
        position_state = (position_state or "FLAT").upper()
        if position_state not in {"FLAT", "LONG"}:
            position_state = "FLAT"

        valid_actions = [0, 1] if position_state == "FLAT" else [1, 2]

        if explore and self.enable_learning and random.random() < self.epsilon:
            return random.choice(valid_actions), np.zeros(self.action_dim)

        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_network(state_tensor).cpu().numpy()[0]

        if not np.all(np.isfinite(q_values)):
            return random.choice(valid_actions), np.zeros(self.action_dim)

        masked = q_values.copy()
        if position_state == "FLAT":
            masked[2] = -np.inf
        else:
            masked[0] = -np.inf

        return int(np.argmax(masked)), q_values

    # -------------------------------------------------------------------------
    # Experience replay
    # -------------------------------------------------------------------------

    def store_experience(self, state: np.ndarray, action: int, reward: float,
                         next_state: np.ndarray, done: bool = False):
        if self.enable_learning:
            self.replay_buffer.push(state, action, reward, next_state, done)

    def train_step(self):
        """
        One training step with Double DQN update.

        Online network selects the greedy action; target network evaluates it.
        This decouples action selection from value estimation, reducing
        the overestimation bias present in vanilla DQN.
        """
        if not self.enable_learning or len(self.replay_buffer) < self.batch_size:
            return

        batch = self.replay_buffer.sample(self.batch_size)

        states      = torch.FloatTensor(np.array([e.state      for e in batch])).to(self.device)
        actions     = torch.LongTensor( [e.action               for e in batch]).to(self.device)
        rewards     = torch.FloatTensor([e.reward               for e in batch]).to(self.device)
        next_states = torch.FloatTensor(np.array([e.next_state for e in batch])).to(self.device)
        dones       = torch.FloatTensor([e.done                 for e in batch]).to(self.device)

        if not (torch.isfinite(states).all() and torch.isfinite(next_states).all()
                and torch.isfinite(rewards).all()):
            self.nan_skips += 1
            return

        # Current Q-values
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Double DQN target: online net picks action, target net evaluates it
        with torch.no_grad():
            best_actions = self.q_network(next_states).argmax(1)
            next_q = self.target_network(next_states).gather(
                1, best_actions.unsqueeze(1)
            ).squeeze(1)
            target_q = rewards + (1 - dones) * self.gamma * next_q
            target_q = torch.clamp(target_q, -50.0, 50.0)

        loss = self.loss_fn(current_q, target_q)
        if not torch.isfinite(loss):
            self.nan_skips += 1
            return

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()

        self.losses.append(loss.item())
        self.training_steps += 1

        if self.training_steps % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    # -------------------------------------------------------------------------
    # High-level signal interface (compatible with backtest)
    # -------------------------------------------------------------------------

    def get_signal(
        self,
        price_history,
        financial_data: dict,
        narrative_state: int = None,
        financial_state: int = None,
        position_ratio: float = 0.0,
        structured_signals: np.ndarray = None,
        structured_signal_deltas: np.ndarray = None,
        context_meta: np.ndarray = None,
        position_state: str = "FLAT",
        explore: bool = False,
        # Legacy parameter accepted but ignored
        narrative_vector: np.ndarray = None,
    ) -> dict:
        """
        Generate a trading signal (compatible with the backtest interface).

        Parameters
        ----------
        narrative_vector : ignored (kept for backward compatibility).
        explore          : if True, use epsilon-greedy exploration.
        """
        from rl_agent import MarkovRLAgent
        temp_agent = MarkovRLAgent()
        stock_features = temp_agent.extract_stock_features(price_history)

        if narrative_state is None:
            narrative_state = 1  # neutral default
        if financial_state is None:
            financial_state = temp_agent.assess_financial_state(financial_data)

        state = self.get_state_vector(
            stock_features=stock_features,
            narrative_state=narrative_state,
            financial_state=financial_state,
            position_ratio=position_ratio,
            structured_signals=structured_signals,
            structured_signal_deltas=structured_signal_deltas,
            context_meta=context_meta,
        )

        action_idx, q_values = self.get_action(state, explore=explore,
                                               position_state=position_state)
        action = self.actions[action_idx]

        if len(q_values) > 0 and np.all(np.isfinite(q_values)):
            q_shifted = q_values - np.max(q_values)
            exp_q = np.exp(np.clip(q_shifted, -50.0, 50.0))
            softmax_probs = exp_q / np.sum(exp_q)
            confidence = float(softmax_probs[action_idx])
        else:
            confidence = 0.5

        self.current_state = state
        self.current_action_idx = action_idx

        return {
            "action": action,
            "confidence": confidence,
            "narrative_state": narrative_state,
            "financial_state": financial_state,
            "stock_features": stock_features,
            "q_values": {
                "BUY":  float(q_values[0]) if len(q_values) > 0 else 0.0,
                "HOLD": float(q_values[1]) if len(q_values) > 0 else 0.0,
                "SELL": float(q_values[2]) if len(q_values) > 0 else 0.0,
            },
            "state_vector": state,
        }

    def update(self, current_portfolio_value: float,
               next_state_key: tuple = None, next_q_values: dict = None):
        """
        Update agent from portfolio value change (backtest interface).
        """
        if not self.enable_learning:
            return
        if not hasattr(self, 'current_state') or not hasattr(self, 'current_action_idx'):
            return

        if (hasattr(self, 'last_portfolio_value')
                and hasattr(self, 'last_state')
                and hasattr(self, 'last_action_idx')):

            raw_reward = ((current_portfolio_value - self.last_portfolio_value)
                          / self.last_portfolio_value * 100)

            self.reward_history.append(float(raw_reward))
            volatility = (np.std(self.reward_history[-20:])
                          if len(self.reward_history) >= 20 else 0.0)

            reward = raw_reward - 0.1 * volatility

            last_action_name = self.actions[self.last_action_idx]
            if last_action_name == "HOLD" and raw_reward < -0.5:
                reward -= 0.3
            if last_action_name == "SELL" and raw_reward > 0:
                reward += 0.2

            reward = np.clip(reward, -10, 10)

            if hasattr(self, 'current_state'):
                self.store_experience(self.last_state, self.last_action_idx,
                                      reward, self.current_state, done=False)
                self.train_step()

                if self.training_steps % 50 == 0 and self.training_steps > 0:
                    stats = self.get_learning_stats()
                    print(f"   🧠 DQN Update #{self.training_steps} | "
                          f"Loss: {stats['avg_loss']:.4f} | "
                          f"ε: {stats['exploration_rate']:.3f} | "
                          f"Buffer: {stats['buffer_size']}")

        if not (hasattr(self, 'last_state') and hasattr(self, 'last_action_idx')
                and hasattr(self, 'last_portfolio_value')):
            self.last_state = self.current_state
            self.last_action_idx = self.current_action_idx
            self.last_portfolio_value = current_portfolio_value
            return

        self.last_state = self.current_state
        self.last_action_idx = self.current_action_idx
        self.last_portfolio_value = current_portfolio_value

    def get_learning_stats(self) -> dict:
        finite_losses   = [v for v in self.losses         if np.isfinite(v)]
        finite_rewards  = [v for v in self.reward_history if np.isfinite(v)]
        return {
            'total_updates':     self.training_steps,
            'avg_reward':        float(np.mean(finite_rewards))        if finite_rewards else 0.0,
            'std_reward':        float(np.std(finite_rewards))         if finite_rewards else 0.0,
            'avg_q_change':      float(np.mean(finite_losses[-100:]))  if finite_losses  else 0.0,
            'q_table_size':      self.training_steps,
            'exploration_rate':  self.epsilon,
            'training_steps':    self.training_steps,
            'buffer_size':       len(self.replay_buffer),
            'avg_loss':          float(np.mean(finite_losses[-100:]))  if finite_losses  else 0.0,
            'network_params':    sum(p.numel() for p in self.q_network.parameters()),
            'nan_skips':         self.nan_skips,
        }

    # -------------------------------------------------------------------------
    # Persistence
    # -------------------------------------------------------------------------

    def save_model(self, filepath: str):
        torch.save({
            'q_network':      self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer':      self.optimizer.state_dict(),
            'training_steps': self.training_steps,
            'epsilon':        self.epsilon,
            'losses':         self.losses,
            'state_dim':      self.state_dim,
        }, filepath)
        print(f"💾 DQN model saved to {filepath}")

    def load_model(self, filepath: str):
        checkpoint = torch.load(filepath, map_location=self.device)

        def _partial_load(model: nn.Module, source_state: dict):
            target_state = model.state_dict()
            compatible = {
                k: v for k, v in source_state.items()
                if k in target_state and target_state[k].shape == v.shape
            }
            target_state.update(compatible)
            model.load_state_dict(target_state, strict=False)
            return len(compatible), len(source_state)

        try:
            self.q_network.load_state_dict(checkpoint['q_network'])
            self.target_network.load_state_dict(checkpoint['target_network'])
        except Exception as e:
            lq, tq = _partial_load(self.q_network, checkpoint.get('q_network', {}))
            lt, tt = _partial_load(self.target_network, checkpoint.get('target_network', {}))
            print(f"⚠️  Partial model load due to architecture mismatch: {e}")
            print(f"   q_network: {lq}/{tq}  |  target_network: {lt}/{tt} tensors loaded")

        try:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
        except Exception:
            pass

        self.training_steps = checkpoint.get('training_steps', 0)
        self.epsilon        = checkpoint.get('epsilon', self.epsilon)
        self.losses         = checkpoint.get('losses', [])

        if self.enable_learning:
            self.q_network.train()
        else:
            self.q_network.eval()
        self.target_network.eval()
        print(f"📂 DQN model loaded from {filepath}")
