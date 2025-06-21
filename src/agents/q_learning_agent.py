"""Q-learning agent for dynamic pricing."""
from typing import Tuple
import numpy as np
from . import __init__

class QLearningAgent:
    def __init__(self, actions: np.ndarray, alpha: float = 0.1, gamma: float = 0.95, epsilon: float = 0.1):
        self.actions = actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = {}

    def get_q(self, state: Tuple[str, float], action: float) -> float:
        return self.q_table.get((state, action), 0.0)

    def choose_action(self, state: Tuple[str, float]) -> float:
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.actions)
        qs = [self.get_q(state, a) for a in self.actions]
        return float(self.actions[int(np.argmax(qs))])

    def learn(self, state: Tuple[str, float], action: float, reward: float, next_state: Tuple[str, float]):
        current_q = self.get_q(state, action)
        max_next_q = max([self.get_q(next_state, a) for a in self.actions])
        new_q = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)
        self.q_table[(state, action)] = new_q

    def train_episode(self, env) -> float:
        state = env.reset()
        total_reward = 0.0
        for _ in range(len(env.action_space)):
            action = self.choose_action(state)
            next_state, reward = env.step(action)
            self.learn(state, action, reward, next_state)
            state = next_state
            total_reward += reward
        return total_reward
