import numpy as np


class QLearningAgent:
    """Q-Learning agent for dynamic pricing decisions."""

    def __init__(self, num_actions, learning_rate=0.1, discount_factor=0.6,
                 epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.001):
        self.num_actions = num_actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.q_table = np.zeros(num_actions)
        self.training_rewards = []
        self.episode_count = 0

    def choose_action(self):
        if np.random.uniform(0, 1) < self.epsilon:
            return np.random.randint(0, self.num_actions)
        return np.argmax(self.q_table)

    def update_q_value(self, action_index, reward):
        current_q = self.q_table[action_index]
        max_next_q = np.max(self.q_table)
        new_q = (1 - self.learning_rate) * current_q + \
                self.learning_rate * (reward + self.discount_factor * max_next_q)
        self.q_table[action_index] = new_q

    def decay_epsilon(self):
        self.epsilon = max(
            self.epsilon_min,
            self.epsilon_min + (1.0 - self.epsilon_min) * np.exp(-self.epsilon_decay * self.episode_count)
        )
        self.episode_count += 1

    def get_optimal_action(self):
        return np.argmax(self.q_table)
