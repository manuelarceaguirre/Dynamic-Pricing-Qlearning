"""Q-learning agent for dynamic pricing."""
from typing import Tuple
import numpy as np

class QLearningAgent:
    def __init__(self, actions: np.ndarray, alpha: float = 0.1, gamma: float = 0.95, 
                 initial_epsilon: float = 1.0, min_epsilon: float = 0.01, epsilon_decay_rate: float = 0.001): # <<< MODIFIED: Epsilon params
        self.actions = actions # This is an array of possible price values
        self.alpha = alpha
        self.gamma = gamma
        
        self.initial_epsilon = initial_epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_decay_rate = epsilon_decay_rate
        self.epsilon = self.initial_epsilon # Current epsilon value

        self.q_table = {} # Q-table: keys are (state_tuple, action_price_float), values are q_values

    def get_q(self, state: Tuple[str, str], action_price: float) -> float: # <<< MODIFIED: state type, action_price type
        # state is (product_id, day_type)
        # action_price is one of the float values from self.actions
        return self.q_table.get((state, action_price), 0.0)

    # <<< NEW METHOD: Epsilon decay >>>
    def decay_epsilon(self, episode_num: int):
        self.epsilon = self.min_epsilon + \
                       (self.initial_epsilon - self.min_epsilon) * np.exp(-self.epsilon_decay_rate * episode_num)

    def choose_action(self, state: Tuple[str, str], exploit_only: bool = False) -> float: # <<< MODIFIED: state type, added exploit_only
        if not exploit_only and np.random.rand() < self.epsilon:
            return np.random.choice(self.actions) # Explore: pick a random price from the action list
        else:
            q_values_for_state = [self.get_q(state, action_price) for action_price in self.actions]
            
            if not q_values_for_state or all(q == 0 for q in q_values_for_state):
                return np.random.choice(self.actions)
            
            max_q_value = np.max(q_values_for_state)
            best_action_indices = [i for i, q in enumerate(q_values_for_state) if q == max_q_value]
            chosen_index = np.random.choice(best_action_indices)
            return float(self.actions[chosen_index])


    def learn(self, state: Tuple[str, str], action_price: float, reward: float, next_state: Tuple[str, str]): # <<< MODIFIED: state types, action_price type
        current_q = self.get_q(state, action_price)
        
        # Find max Q-value for the next_state over all possible actions
        max_next_q = 0.0 # Default if no actions or next_state leads to no further q-values
        if self.actions.size > 0: # Ensure there are actions to choose from
            q_values_for_next_state = [self.get_q(next_state, next_action_price) for next_action_price in self.actions]
            if q_values_for_next_state: # Ensure list is not empty
                 max_next_q = np.max(q_values_for_next_state)
        
        new_q = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)
        self.q_table[(state, action_price)] = new_q

    def train_episode(self, env) -> float: # env is an instance of PricingEnvironment
        state = env.reset() # state is (product_id, day_type)
        total_reward = 0.0
        
        num_steps_per_episode = 7 # Or len(env.action_space) if you want to try all actions

        for _ in range(num_steps_per_episode): 
            action_price = self.choose_action(state) # state is (product_id, day_type)
            next_state, reward = env.step(action_price) # next_state is (product_id, new_day_type)
            self.learn(state, action_price, reward, next_state)
            state = next_state
            total_reward += reward
        return total_reward
