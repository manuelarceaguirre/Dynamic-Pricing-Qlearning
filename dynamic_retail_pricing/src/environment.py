import numpy as np


def calculate_demand(price, base_demand, base_price, elasticity):
    """Calculate demand using Equation 1 from the paper."""
    if base_price == 0:
        return base_demand
    demand = base_demand + (base_demand * elasticity * (price - base_price) / base_price)
    return max(0, demand)


def calculate_reward(price, demand, cost_per_unit):
    """Calculate reward (profit) using Equation 3 from the paper."""
    revenue = price * demand
    total_cost = cost_per_unit * demand
    profit = revenue - total_cost
    return profit


class RetailEnvironment:
    """Simulates the retail pricing environment for Q-Learning."""

    def __init__(self, base_demand, base_price, elasticity, cost_per_unit, possible_prices):
        self.base_demand = base_demand
        self.base_price = base_price
        self.elasticity = elasticity
        self.cost_per_unit = cost_per_unit
        self.possible_prices = possible_prices
        self.num_actions = len(possible_prices)

    def step(self, action_index):
        price = self.possible_prices[action_index]
        demand = calculate_demand(price, self.base_demand, self.base_price, self.elasticity)
        reward = calculate_reward(price, demand, self.cost_per_unit)
        return reward, demand

    def get_price_from_action(self, action_index):
        return self.possible_prices[action_index]
