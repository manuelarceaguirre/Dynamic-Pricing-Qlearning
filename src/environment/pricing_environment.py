"""Retail pricing simulation environment."""
from dataclasses import dataclass
from typing import Tuple
import numpy as np

@dataclass
class Product:
    product_id: str  # <<< MODIFIED: Added product_id
    base_price: float
    base_demand: float
    elasticity: float
    cost: float

class PricingEnvironment:
    """Environment for dynamic pricing."""
    def __init__(self, product: Product, price_grid: np.ndarray):
        self.product = product
        self.price_grid = price_grid
        # self.state_space = ['weekday', 'weekend'] # This can be implicit in current_state
        self.action_space = price_grid # These are the actual price values
        self.day = 0 # Represents day of the week, 0-6 (e.g., Mon-Sun)

    def reset(self) -> Tuple[str, str]: 
        self.day = 0
        return self.current_state()

    def step(self, price_action: float) -> Tuple[Tuple[str, str], float]: # <<< MODIFIED: price_action, Return type
        # price_action is an actual price value chosen from self.action_space
        demand = self._demand(price_action)
        reward = (price_action - self.product.cost) * demand
        self.day = (self.day + 1) % 7 # Cycle through days of the week
        return self.current_state(), reward

    def current_state(self) -> Tuple[str, str]: # <<< MODIFIED: Return type
        day_type = 'weekend' if self.day >= 5 else 'weekday' # Assuming 0-4 are weekdays, 5-6 are weekend
        return self.product.product_id, day_type

    def _demand(self, price: float) -> float:
        p = self.product
        if p.base_price == 0: # Avoid division by zero
            return p.base_demand 
        
        demand_change_factor = p.elasticity * (price - p.base_price) / p.base_price
        demand = p.base_demand + (p.base_demand * demand_change_factor)
        return max(demand, 0.0)

    def get_demand_at_price(self, price: float) -> float:
        """Calculates demand for the current product at a given price using its defined parameters."""
        return self._demand(price)
