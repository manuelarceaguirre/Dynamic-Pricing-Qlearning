"""Retail pricing simulation environment."""
from dataclasses import dataclass
from typing import Tuple
import numpy as np

@dataclass
class Product:
    base_price: float
    base_demand: float
    elasticity: float
    cost: float

class PricingEnvironment:
    """Environment for dynamic pricing."""
    def __init__(self, product: Product, price_grid: np.ndarray):
        self.product = product
        self.price_grid = price_grid
        self.state_space = ['weekday', 'weekend']
        self.action_space = price_grid
        self.day = 0

    def reset(self) -> Tuple[str, float]:
        self.day = 0
        return self.current_state()

    def step(self, price: float) -> Tuple[Tuple[str, float], float]:
        demand = self._demand(price)
        reward = (price - self.product.cost) * demand
        self.day = (self.day + 1) % 7
        return self.current_state(), reward

    def current_state(self) -> Tuple[str, float]:
        day_type = 'weekend' if self.day in [5, 6] else 'weekday'
        return day_type, self.product.base_price

    def _demand(self, price: float) -> float:
        p = self.product
        demand = p.base_demand + (p.base_demand * p.elasticity * (price - p.base_price) / p.base_price)
        return max(demand, 0.0)
