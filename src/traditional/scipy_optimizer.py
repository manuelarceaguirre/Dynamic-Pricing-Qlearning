"""Traditional optimization using SciPy."""
from dataclasses import dataclass
import numpy as np
from scipy.optimize import minimize

@dataclass
class ProductParams:
    base_price: float
    base_demand: float
    elasticity: float
    cost: float


def demand(price: float, params: ProductParams) -> float:
    d = params.base_demand + (
        params.base_demand * params.elasticity * (price - params.base_price) / params.base_price
    )
    return max(d, 0.0)


def revenue(price: float, params: ProductParams) -> float:
    d = demand(price, params)
    return (price - params.cost) * d


def optimize_price(params: ProductParams, price_bounds: tuple) -> float:
    result = minimize(
        lambda p: -revenue(p[0], params),
        x0=[params.base_price],
        bounds=[price_bounds],
    )
    return float(result.x[0])
