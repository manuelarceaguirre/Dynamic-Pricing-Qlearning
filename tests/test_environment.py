import numpy as np
from src.environment.pricing_environment import Product, PricingEnvironment


def test_demand_positive():
    product = Product(base_price=100, base_demand=50, elasticity=-1.0, cost=60)
    env = PricingEnvironment(product, price_grid=np.array([100]))
    _, reward = env.step(100)
    assert reward >= 0
