"""Generate tables and figures."""
import numpy as np
from pathlib import Path
from src.environment.pricing_environment import Product, PricingEnvironment
from src.agents.q_learning_agent import QLearningAgent
from src.traditional.scipy_optimizer import ProductParams, optimize_price
from src.utils.visualization import plot_revenue_curve
from src.config import settings


def main():
    product = Product(
        base_price=settings.BASE_PRICE,
        base_demand=settings.BASE_DEMAND,
        elasticity=settings.ELASTICITY,
        cost=settings.COST,
    )
    env = PricingEnvironment(product, price_grid=np.array(settings.PRICE_GRID))
    agent = QLearningAgent(env.action_space, settings.ALPHA, settings.GAMMA, settings.EPSILON)
    for _ in range(settings.EPISODES):
        agent.train_episode(env)

    prices = np.array(settings.PRICE_GRID)
    revenues = [(p - settings.COST) * product.base_demand for p in prices]
    plot_revenue_curve(prices, revenues, Path('results/figures/revenue_curve.png'))

    params = ProductParams(
        base_price=settings.BASE_PRICE,
        base_demand=settings.BASE_DEMAND,
        elasticity=settings.ELASTICITY,
        cost=settings.COST,
    )
    optimal_price = optimize_price(params, (min(prices), max(prices)))
    print(f"Optimal price via scipy: {optimal_price}")

if __name__ == "__main__":
    main()
