"""Run a full training experiment."""
import numpy as np
from src.environment.pricing_environment import Product, PricingEnvironment
from src.agents.q_learning_agent import QLearningAgent
from src.config import settings


def main(episodes: int = settings.EPISODES):
    product = Product(
        base_price=settings.BASE_PRICE,
        base_demand=settings.BASE_DEMAND,
        elasticity=settings.ELASTICITY,
        cost=settings.COST,
    )
    env = PricingEnvironment(product, price_grid=np.array(settings.PRICE_GRID))
    agent = QLearningAgent(env.action_space, settings.ALPHA, settings.GAMMA, settings.EPSILON)

    for _ in range(episodes):
        agent.train_episode(env)

    print("Training complete")

if __name__ == "__main__":
    main()
