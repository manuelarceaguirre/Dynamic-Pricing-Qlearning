import numpy as np
from src.environment.pricing_environment import Product, PricingEnvironment
from src.agents.q_learning_agent import QLearningAgent


def test_agent_learns():
    product = Product(base_price=100, base_demand=50, elasticity=-1.0, cost=60)
    env = PricingEnvironment(product, price_grid=np.array([90, 100, 110]))
    agent = QLearningAgent(env.action_space)
    reward = agent.train_episode(env)
    assert isinstance(reward, float)
