import numpy as np
from dynamic_retail_pricing.src.environment import RetailEnvironment, calculate_demand
from dynamic_retail_pricing.src.q_learning_agent import QLearningAgent
from dynamic_retail_pricing.src.config import (
    Q_LEARNING_CONFIG,
    COST_MULTIPLIER,
    PRICE_RANGE_MULTIPLIER,
    NUM_PRICE_STEPS,
    PRODUCTS_DATA,
)


def train_product_q_learning(product_data, config):
    base_demand = product_data['demand']
    base_price = product_data['price']
    elasticity = product_data['price_elasticity']
    cost_per_unit = COST_MULTIPLIER * base_price

    price_min = base_price * (1 - PRICE_RANGE_MULTIPLIER)
    price_max = base_price * (1 + PRICE_RANGE_MULTIPLIER)
    possible_prices = np.linspace(price_min, price_max, NUM_PRICE_STEPS).tolist()

    env = RetailEnvironment(base_demand, base_price, elasticity, cost_per_unit, possible_prices)
    agent = QLearningAgent(
        num_actions=len(possible_prices),
        learning_rate=config['learning_rate'],
        discount_factor=config['discount_factor'],
        epsilon=config['epsilon_start'],
        epsilon_min=config['epsilon_min'],
        epsilon_decay=config['epsilon_decay'],
    )

    print(f"Training Q-Learning for {product_data['name']}...")
    for episode in range(config['num_episodes']):
        action_index = agent.choose_action()
        reward, _ = env.step(action_index)
        agent.update_q_value(action_index, reward)
        agent.training_rewards.append(reward)
        agent.decay_epsilon()

        if episode % 1000 == 0:
            best_action = agent.get_optimal_action()
            best_price = env.get_price_from_action(best_action)
            print(f"  Episode {episode}: Current best price = ${best_price:.2f}")

    optimal_action = agent.get_optimal_action()
    optimal_price = env.get_price_from_action(optimal_action)
    optimal_demand = calculate_demand(optimal_price, base_demand, base_price, elasticity)

    return {
        'product_name': product_data['name'],
        'optimal_price': optimal_price,
        'optimal_demand': optimal_demand,
        'q_values': agent.q_table.copy(),
        'training_rewards': agent.training_rewards.copy(),
        'possible_prices': possible_prices.copy(),
    }


def test_single_product():
    samsung_24 = PRODUCTS_DATA[0]
    print("=== Testing Q-Learning Implementation ===")
    print(f"Product: {samsung_24['name']}")
    print(f"Base Price: ${samsung_24['price']}")
    print(f"Base Demand: {samsung_24['demand']}")
    print(f"Price Elasticity: {samsung_24['price_elasticity']}")
    print(f"Assumed Cost per Unit: ${COST_MULTIPLIER * samsung_24['price']:.2f}")

    results = train_product_q_learning(samsung_24, Q_LEARNING_CONFIG)

    print("\n=== Q-Learning Results ===")
    print(f"Optimal Price: ${results['optimal_price']:.1f}")
    print(f"Optimal Demand: {results['optimal_demand']:.1f}")

    print("\nExpected from paper Table II:")
    print("  Optimal Price: $139.6")
    print("  Optimal Demand: 68.2")

    return results


if __name__ == "__main__":
    test_results = test_single_product()
