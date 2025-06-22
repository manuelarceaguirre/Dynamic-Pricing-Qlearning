# THIS MUST BE AT THE ABSOLUTE TOP OF scripts/generate_results.py (or your chosen script name)
import sys
import os
import inspect 

print(f"--- Script Start ---")
print(f"Current working directory: {os.getcwd()}")
print(f"Initial sys.path: {sys.path}")

current_script_path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
print(f"Script directory: {current_script_path}")

project_root = os.path.abspath(os.path.join(current_script_path, '..'))
print(f"Calculated project root: {project_root}")

if project_root not in sys.path:
    sys.path.insert(0, project_root)
    print(f"SUCCESS: Project root '{project_root}' ADDED to sys.path.")
    print(f"Modified sys.path: {sys.path}")
else:
    print(f"INFO: Project root '{project_root}' already in sys.path.")
print(f"--- End Path Modification ---")

import numpy as np
import pandas as pd
from pathlib import Path 

import sys
import os
import inspect 

print(f"--- Script Start ---")
print(f"Current working directory: {os.getcwd()}")
print(f"Initial sys.path: {sys.path}")

current_script_path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
print(f"Script directory: {current_script_path}")

project_root = os.path.abspath(os.path.join(current_script_path, '..'))
print(f"Calculated project root: {project_root}")

if project_root not in sys.path:
    sys.path.insert(0, project_root)
    print(f"SUCCESS: Project root '{project_root}' ADDED to sys.path.")
    print(f"Modified sys.path: {sys.path}")
else:
    print(f"INFO: Project root '{project_root}' already in sys.path.")
print(f"--- End Path Modification ---")


from src.environment.pricing_environment import Product, PricingEnvironment
from src.agents.q_learning_agent import QLearningAgent
from src.traditional.scipy_optimizer import ProductParams, optimize_price
from src.traditional.scipy_optimizer import demand as scipy_demand_func

# --- Configuration ---
# Q-Learning Hyperparameters
ALPHA = 0.1
GAMMA = 0.90
INITIAL_EPSILON = 1.0
MIN_EPSILON = 0.01
NUM_EPISODES = 5000 
EPSILON_DECAY_RATE = -np.log(0.05) / (NUM_EPISODES * 0.8) 

# RL Action Space Configuration
NUM_PRICE_POINTS_RL = 21
PRICE_RANGE_PERCENT_RL = 0.35 #
# Cost Assumption
ASSUMED_COST_PERCENTAGE_OF_BASE_PRICE = 0.18 
# SciPy Optimizer settings
SCIPY_PRICE_BOUND_LOWER_FACTOR = 0.1 
SCIPY_PRICE_BOUND_UPPER_FACTOR = 3.0 

paper_products_data_table_i = [
    {"Product Name": "Samsung 24” HD", "Price Elasticity": -0.5, "Base Price": 109.2, "Base Demand": 80.0},
    {"Product Name": "Samsung 55” 4K", "Price Elasticity": -1.7, "Base Price": 674.3, "Base Demand": 54.0},
    {"Product Name": "Hisense 65” 4K", "Price Elasticity": -1.1, "Base Price": 1412.1, "Base Demand": 49.0},
    {"Product Name": "Samsung 40” FHD", "Price Elasticity": -0.7, "Base Price": 260.5, "Base Demand": 67.0},
    {"Product Name": "Samsung 49” 4K MU6290", "Price Elasticity": -0.3, "Base Price": 444.7, "Base Demand": 57.0},
    {"Product Name": "Samsung 49” 4K Q6F", "Price Elasticity": -4.4, "Base Price": 829.0, "Base Demand": 97.0},
    {"Product Name": "Samsung 50” FHD", "Price Elasticity": -0.8, "Base Price": 418.4, "Base Demand": 56.0},
    {"Product Name": "Samsung 55” 4K Q8F", "Price Elasticity": -8.4, "Base Price": 2011.6, "Base Demand": 60.0},
    {"Product Name": "Samsung 65” 4K Q7F", "Price Elasticity": -7.8, "Base Price": 2411.6, "Base Demand": 60.0},
    {"Product Name": "Samsung 24” HD UN24H4500", "Price Elasticity": -1.9, "Base Price": 142.7, "Base Demand": 40.0},
    {"Product Name": "Sony 40” FHD", "Price Elasticity": -0.8, "Base Price": 423.8, "Base Demand": 27.0},
    {"Product Name": "Sony 43” 4K UHD", "Price Elasticity": -5.6, "Base Price": 648.0, "Base Demand": 154.0},
    {"Product Name": "VIZIO 39” FHD", "Price Elasticity": -1.8, "Base Price": 249.8, "Base Demand": 59.0},
    {"Product Name": "VIZIO 70” 4K XHDR", "Price Elasticity": -6.5, "Base Price": 1300.0, "Base Demand": 36.0},
]

def main():
    results_rl_list = []
    results_scipy_list = []

    print(f"Starting experiments with {NUM_EPISODES} Q-learning episodes per product.")
    print(f"Assumed Cost: {ASSUMED_COST_PERCENTAGE_OF_BASE_PRICE*100:.1f}% of Base Price.") 
    print(f"RL Action Space: {NUM_PRICE_POINTS_RL} points within +/- {PRICE_RANGE_PERCENT_RL*100:.1f}% of Base Price.") #    print(f"Epsilon decay rate calculated for target by ~80% of episodes: {EPSILON_DECAY_RATE:.5f}\n")


    for prod_data in paper_products_data_table_i:
        product_name = prod_data["Product Name"]
        base_price = prod_data["Base Price"]
        base_demand = prod_data["Base Demand"]
        elasticity = prod_data["Price Elasticity"]
        
        assumed_cost = base_price * ASSUMED_COST_PERCENTAGE_OF_BASE_PRICE

        print(f"--- Processing Product: {product_name} ---")
        print(f"  Base Price: ${base_price:.2f}, Base Demand: {base_demand:.1f}, Elasticity: {elasticity:.1f}, Assumed Cost: ${assumed_cost:.2f}")

        min_rl_action_price = max(0.01, base_price * (1 - PRICE_RANGE_PERCENT_RL)) 
        max_rl_action_price = base_price * (1 + PRICE_RANGE_PERCENT_RL)
        rl_action_prices = np.linspace(min_rl_action_price, max_rl_action_price, NUM_PRICE_POINTS_RL)

        current_product_obj = Product(
            product_id=product_name,
            base_price=base_price,
            base_demand=base_demand,
            elasticity=elasticity,
            cost=assumed_cost
        )
        env = PricingEnvironment(product=current_product_obj, price_grid=rl_action_prices)
        
        agent = QLearningAgent(
            actions=rl_action_prices, 
            alpha=ALPHA,
            gamma=GAMMA,
            initial_epsilon=INITIAL_EPSILON,
            min_epsilon=MIN_EPSILON,
            epsilon_decay_rate=EPSILON_DECAY_RATE
        )

        print(f"  Training Q-Learning Agent...")
        for episode_num in range(NUM_EPISODES):
            agent.train_episode(env)
            agent.decay_epsilon(episode_num) 
            if (episode_num + 1) % (NUM_EPISODES // 20) == 0: 
                 print(f"\r    Episode {episode_num + 1}/{NUM_EPISODES} complete. Epsilon: {agent.epsilon:.4f}", end="")
        print("\r    Training complete.                                                     ")

        eval_state_rl = (product_name, 'weekday') 
        optimal_price_rl = agent.choose_action(eval_state_rl, exploit_only=True)
        optimal_demand_rl = env.get_demand_at_price(optimal_price_rl)
        # <<< MODIFICATION: Calculate RL Profit >>>
        profit_rl = (optimal_price_rl - assumed_cost) * optimal_demand_rl
        
        results_rl_list.append({
            'Product Name': product_name,
            'Optimal Price': optimal_price_rl,
            'Optimal Demand': optimal_demand_rl,
            'Est. Profit': profit_rl # Added profit to results list
        })
        print(f"  RL Optimal Price: ${optimal_price_rl:.2f}, Est. Demand: {optimal_demand_rl:.2f}, Est. Profit: ${profit_rl:.2f}")

        scipy_product_params = ProductParams(
            base_price=base_price,
            base_demand=base_demand,
            elasticity=elasticity,
            cost=assumed_cost
        )
        scipy_price_bound_low = max(0.01, base_price * SCIPY_PRICE_BOUND_LOWER_FACTOR)
        scipy_price_bound_high = base_price * SCIPY_PRICE_BOUND_UPPER_FACTOR
        
        optimal_price_scipy = optimize_price(scipy_product_params, (scipy_price_bound_low, scipy_price_bound_high))
        optimal_demand_scipy = scipy_demand_func(optimal_price_scipy, scipy_product_params)
        profit_scipy = (optimal_price_scipy - assumed_cost) * optimal_demand_scipy

        results_scipy_list.append({
            'Product Name': product_name,
            'Optimal Price': optimal_price_scipy,
            'Optimal Demand': optimal_demand_scipy,
            'Est. Profit': profit_scipy         })
        print(f"  SciPy Optimal Price: ${optimal_price_scipy:.2f}, Est. Demand: {optimal_demand_scipy:.2f}, Est. Profit: ${profit_scipy:.2f}")
        print("-" * 30)

    df_rl_results = pd.DataFrame(results_rl_list)
    df_scipy_results = pd.DataFrame(results_scipy_list)

    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    pd.set_option('display.colheader_justify', 'center')
    pd.set_option('display.precision', 2)

    print("\n\n" + "="*20 + " Reinforcement Learning Optimized Prices & Profit " + "="*20)
    print(df_rl_results[['Product Name', 'Optimal Price', 'Optimal Demand', 'Est. Profit']].to_string(index=False))

    print("\n\n" + "="*20 + " Traditional Optimization with SciPy & Profit " + "="*20)
    print(df_scipy_results[['Product Name', 'Optimal Price', 'Optimal Demand', 'Est. Profit']].to_string(index=False))


if __name__ == "__main__":
    main()
