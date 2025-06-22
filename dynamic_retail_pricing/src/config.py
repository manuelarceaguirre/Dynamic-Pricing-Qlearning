# Q-Learning configuration and product data

Q_LEARNING_CONFIG = {
    'learning_rate': 0.1,
    'discount_factor': 0.6,
    'epsilon_start': 1.0,
    'epsilon_min': 0.01,
    'epsilon_decay': 0.001,
    'num_episodes': 5000
}

# Cost assumption: variable cost is 40% of base price
COST_MULTIPLIER = 0.4

# Action space configuration
PRICE_RANGE_MULTIPLIER = 0.5  # +/-50%
NUM_PRICE_STEPS = 21

PRODUCTS_DATA = [
    {"name": "Samsung 24\" HD", "price_elasticity": -0.5, "price": 109.2, "demand": 80.0},
    {"name": "Samsung 55\" 4K", "price_elasticity": -1.7, "price": 674.3, "demand": 54.0},
    {"name": "Hisense 65\" 4K", "price_elasticity": -1.1, "price": 1412.1, "demand": 49.0},
    {"name": "Samsung 40\" FHD", "price_elasticity": -0.7, "price": 260.5, "demand": 67.0},
    {"name": "Samsung 49\" 4K MU6290", "price_elasticity": -0.3, "price": 444.7, "demand": 57.0},
    {"name": "Samsung 49\" 4K Q6F", "price_elasticity": -4.4, "price": 829.0, "demand": 97.0},
    {"name": "Samsung 50\" FHD", "price_elasticity": -0.8, "price": 418.4, "demand": 56.0},
    {"name": "Samsung 55\" 4K Q8F", "price_elasticity": -8.4, "price": 2011.6, "demand": 60.0},
    {"name": "Samsung 65\" 4K Q7F", "price_elasticity": -7.8, "price": 2411.6, "demand": 60.0},
    {"name": "Samsung 24\" HD UN24H4500", "price_elasticity": -1.9, "price": 142.7, "demand": 40.0},
    {"name": "Sony 40\" FHD", "price_elasticity": -0.8, "price": 423.8, "demand": 27.0},
    {"name": "Sony 43\" 4K UHD", "price_elasticity": -5.6, "price": 648.0, "demand": 154.0},
    {"name": "VIZIO 39\" FHD", "price_elasticity": -1.8, "price": 249.8, "demand": 59.0},
    {"name": "VIZIO 70\" 4K XHDR", "price_elasticity": -6.5, "price": 1300.0, "demand": 36.0}
]
