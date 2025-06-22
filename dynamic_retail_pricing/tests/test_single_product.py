import unittest
from dynamic_retail_pricing.main import train_product_q_learning
from dynamic_retail_pricing.src.config import (
    PRODUCTS_DATA,
    Q_LEARNING_CONFIG,
)


class TestSingleProduct(unittest.TestCase):
    def test_training_runs(self):
        product = PRODUCTS_DATA[0]
        results = train_product_q_learning(product, Q_LEARNING_CONFIG)
        self.assertIn('optimal_price', results)
        self.assertIn('optimal_demand', results)


if __name__ == '__main__':
    unittest.main()
