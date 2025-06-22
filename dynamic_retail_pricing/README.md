# Dynamic Retail Pricing via Q-Learning

This package implements Phase 1 of a replication of the research paper "Dynamic Retail Pricing via Q-Learning" (Apte et al., 2024).

Key assumptions:
- Variable cost is assumed to be 40% of the base price for all products.
- Action space consists of 21 discrete prices ranging from 50% below to 50% above the base price.

Run `python main.py` to train the Q-learning agent on the Samsung 24" HD product.
