Okay, here's the updated `README.md` incorporating the structure for documenting your replication attempt, using placeholders for where your latest tables would go. I've removed contact information and kept it focused on the technical aspects of the replication.

Remember to replace the placeholder comments like `<!-- PASTE YOUR RL TABLE HERE -->` with your actual output once you have a set of results you want to document.

```markdown
# Dynamic Retail Pricing via Q-Learning - A Replication Attempt

This repository contains an attempt to implement and reproduce the experiments from the paper **"Dynamic Retail Pricing via Q-Learning - A Reinforcement Learning Framework for Enhanced Revenue Management"** by Apte et al. (arXiv:2411.18261v1).

## Original Paper Abstract (for reference)
This paper explores the application of a reinforcement learning (RL) framework using the Q-Learning algorithm to enhance dynamic pricing strategies in the retail sector. Unlike traditional pricing methods, which often rely on static demand models, our RL approach continuously adapts to evolving market dynamics, offering a more flexible and responsive pricing strategy. By creating a simulated retail environment, we demonstrate how RL effectively addresses real-time changes in consumer behavior and market conditions, leading to improved revenue outcomes. Our results illustrate that the RL model not only surpasses traditional methods in terms of revenue generation but also provides insights into the complex interplay of price elasticity and consumer demand. This research underlines the significant potential of applying artificial intelligence in economic decision-making, paving the way for more sophisticated, data-driven pricing models in various commercial domains.

## Replication Goal
The primary goal of this project is to replicate the core Q-learning dynamic pricing mechanism described in the paper and to compare its performance against a traditional optimization method, aiming to reproduce results similar to those presented in Tables II and III of the paper.

## Key Features of this Implementation
- Implementation of a retail pricing environment based on the demand function (Eq. 1) and reward function (Eq. 3) from the paper.
- A Q-learning agent with epsilon-greedy exploration and configurable hyperparameters.
- A baseline traditional optimization method using `scipy.optimize.minimize_scalar` to maximize profit given the demand function.
- Scripts to run experiments for multiple products as listed in Table I of the paper.
- Calculation of price, demand, and profit for both RL and traditional optimization methods.

## Installation
1.  Clone this repository.
2.  (Recommended) Create and activate a Python virtual environment.
    ```bash
    python -m venv venv
    # On Windows:
    venv\Scripts\activate
    # On macOS/Linux:
    source venv/bin/activate
    ```
3.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
4.  (Optional, for making `src` importable project-wide when the venv is active) Install the project in editable mode:
    ```bash
    pip install -e .
    ```

## Running the Replication Experiment
To run the main experiment which processes all products from Table I of the paper and generates comparative results:
Navigate to the project root directory in your terminal and run:
```bash
python scripts/generate_results.py
```
The script will output the optimal prices, demands, and profits found by both the Q-learning agent and the SciPy optimizer for each product.

## Methodology of this Replication

### Environment
-   **State Space (Paper: III-D):** Defined as `(product_id, day_type)`, where `day_type` is 'weekday' or 'weekend'. In this implementation, the `day_type` cycles but does not currently alter the product's base demand or elasticity.
-   **Action Space (Paper: III-D):** A discrete set of possible prices. This implementation generates `N` price points for the RL agent, typically ranging from `Base Price * (1 - X%)` to `Base Price * (1 + X%)`.
-   **Demand Function (Paper: Eq. 1):**
    `Demand = Base_Demand + (Base_Demand * Elasticity * (Price - Base_Price) / Base_Price)`
-   **Reward Function (Paper: Eq. 3):**
    `Reward = (Price * Demand) - (Cost * Demand)`

### Q-Learning Agent (Paper: III-C)
-   Uses the standard Q-learning update rule (Eq. 2).
-   Employs an epsilon-greedy policy for action selection, with epsilon decaying over episodes.
-   Key hyperparameters are configurable in `scripts/generate_results.py`.

### Traditional Optimization (Paper: Table III)
-   Uses `scipy.optimize.minimize_scalar` to find the price that maximizes the profit function derived from the demand equation, given a unit cost.
-   The profit function is `Profit(Price) = (Price - Cost) * Demand(Price)`.

## Replication Configuration & Assumptions

This section details the specific parameters and assumptions made during this replication attempt, as some details were not fully specified in the original paper.

### 1. Product Costs (Paper: III-B, not quantified in Table I)
-   The paper mentions "Costs: Variable costs associated with production" but does not provide specific cost values for products in Table I.
-   **Assumption Made:** In this replication, unit cost for each product is assumed to be a fixed percentage of its `Base Price`.
-   **Current Value Used for Documented Results:** `ASSUMED_COST_PERCENTAGE_OF_BASE_PRICE = 0.18` (i.e., 18% of Base Price).
    *(Note: This value was iteratively tuned from initial values like 60%, 40%, 25%, 10%. An 18% cost was chosen as one of the values explored during the tuning process. Perfect alignment across all products with the paper's Table III (SciPy) using a single cost percentage proved challenging, suggesting product-specific costs or different optimization objectives/constraints in the original paper.)*

### 2. Q-Learning Agent Hyperparameters (Paper: III-C, general)
-   **Learning Rate (`ALPHA`):** `0.1`
-   **Discount Factor (`GAMMA`):** `0.90`
-   **Initial Epsilon (`INITIAL_EPSILON`):** `1.0`
-   **Minimum Epsilon (`MIN_EPSILON`):** `0.01`
-   **Epsilon Decay Rate (`EPSILON_DECAY_RATE`):** Calculated as `-np.log(0.05) / (NUM_EPISODES * 0.8)`, resulting in `approx 0.00075` for 5000 episodes.
-   **Number of Episodes (`NUM_EPISODES`):** `5000` per product.

### 3. RL Agent Action Space (Paper: III-D, "set of possible prices")
-   **Number of Discrete Price Points (`NUM_PRICE_POINTS_RL`):** `21`
-   **Price Range (`PRICE_RANGE_PERCENT_RL`):** `Base Price +/- 35%`.
    *(Note: The `generate_results.py` script includes commented-out examples for product-specific action space tuning, which was explored for certain products like "Samsung 24” HD" during the tuning process.)*

### 4. SciPy Optimizer Configuration
-   **Objective:** Maximize `(Price - Cost) * Demand(Price)`.
-   **Bounds:** `Base Price * 0.1` (lower) to `Base Price * 3.0` (upper).

## Current Replication Status & Results Summary

This replication attempt focused on implementing the core components described in the paper and iteratively tuning key assumptions (primarily unit cost and RL action space) to understand the model's behavior and sensitivities in relation to the published results.

**Key Findings:**

*   **Impact of Cost Assumption:** The assumed unit cost is a highly sensitive parameter. Different cost assumptions significantly alter the optimal prices and demands found by both the RL agent and the SciPy optimizer. The inability to precisely match the paper's Table III (Traditional Optimization) across all products with a single cost percentage suggests that the original study likely used product-specific costs or had a more complex setup for their traditional optimization baseline. Attempts to reverse-engineer a single positive cost percentage to match Table III for specific products (e.g., "Samsung 24” HD") were analytically inconsistent with simple profit maximization.

*   **RL Agent Behavior:** The Q-learning agent successfully learns pricing policies. With the chosen parameters (e.g., 18% cost, +/-35% RL price range), the agent often identifies the most profitable price within its discrete action set. In some cases (e.g., "Samsung 24” HD"), this led to the selection of the maximum available price in its action range, which, under the assumed cost, was indeed more profitable than the paper's reported RL price for that product. This indicates that the paper's RL agent might have operated with different cost assumptions, a more constrained action space, or its convergence/exploration resulted in a different local optimum.

*   **RL vs. Traditional Optimization (This Implementation):** Under the static environment conditions and consistent cost assumptions used in this replication, the SciPy (traditional) optimizer generally achieved slightly higher profits than the Q-learning agent. This is theoretically expected, as SciPy can find a continuous optimum, while Q-learning is limited by its discrete action space. The paper's assertion that RL surpasses traditional methods likely relies on the "evolving market dynamics" which were not explicitly detailed for replication and thus not implemented in the current static environment. Introducing such dynamics would be a key area where RL's adaptive nature could demonstrate superiority.

**Challenges in Replication:**
*   **Unspecified Product Costs:** This is the most significant challenge for precise numerical replication.
*   **Ambiguity in "Traditional Optimization" Setup:** The exact bounds, constraints, or potential nuances of the objective function for the paper's SciPy optimization are not fully detailed.
*   **Lack of Detail on "Evolving Market Dynamics":** The specific mechanisms of environmental changes in the paper's simulation are not provided, making it difficult to replicate the conditions where RL's adaptiveness would be most pronounced.

### Generated Results (Cost: 18% of Base, RL Action Space: Base +/- 35%)

**Reinforcement Learning Optimized Prices & Profit:**
```
==================== Reinforcement Learning Optimized Prices & Profit ====================
      Product Name        Optimal Price  Optimal Demand  Est. Profit
          Samsung 24” HD     147.42           66.00        8432.42
          Samsung 55” 4K     603.50           63.64       30681.92
          Hisense 65” 4K    1461.52           47.11       56882.27
         Samsung 40” FHD     342.56           52.23       15441.68
   Samsung 49” 4K MU6290     600.35           51.02       26543.05
      Samsung 49” 4K Q6F     596.88          216.50       96920.18
         Samsung 50” FHD     506.26           46.59       20078.92
      Samsung 55” 4K Q8F    1307.54          236.40      223504.85
      Samsung 65” 4K Q7F    1567.54          223.80      253666.56
Samsung 24” HD UN24H4500     122.72           50.64        4913.90
            Sony 40” FHD     512.80           22.46        9805.85
         Sony 43” 4K UHD     443.88          425.66      139291.67
           VIZIO 39” FHD     214.83           73.87       12547.51
       VIZIO 70” 4K XHDR     845.00          117.90       72036.90
```

**Traditional Optimization with SciPy & Profit:**
```
==================== Traditional Optimization with SciPy & Profit ====================
      Product Name        Optimal Price  Optimal Demand  Est. Profit
          Samsung 24” HD     173.63           56.40        8684.02
          Samsung 55” 4K     596.16           64.64       30689.25
          Hisense 65” 4K    1475.00           46.60       56889.21
         Samsung 40” FHD     339.77           52.73       15443.08
   Samsung 49” 4K MU6290    1003.54           35.51       32794.18
      Samsung 49” 4K Q6F     583.31          223.49       97014.92
         Samsung 50” FHD     508.36           46.37       20079.38
      Samsung 55” 4K Q8F    1306.58          236.64      223505.08
      Samsung 65” 4K Q7F    1577.43          221.88      253685.55
Samsung 24” HD UN24H4500     121.75           51.16        4914.41
            Sony 40” FHD     514.92           22.36        9806.08
         Sony 43” 4K UHD     440.18          430.58      139309.92
           VIZIO 39” FHD     216.77           73.04       12549.12
       VIZIO 70” 4K XHDR     867.00          113.94       72124.02
```

## Repository Structure
-   `src/`: Contains the Python source code for the environment, agent, and optimizer.
-   `scripts/`: Contains scripts to run experiments and generate results (e.g., `generate_results.py`).
-   `data/`: Contains product data (derived from paper's Table I) used in `generate_results.py`, and the example `electronic_products_pricing.csv`.
-   `notebooks/`: Jupyter notebooks for exploration and detailed analysis (currently placeholders).
-   *(Add other directories like `docs/` or `results/` if you use them).*

## Dependencies
See `requirements.txt`. Key dependencies include:
- `numpy`
- `pandas`
- `scipy`

## Original Paper Citation
```
@misc{apte2024dynamic,
      title={Dynamic Retail Pricing via Q-Learning - A Reinforcement Learning Framework for Enhanced Revenue Management}, 
      author={Mohit Apte and Ketan Kale and Pranav Datar and P. R. Deshmukh},
      year={2024},
      eprint={2411.18261},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      license={CC BY 4.0}
}
```

