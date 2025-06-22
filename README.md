# Dynamic Retail Pricing via Q-Learning - A Replication Attempt

This repository contains an attempt to implement and reproduce the experiments from the paper **"Dynamic Retail Pricing via Q-Learning - A Reinforcement Learning Framework for Enhanced Revenue Management"** by Apte et al. (arXiv:2411.18261v1).

## Original Paper Abstract (for reference)
This paper explores the application of a reinforcement learning (RL) framework using the Q-Learning algorithm to enhance dynamic pricing strategies in the retail sector. Unlike traditional pricing methods, which often rely on static demand models, our RL approach continuously adapts to evolving market dynamics, offering a more flexible and responsive pricing strategy. By creating a simulated retail environment, we demonstrate how RL effectively addresses real-time changes in consumer behavior and market conditions, leading to improved revenue outcomes. Our results illustrate that the RL model not only surpasses traditional methods in terms of revenue generation but also provides insights into the complex interplay of price elasticity and consumer demand. This research underlines the significant potential of applying artificial intelligence in economic decision-making, paving the way for more sophisticated, data-driven pricing models in various commercial domains.

## Replication Goal
The primary goal of this project is to replicate the core Q-learning dynamic pricing mechanism described in the paper and to compare its performance against a traditional optimization method, aiming to reproduce results similar to those presented in Tables II and III of the paper.

## Key Features of this Implementation
- Implementation of a retail pricing environment based on the demand function (Eq. 1) and reward function (Eq. 3) from the paper.
- A Q-learning agent with epsilon-greedy exploration and configurable hyperparameters.
- A baseline traditional optimization method using `scipy.optimize` to maximize profit given the demand function.
- Scripts to run experiments for multiple products as listed in Table I of the paper.
- Calculation of price, demand, and profit for both RL and traditional optimization methods.

## Installation
1.  Clone this repository:
    ```bash
    git clone <your-repo-url>
    cd <your-repo-name>
    ```
2.  (Recommended) Create and activate a Python virtual environment:
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
4.  (Optional, for making `src` importable from anywhere in the project when the venv is active) Install the project in editable mode:
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
-   **State Space (Paper: III-D):** Defined as `(product_id, day_type)`, where `day_type` is 'weekday' or 'weekend'. In this implementation, the `day_type` cycles but does not currently alter the product's base demand or elasticity unless explicitly modified in the `PricingEnvironment`.
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
-   Uses `scipy.optimize.minimize_scalar` (or `minimize`) to find the price that maximizes the profit function derived from the demand equation, given a unit cost.
-   The profit function is `Profit(Price) = (Price - Cost) * Demand(Price)`.

## Replication Configuration & Assumptions

This section details the specific parameters and assumptions made during this replication attempt, as some details were not fully specified in the original paper.

### 1. Product Costs (Paper: III-B, not quantified in Table I)
-   The paper mentions "Costs: Variable costs associated with production" but does not provide specific cost values for products in Table I.
-   **Assumption Made:** In this replication, unit cost for each product is assumed to be a fixed percentage of its `Base Price`.
-   **Current Value:** `ASSUMED_COST_PERCENTAGE_OF_BASE_PRICE = 0.18` (i.e., 18% of Base Price).
    *(Note: This value was iteratively tuned. Previous attempts included 60%, 40%, 25%, and 10%. An 18% cost brought the SciPy optimization results for some products closer to the paper's Table III, but perfect alignment across all products with a single percentage proved challenging, suggesting product-specific costs or different optimization objectives in the original paper.)*

### 2. Q-Learning Agent Hyperparameters (Paper: III-C, general)
-   **Learning Rate (`ALPHA`):** `0.1`
-   **Discount Factor (`GAMMA`):** `0.90`
-   **Initial Epsilon (`INITIAL_EPSILON`):** `1.0`
-   **Minimum Epsilon (`MIN_EPSILON`):** `0.01`
-   **Epsilon Decay Rate (`EPSILON_DECAY_RATE`):** Calculated to make epsilon reach `MIN_EPSILON` around 80% of total episodes. Current: `approx 0.00075` for 5000 episodes.
-   **Number of Episodes (`NUM_EPISODES`):** `5000` per product.

### 3. RL Agent Action Space (Paper: III-D, "set of possible prices")
-   **Number of Discrete Price Points (`NUM_PRICE_POINTS_RL`):** `21`
-   **Price Range (`PRICE_RANGE_PERCENT_RL`):** Generally `Base Price +/- 35%`.
    *(Note: For some specific products like "Samsung 24” HD", this was experimented with (e.g., +/-30% with 25 points) to try and match the paper's RL results more closely. The `generate_results.py` script currently uses a general setting but includes commented-out examples for product-specific action space tuning.)*

### 4. SciPy Optimizer Configuration
-   **Objective:** Maximize `(Price - Cost) * Demand(Price)`.
-   **Bounds:** `Base Price * 0.1` to `Base Price * 3.0`.

## Current Replication Status & Results Summary
*(This is where you'll summarize your findings once you settle on a final set of parameters or decide on the scope of your replication.)*

**Example Draft (update this as you progress):**

> After several iterations of tuning the assumed cost percentage and RL agent parameters, the current configuration (18% cost margin, RL prices +/-35% of base) yields the following general observations:
>
> *   **Comparison with Paper's Table II (RL Results):**
>     *   For some products (e.g., "Samsung 55” 4K"), our RL agent finds prices and demands that are in a plausible range compared to the paper, although numerical matches are not exact.
>     *   For other products (e.g., "Samsung 24” HD"), our RL agent consistently chooses the highest price in its allowed action space, which is more profitable under our cost assumption than the paper's reported RL price for that product. This suggests that either the paper's true cost for this item was significantly different, their RL action space was more constrained, or their agent did not fully converge to the maximum profit point available to it.
>
> *   **Comparison with Paper's Table III (Traditional Optimization):**
>     *   Our SciPy optimization results, using the derived profit maximization formula, differ from the paper's Table III values. Attempts to match Table III for specific products (like "Samsung 24” HD") by solely adjusting a uniform cost percentage led to analytically inconsistent results (e.g., requiring negative costs). This suggests the paper's "Traditional Optimization" might have involved product-specific costs, different optimization bounds, or a slightly different objective/model than a simple unconstrained profit maximization based on the provided demand function and a positive cost.
>
> *   **RL vs. Traditional Optimization (Our Implementation):**
>     *   Under the current static environment and cost assumptions, our SciPy (traditional) optimizer generally finds slightly higher profit margins than our Q-learning agent. This is expected, as SciPy can find a continuous optimum, while Q-learning operates on a discrete action space.
>     *   The paper's claim that RL surpasses traditional methods likely hinges on the "evolving market dynamics" and "real-time changes" mentioned in their abstract, which are not explicitly detailed for replication and not yet implemented in this static version of the environment. If the environment parameters (base demand, elasticity) were to fluctuate, an adaptive RL agent could indeed show superior performance over a statically-optimized traditional model.
>
> **Challenges in Replication:**
> *   **Unspecified Costs:** The primary challenge is the lack of specific cost data for each product in the original paper.
> *   **Ambiguity in "Traditional Optimization":** The exact setup (bounds, constraints, objective nuances) of the SciPy optimization used for Table III is not fully detailed.
> *   **Details of "Evolving Market Dynamics":** The mechanism by which the market evolves in the paper's simulation is not specified, making it difficult to replicate the conditions under which RL's adaptive capabilities would be most evident.
>
> *(You would then include your generated tables, or a summary/link to them.)*
>
> **Generated Profit Comparison (using 18% cost, +/-35% RL range):**
> *(Here, you'd paste or summarize your latest profit tables)*
>
> ```
> ==================== Reinforcement Learning Optimized Prices & Profit ====================
>       Product Name        Optimal Price  Optimal Demand  Est. Profit
>           Samsung 24” HD     147.42           66.00        8432.42
>           ... (rest of your RL table) ...
>
> ==================== Traditional Optimization with SciPy & Profit ====================
>       Product Name        Optimal Price  Optimal Demand  Est. Profit
>           Samsung 24” HD     173.63           56.40        8684.02
>           ... (rest of your SciPy table) ...
> ```

## Future Work / Potential Extensions for this Replication
-   Experiment with product-specific RL action space configurations.
-   Implement a version of the `PricingEnvironment` with stochastic or dynamic changes to `base_demand` or `elasticity` to test RL's adaptability as suggested by the paper.
-   Further investigate the SciPy optimization by trying different objectives (e.g., revenue maximization) or tight bounds to see if Table III can be better approximated.
-   Explore other RL algorithms mentioned as potentially suitable for dynamic pricing.

## Repository Structure
-   `src/`: Contains the Python source code for the environment, agent, and optimizer.
-   `scripts/`: Contains scripts to run experiments and generate results (e.g., `generate_results.py`).
-   `data/`: Contains raw data (from paper's Table I values used in `generate_results.py`, and the example `electronic_products_pricing.csv`).
-   `notebooks/`: Jupyter notebooks for exploration and detailed analysis (currently placeholders).
-   `docs/`: (If you add more detailed documentation).
-   `results/`: (If you save table/figure outputs here).

## Dependencies
See `requirements.txt`. Key dependencies include:
- `numpy`
- `pandas`
- `scipy`
- `matplotlib` (optional, for potential visualizations)

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

## License for this Replication Code
This project is licensed under the MIT License. See `LICENSE` for details.

## Contributing
Contributions or suggestions to improve this replication are welcome! Please open an issue or pull request.

