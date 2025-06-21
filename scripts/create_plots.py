"""Create visualizations from results."""
from pathlib import Path
import pandas as pd
from src.utils.visualization import plot_revenue_curve


def main():
    df = pd.read_csv(Path('data/raw/electronic_products_pricing.csv'))
    prices = df['Base_Price']
    revenues = (df['Base_Price'] - df['Cost']) * df['Base_Demand']
    plot_revenue_curve(prices, revenues, Path('results/figures/revenue_curve.png'))

if __name__ == "__main__":
    main()
