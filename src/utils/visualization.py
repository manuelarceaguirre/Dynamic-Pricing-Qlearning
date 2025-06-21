"""Visualization utilities."""
import matplotlib.pyplot as plt


def plot_revenue_curve(prices, revenues, path):
    plt.figure()
    plt.plot(prices, revenues, marker='o')
    plt.xlabel('Price')
    plt.ylabel('Revenue')
    plt.title('Revenue Curve')
    plt.savefig(path)
    plt.close()
