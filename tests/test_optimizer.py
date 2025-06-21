from src.traditional.scipy_optimizer import ProductParams, optimize_price


def test_optimize_price():
    params = ProductParams(base_price=100, base_demand=50, elasticity=-1.0, cost=60)
    price = optimize_price(params, (80, 120))
    assert 80 <= price <= 120
