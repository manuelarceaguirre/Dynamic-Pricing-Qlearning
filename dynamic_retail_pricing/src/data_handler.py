from .config import PRODUCTS_DATA


def get_product_data(name):
    """Retrieve product dictionary by name."""
    for product in PRODUCTS_DATA:
        if product['name'] == name:
            return product
    raise ValueError(f"Product '{name}' not found")
