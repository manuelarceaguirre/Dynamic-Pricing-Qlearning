from setuptools import setup, find_packages

setup(
    name='dynamic_retail_pricing_qlearning',
    version='0.1.0',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    install_requires=[
        'numpy>=1.21.0',
        'pandas>=1.3.0',
        'matplotlib>=3.4.0',
        'seaborn>=0.11.0',
        'scipy>=1.7.0',
        'scikit-learn>=1.0.0'
    ],
)
