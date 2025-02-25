# setup.py update
from setuptools import setup, find_packages

setup(
    name="yflow",
    version="0.1.0",
    packages=find_packages(),
    # Core dependencies only
    install_requires=[
        "numpy>=1.19.0",
        "scikit-learn>=0.24.0",
    ],
    # Optional dependencies
    extras_require={
        'gpu': [
            'cupy>=9.0.0',  # Adding CuPy
        ],
        'dev': [
            'pytest>=6.0',  # For testing
        ]
    }
)