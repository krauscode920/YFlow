# setup.py
from setuptools import setup, find_packages

setup(
    name="yflow",
    version="0.1.0",
    author="YFlow Team",
    author_email="sidkraus920@gmail.com",  # Add your email if you're comfortable
    description="A deep learning library built from scratch with GPU support",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/krauscode920/YFlow",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.7",
    # Core dependencies only
    install_requires=[
        "numpy>=1.19.0",
        "scikit-learn>=0.24.0",
    ],
    # Optional dependencies
    extras_require={
        'gpu': [
            'cupy>=9.0.0',  # GPU acceleration with CuPy
        ],
        'dev': [
            'pytest>=6.0',  # For testing
            'black',        # For code formatting
            'flake8',       # For linting
            'sphinx',       # For documentation
        ],
        'examples': [
            'matplotlib>=3.3.0',  # For visualization
            'jupyter>=1.0.0',     # For notebooks
        ]
    },
    project_urls={
        "Bug Tracker": "https://github.com/krauscode920/YFlow/issues",
        "Documentation": "https://github.com/krauscode920/YFlow",
        "Source Code": "https://github.com/krauscode920/YFlow",
    },
)
