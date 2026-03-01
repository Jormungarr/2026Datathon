from setuptools import setup, find_packages

setup(
    name="datathon2026",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "matplotlib",
        "seaborn",
        "scipy",
        "scikit-learn",
        "geopy",
        "plotly",
        "pygam"
        ],
    author="Your Name",
    description="Datathon 2026 project utilities and analysis.",
    python_requires=">=3.7",
)