from setuptools import setup, find_packages

setup(
    name="ebf",
    version="1.0.0",
    description="Eigen-Bias Folding: Zero-Latency LLM Adaptation",
    author="Adaptation Labs Candidate",
    packages=find_packages(),
    install_requires=[
        "torch>=2.1.0",
        "transformers>=4.35.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "safetensors>=0.4.0",
        "tqdm"
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
    ],
)