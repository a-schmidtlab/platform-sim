"""Setup script for epsilon-sim package."""

from setuptools import setup, find_packages
import os

# Read README file for long description
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

# Read version from package
def read_version():
    version_file = os.path.join("src", "epsilon_sim", "__init__.py")
    with open(version_file, "r", encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("__version__"):
                return line.split("=")[1].strip().strip('"').strip("'")
    return "0.1.0"

setup(
    name="epsilon-sim",
    version=read_version(),
    author="Your Name",
    author_email="your.email@example.com",
    description="2D floating platform simulation with mooring line dynamics",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/epsilon-sim",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.11",
    install_requires=read_requirements(),
    extras_require={
        "advanced": ["moorpy>=1.0.0", "moordyn>=2.0.0"],
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0", 
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
        "docs": [
            "sphinx>=6.0.0",
            "sphinx-rtd-theme>=1.2.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "epsilon-sim=epsilon_sim.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "epsilon_sim": ["config/*.yaml"],
    },
) 