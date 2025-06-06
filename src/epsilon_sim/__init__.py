"""
Epsilon-Sim: Floating Platform Mooring Line Simulation

A 2D simulation package for analyzing floating platform dynamics
with mooring line systems, including line breaking scenarios.
"""

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

from .core.simulator import PlatformSimulator
from .core.platform import Platform
from .physics.mooring import MooringSystem
from .visualization.animator import TimelapseAnimator, InteractiveAnimator

__all__ = [
    "PlatformSimulator",
    "Platform", 
    "MooringSystem",
    "TimelapseAnimator",
    "InteractiveAnimator"
] 