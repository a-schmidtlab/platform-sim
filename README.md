# Epsilon-Sim: Floating Platform Simulation

A 2D simulation package for analyzing floating platform dynamics with mooring line systems, specifically designed to study line breaking scenarios and their effects on platform motion.

## Overview

This simulation models a semi-ballasted four-column floating platform connected to anchor points via mooring lines. The primary use case is analyzing the platform's response when one or more mooring lines break, calculating:

- Platform trajectory (surge, sway, yaw)
- Tension forces in remaining mooring lines  
- Timelapse animations of the platform motion

## Features

- **3-DOF Platform Dynamics**: Surge, sway, and yaw motion simulation
- **Mooring Line Physics**: Linear spring model with line breaking capabilities
- **Timelapse Animation**: High-quality MP4 output with customizable speed
- **Parameter Configuration**: YAML-based configuration system
- **Extensible Design**: Modular architecture for adding advanced features

## Installation

### Prerequisites
- Python 3.11 or higher
- FFmpeg (for video generation)

### Setup
```bash
# Clone the repository
git clone <repository-url>
cd epsilon-sim

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

## Quick Start

### Running Simulations

#### Basic Python API
```python
import epsilon_sim as eps

# Create simulation with default parameters
sim = eps.PlatformSimulator()

# Run simulation with line break at t=0
results = sim.run(break_line=0, duration=120.0)

# Generate timelapse animation
animator = eps.TimelapseAnimator(results)
animator.create_video("output/simulation.mp4", speedup=6)
```

#### Command Line Simulations
```bash
# Standard simulation with line break at t=0
python run_simulation.py

# Dramatic simulation with significant initial displacement
python run_dramatic_simulation.py
```

#### Dynamic Real-time Simulation (NEW!)
Experience the complete platform dynamics sequence in real-time:

```bash
# Run dynamic simulation with visual break sequence
python run_dynamic_simulation.py
```

**Dynamic Simulation Features:**
- **Visual Platform**: Platform displayed as rotatable square (120m x 120m)
- **Break Timeline**: Pre-break equilibrium → line failure event → oscillating response  
- **Real-time Data**: Live force bars, position tracking, and energy visualization
- **Interactive Controls**: Play/Pause, Reset, Speed adjustment (0.1x to 10x)
- **Automatic Looping**: Continuous replay of the complete sequence
- **Realistic Physics**: Shows characteristic oscillating turning motion after line break

**What You'll See:**
1. **t = 0-5s**: Platform in slight offset equilibrium with all lines tensioned
2. **t = 5s**: Line 0 breaks (dramatic visual change from red to gray dashed line)
3. **t = 5-60s**: Platform exhibits realistic oscillating drift with rotation
4. **Data Plots**: Real-time force redistribution, position tracking, energy evolution

## Project Structure

```
epsilon-sim/
├── src/epsilon_sim/           # Main source code
│   ├── core/                  # Core simulation components
│   │   ├── simulator.py       # Main simulation engine
│   │   ├── platform.py        # Platform dynamics model
│   │   └── state.py           # State management
│   ├── physics/               # Physics models
│   │   ├── mooring.py         # Mooring line dynamics
│   │   ├── forces.py          # Force calculations
│   │   └── integration.py     # Numerical integration
│   ├── visualization/         # Visualization and animation
│   │   ├── animator.py        # Animation generation
│   │   ├── plotter.py         # Data visualization
│   │   └── renderer.py        # Real-time rendering
│   └── utils/                 # Utility functions
│       ├── config.py          # Configuration management
│       ├── validation.py      # Input validation
│       └── io.py              # File I/O operations
├── tests/                     # Test suite
│   ├── unit/                  # Unit tests
│   └── integration/           # Integration tests
├── config/                    # Configuration files
│   ├── default.yaml           # Default parameters
│   └── scenarios/             # Predefined scenarios
├── examples/                  # Example scripts and notebooks
├── docs/                      # Documentation
├── scripts/                   # Utility scripts
├── output/                    # Generated outputs
└── data/                      # Input data files
```

## Physics Model and Assumptions

The simulation implements a 3-DOF (surge, sway, yaw) model of a floating platform with enhanced hydrodynamics:

### Platform Model
- **Geometry**: Semi-ballasted square platform (120m × 120m, 1.25×10⁷ kg)
- **Degrees of Freedom**: Surge (x), Sway (y), Yaw (ψ) motion only
- **Added Mass**: 15% of platform mass for surge/sway, 10% for yaw inertia
- **Coordinate System**: Global frame with origin at initial anchor center

### Mooring System  
- **Configuration**: 4 catenary lines connecting platform corners to seabed anchors
- **Line Properties**: 400m unstretched length, 1.2×10⁹ N axial stiffness
- **Force Model**: Linear elastic (Hooke's law) with geometric nonlinearity
- **Breaking Model**: Instantaneous complete failure at specified time
- **Anchor Positions**: Fixed at (±60m, ±60m) on seabed

### Hydrodynamic Effects
- **Linear Damping**: Velocity-proportional resistance (8×10⁵ N·s/m)
- **Quadratic Damping**: Velocity-squared drag effects (reduced coefficient)
- **Wave Forces**: Multi-frequency sinusoidal excitation forces
- **Memory Effects**: Simplified radiation damping for fluid-structure interaction
- **Hydrostatic Restoring**: Yaw restoring moment only (2.5×10⁸ N·m/rad)

### Key Assumptions and Limitations

#### Simplifications Made:
1. **2D Motion**: No heave, roll, or pitch degrees of freedom
2. **Shallow Water**: No wave-frequency dependent effects
3. **Linear Mooring**: Catenary effects approximated by geometric constraints
4. **Rigid Body**: Platform treated as single rigid body
5. **Instantaneous Breaking**: No gradual line degradation or cascading failures
6. **Fixed Anchors**: No anchor dragging or pullout
7. **Calm Weather**: No wind loads or current effects included

#### Physical Validity:
- **Natural Periods**: Platform oscillations ~40-60 seconds (typical for semi-submersibles)
- **Damping Ratios**: Approximately 5-10% critical damping for sustained oscillations
- **Force Magnitudes**: Wave forces up to 8 MN (realistic for extreme conditions)
- **Displacement Range**: Platform can drift 100+ meters after line failure
- **Speed Range**: Transient speeds up to 3-5 m/s during failure response

#### Validation Status:
- **Energy Conservation**: Verified for conservative forces
- **Equilibrium**: Static equilibrium validated for intact system
- **Stability**: Integration stability maintained with adaptive time stepping
- **Oscillation Characteristics**: Match literature values for floating platforms

#### Known Limitations:
1. **No 6-DOF Coupling**: Missing roll-pitch-heave interactions
2. **No Environmental Loading**: No waves, wind, or current
3. **Simplified Damping**: Real viscous effects more complex
4. **No Line Dynamics**: Chain/cable dynamics neglected
5. **No Seabed Interaction**: No anchor-soil mechanics
6. **No Multiple Failures**: Only single line break scenarios

### Recommended Use Cases:
- Preliminary design studies of mooring configurations
- Investigation of platform response to single line failures  
- Educational demonstration of floating platform dynamics
- Basis for more detailed analysis with specialized software

### Not Suitable For:
- Final design validation (use AQWA, OrcaFlex, etc.)
- Multi-failure cascade analysis
- Extreme weather condition assessment
- Detailed fatigue or ultimate strength analysis

## Configuration

The simulation uses YAML configuration files for parameters:

```yaml
# config/default.yaml
platform:
  mass: 1.25e7                 # kg, total mass (semi-ballasted)
  length: 120.0                # m, platform length
  width: 120.0                 # m, platform width  
  inertia_z: 8.0e9            # kg⋅m², yaw moment of inertia

mooring:
  num_lines: 4
  length: 400.0                # m, unstretched length
  stiffness: 1.2e9            # N, axial stiffness EA
  anchor_positions:           # m, [x, y] coordinates
    - [60, 60]
    - [-60, 60] 
    - [-60, -60]
    - [60, -60]

damping:
  linear_coeff: 3.5e6         # N⋅s/m, linear damping
  angular_coeff: 1.75e8       # N⋅m⋅s/rad, angular damping

simulation:
  duration: 120.0             # s, total simulation time
  max_timestep: 0.25          # s, maximum integration timestep
  tolerance: 1e-6             # Integration tolerance
```

## Examples

See the `examples/` directory for:
- Basic simulation example
- Parameter sensitivity analysis
- Advanced visualization techniques
- Batch simulation scripts

## Development

### Running Tests
```bash
pytest tests/
```

### Code Quality
```bash
black src/ tests/
flake8 src/ tests/
mypy src/
```

### Building Documentation
```bash
cd docs/
make html
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Acknowledgments

- Based on offshore platform dynamics principles
- Inspired by MoorDyn and MoorPy mooring analysis tools
- FFmpeg for video encoding capabilities 