# Platform-Sim: Floating Platform Dynamics

A 2D simulation exploring how floating platforms behave when mooring lines break. The focus is on understanding what happens when anchor lines fail and how the platform responds.

## Content

1. [The Physics Behind It](#the-physics-behind-it)
2. [How the Model Works](#how-the-model-works)  
3. [What the Software Does](#what-the-software-does)
4. [Getting Started](#getting-started)
5. [Project Organization](#project-organization)
6. [Development](#development)

---

## The Physics Behind It

### Why This Matters

Floating platforms are essential infrastructure for offshore operations - oil rigs, wind farms, research stations. They are ofte brought into position using anchor lines, but sometimes these lines break. Understanding the dynamic ist interesting.

### Basic Physics Concepts

#### Platform Motion

Real floating platforms can move in six ways:
- Slide forward/backward (surge), left/right (sway), up/down (heave)
- Tilt forward/backward (pitch), left/right (roll), or spin (yaw)

This simulation focuses on horizontal motions (surge, sway, yaw) since these dominate the response to mooring line failures.

The fundamental equation looks like this:

```
[M + A(ω)] ẍ + [B(ω) + B_linear] ẋ + [C] x = F_external(t)
```

Where:
- `[M]` = Platform mass/inertia
- `[A(ω)]` = Added mass from moving water
- `[B(ω)]` = Energy lost creating waves
- `[B_linear]` = Energy lost to friction
- `[C]` = Restoring forces
- `F_external(t)` = External forces

#### Mooring Line Physics

Anchor lines follow catenary curves - how heavy chains hang between points:

```
Horizontal Tension: TH = constant along line
Vertical Tension: TV = TH × sinh(s/a)
Line Shape: y = a × [cosh(x/a) - 1]
```

Total line tension combines static and dynamic components:
```
T_total = T_static + T_dynamic
```

Lines break due to:
- Overload (exceeding strength limits)
- Fatigue (repeated stress cycles)  
- Wear at connection points
- Corrosion from seawater

#### Force Buildup Before Breaking

Forces increase before the line breaks at t=8.0s because:

1. Platform oscillates constantly due to waves
2. Larger movements create more line stretch
3. More stretch means higher force (Hooke's Law: `F = (EA/L₀) × extension`)
4. Peak movements can generate 200+ MN forces
5. When Line 0 breaks, remaining lines redistribute the total load

This represents normal behavior for mooring systems under wave loading.

---

## How the Model Works

### Approach

The simulation uses a "reduced-order model" - simplified but captures essential physics without excessive complexity.

#### Platform Model

Platform treated as rigid body with three motion directions:

```python
# State vector: [x, y, ψ, ẋ, ẏ, ψ̇]
# Mass matrix [3×3]:
M = [
    [m + m_added_surge,              0,           0        ],
    [     0,           m + m_added_sway,        0        ],  
    [     0,                 0,      I_zz + I_added_yaw]
]

# Damping matrix [3×3]:
B = [
    [b_surge,    0,      0   ],
    [   0,    b_sway,   0   ],
    [   0,       0,   b_yaw ]
]
```

Parameters:
- Platform Mass: 12.5 million kg
- Platform Size: 120m × 120m square
- Added Mass: 15% extra for water motion
- Damping: Tuned for realistic oscillations

#### Force Calculation

Real-time force computation for each frame:

```python
def calculate_line_force(attachment_pos, anchor_pos):
    # Distance calculation
    line_vector = attachment_pos - anchor_pos
    current_length = np.linalg.norm(line_vector)
    
    # Line stretch
    extension = max(0.0, current_length - L0)
    
    # Apply Hooke's law
    force_magnitude = (EA / L0) * extension
    
    return force_magnitude
```

Line parameters:
- Unstretched Length: 300m
- Stiffness: 1.2×10⁹ N (steel chain equivalent)
- Breaking: Instant failure at set time
- Layout: Four lines from platform corners to diagonal anchors

#### Time Integration

Adaptive Runge-Kutta integration:
- Automatic step size adjustment
- Minimum step: 0.001s for stability
- Maximum step: 0.01s for dynamics
- High precision force calculations

### Validation and Limitations

Working correctly:
- Energy conservation
- Equilibrium positions
- Oscillation periods (40-60 seconds)
- Force magnitudes (0-400+ MN range)
- Pre-break force physics

Current limitations:
- 2D motion only (no heave, roll, pitch)
- Simplified mooring model
- Basic hydrodynamics

Suitable for:
- Understanding platform behavior
- Single line failure studies
- Physics exploration
- Parameter studies

Not suitable for:
- Engineering design validation
- Multiple line failures
- Extreme weather analysis

---

## What the Software Does

### Main Simulation

Run the interactive simulation:

```bash
python run_dynamic_simulation.py
```

Interface features:
- Professional display showing "Floating Platform Dynamic Analysis: 120m×120m Semi-Ballasted Structure | 4-Line Diagonal Mooring System | Line Break Response at t=8.0s"
- Real-time force monitoring with color coding:
  - Normal operation (under 50 MN): standard colors
  - Overload condition (50-100 MN): bright colors  
  - Critical loading (over 100 MN): dark colors
  - Broken line: gray (0.00 MN)
- Auto-scaling plots handling 0-400+ MN forces
- Reference lines for working loads and breaking points

Simulation sequence:
1. First 8 seconds: Normal oscillation, forces 80-200 MN
2. t=8.0s: Line 0 breaks (turns gray)
3. After t=8s: Platform drifts, remaining lines carry 140-362+ MN
4. Position tracking: surge (30-80m), sway (20-40m), rotation

Controls:
- Speed slider (0.1x to 5.0x)
- Play/pause
- Reset

### Programmatic Use

Python API:

```python
import epsilon_sim as eps
import numpy as np

# Setup
sim = eps.PlatformSimulator()
sim.set_line_break(line_id=0, break_time=8.0)

# Run
results = sim.run(duration=120.0, max_step=0.01)

# Analysis
max_force = np.max(results.line_forces) / 1e6  # Convert to MN
max_displacement = np.max(np.sqrt(results.x**2 + results.y**2))
print(f"Maximum line force: {max_force:.1f} MN")
print(f"Maximum displacement: {max_displacement:.1f} m")
```

### Configuration

Parameters in YAML files:

```yaml
platform:
  mass: 1.25e7                 # kg - Total mass
  length: 120.0                # m - Dimensions
  width: 120.0                 # m
  inertia_z: 8.0e9            # kg⋅m² - Rotational inertia

mooring:
  num_lines: 4                 # Four-point mooring
  length: 300.0                # m - Line length
  stiffness: 1.2e9            # N - Line stiffness
  anchor_positions:           # Anchor locations
    - [280, 280]              # Northeast
    - [-280, 280]             # Northwest
    - [-280, -280]            # Southwest  
    - [280, -280]             # Southeast

simulation:
  break_time: 8.0             # Line failure time
  duration: 120.0             # Simulation length
```

---

## Getting Started

### Requirements
- Python 3.11 or newer
- FFmpeg (for videos, optional)
- Linux, macOS, or Windows

### Setup
```bash
git clone https://github.com/a-schmidtlab/platform-sim.git
cd epsilon-sim
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

### Running
```bash
# Interactive simulation
python run_dynamic_simulation.py

# Batch run
python run_simulation.py
```

### Expected Results
- Platform movement: 30-80m surge, 20-40m sway after line break
- Line forces: 0-400+ MN showing offshore loading
- Natural period: 40-60 seconds
- Computation time: 5-15 seconds for 120-second simulation

---

## Project Organization

```
epsilon-sim/
├── src/epsilon_sim/           # Main code
│   ├── core/                  # Platform physics
│   ├── physics/               # Force models
│   ├── visualization/         # Graphics and animation  
│   └── utils/                 # Configuration tools
├── scripts/                   # Runnable programs
│   ├── run_dynamic_simulation.py    # Main interface
│   ├── run_simulation.py            # Basic version
│   └── run_dramatic_simulation.py   # Extreme conditions
├── config/                    # Parameter files
├── tests/                     # Testing code
└── examples/                  # Learning examples
```

---

## Development

### Contributing
```bash
pip install -r requirements-dev.txt
pytest tests/ -v
black src/ tests/
```

### Extension Ideas
- Add full 6-DOF motion (heave, roll, pitch)
- Include wind and current effects
- Model fatigue and gradual failure
- Multiple line break scenarios

---

## Technical Sources and Further Reading

### Academic References

**Floating Platform Dynamics:**
- Faltinsen, O.M. (1990). "Sea Loads on Ships and Offshore Structures". Cambridge University Press.
- Newman, J.N. (1977). "Marine Hydrodynamics". MIT Press.
- Chakrabarti, S.K. (2005). "Handbook of Offshore Engineering". Elsevier.

**Mooring System Analysis:**
- API RP 2SK (2005). "Design and Analysis of Stationkeeping Systems for Floating Structures". American Petroleum Institute.
- DNV-OS-E301 (2013). "Position Mooring". Det Norske Veritas.
- Vryhof Anchors (2010). "Anchor Manual - The Guide to Anchoring". Vryhof Anchors BV.

**Numerical Methods:**
- Cummins, W.E. (1962). "The Impulse Response Function and Ship Motions". DTMB Report 1661.
- Ogilvie, T.F. (1964). "Recent Progress Toward the Understanding and Prediction of Ship Motions". 5th ONR Symposium on Naval Hydrodynamics.

### Industry Standards

**Design Codes:**
- API RP 2FPS: "Planning, Designing and Constructing Floating Production Systems"
- ISO 19901-7: "Petroleum and Natural Gas Industries - Stationkeeping Systems for Floating Offshore Petroleum and Gas Structures"
- ABS Guide for Position Mooring Systems

**Classification Societies:**
- DNV GL: Rules for Classification of Ships and Offshore Units
- ABS: Rules for Building and Classing Mobile Offshore Drilling Units
- Lloyd's Register: Rules and Regulations for the Classification of Mobile Offshore Units

### Professional Software

**Commercial Mooring Analysis:**
- OrcaFlex (Orcina): Time-domain dynamic analysis
- AQWA (ANSYS): Frequency and time-domain analysis
- WAMIT: Wave analysis for floating bodies
- MoorDyn: Open-source mooring dynamics

**Platform Motion Analysis:**
- SESAM (DNV GL): Complete offshore analysis suite
- MOSES (Bentley): Multi-purpose offshore analysis
- FAST (NREL): Wind turbine simulation including floating platforms

### Open Source Tools

**Analysis Software:**
- OpenFOAM: Computational fluid dynamics
- Code_Aster: Finite element analysis
- FEniCS: Partial differential equation solving
- NumPy/SciPy: Scientific computing in Python

**Data and Validation:**
- NDBC (National Data Buoy Center): Real ocean measurement data
- ECMWF: Global weather and wave data
- JONSWAP spectrum: Standard wave spectra for analysis

---

## License

Open source project released under MIT License.

---



Greetings, Axel Schmidt
