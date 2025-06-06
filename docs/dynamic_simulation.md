# Dynamic Real-time Platform Simulation

## Overview

The dynamic real-time simulation provides an immersive visualization of floating platform dynamics with mooring line breaking scenarios. Unlike static post-processing animations, this simulation shows the complete sequence of events in real-time with interactive controls and live data visualization.

## Key Features

### Visual Elements

#### Platform Representation
- **Square Visualization**: Platform displayed as a 120m × 120m blue square
- **Real-time Rotation**: Platform rotates according to yaw angle ψ
- **Position Tracking**: Square translates according to surge/sway motion
- **Visual Anchors**: Red 'X' markers show anchor positions at ±60m corners

#### Mooring Line Visualization
- **Four Color-coded Lines**: 
  - Line 0 (Red) - breaks at t=5s
  - Line 1 (Blue) - remains intact
  - Line 2 (Green) - remains intact  
  - Line 3 (Purple) - remains intact
- **Break Animation**: Broken line changes to gray dashed appearance
- **Dynamic Connection**: Lines connect platform corners to anchors in real-time

### Timeline Sequence

The simulation follows a realistic timeline designed to show the complete dynamics:

#### Phase 1: Pre-Break Equilibrium (t = 0-5s)
- Platform starts with small offset (8m surge, 6m sway, 1.7° heading)
- All four mooring lines under tension
- Status indicator shows "PRE-BREAK" in green
- Countdown timer shows time until line failure

#### Phase 2: Line Break Event (t = 5s)
- Line 0 fails instantaneously
- Visual change: Line 0 becomes gray and dashed
- Status indicator shows "LINE BREAK!" in red
- Force redistribution begins immediately

#### Phase 3: Dynamic Response (t = 5-60s)
- Platform exhibits characteristic oscillating motion
- Combination of translation and rotation
- Remaining lines carry redistributed loads
- Status shows "POST-BREAK" in orange

#### Phase 4: Loop Restart
- Automatic return to t=0 for continuous observation
- Data plots reset for fresh visualization
- Complete sequence repeats

## Real-time Data Visualization

### Force Bar Chart (Top Right)
- **Live Force Display**: Real-time bar chart of line tensions in MN
- **Color Coordination**: Bars match line colors
- **Break Indication**: Line 0 bar grays out after break
- **Auto-scaling**: Y-axis adjusts to maximum force values

### Position Tracking (Bottom Left)
- **Three Traces**:
  - Blue: Surge position X(t) [m]
  - Red: Sway position Y(t) [m]  
  - Green: Heading ψ(t) [degrees/10] (scaled for visibility)
- **Break Marker**: Vertical red dashed line at t=5s
- **Real-time Update**: Traces extend as simulation progresses

### System Energy (Bottom Right)
- **Energy Components**:
  - Red: Kinetic energy [MJ]
  - Blue: Potential energy [MJ]
  - Green: Total energy [MJ]
- **Energy Jump**: Visible spike at line break due to sudden force change
- **Oscillations**: Energy exchange between kinetic and potential

## Interactive Controls

### Basic Controls
- **Play Button**: Start/pause animation
- **Reset Button**: Return to t=0 and clear data
- **Speed +/-**: Adjust playback speed (0.1x to 10x)

### Real-time Feedback
- **Time Display**: Current simulation time
- **Speed Indicator**: Current playback speed multiplier
- **Status Display**: Current phase with color-coded background

## Physical Interpretation

### Oscillating Motion
The post-break motion exhibits several characteristic behaviors:

1. **Initial Drift**: Platform moves away from failed anchor
2. **Restoring Forces**: Remaining lines provide centering force
3. **Yaw Oscillations**: Platform rotates due to asymmetric loading
4. **Coupled Motion**: Translation and rotation are coupled through geometry

### Force Redistribution
After line break:
- **Immediate Response**: Forces in remaining lines increase instantly
- **Dynamic Loading**: Forces oscillate as platform moves
- **Equilibrium Shift**: New steady-state with three-line system

### Energy Evolution
- **Break Event**: Sudden potential energy change creates kinetic energy spike
- **Oscillations**: Energy alternates between kinetic and potential
- **Damping**: Gradual energy dissipation due to linear damping

## Technical Implementation

### Simulation Parameters
- **Platform Mass**: 1.25 × 10⁷ kg
- **Platform Size**: 120m × 120m square
- **Moment of Inertia**: 8.0 × 10⁹ kg⋅m²
- **Line Length**: 400m unstretched
- **Line Stiffness**: 1.2 × 10⁹ N
- **Linear Damping**: 3.5 × 10⁶ N⋅s/m
- **Angular Damping**: 1.75 × 10⁸ N⋅m⋅s/rad

### Initial Conditions
- **Position**: (8, 6, 0.03) - surge, sway, yaw [m, m, rad]
- **Velocity**: (0, 0, 0) - starts from rest
- **Purpose**: Creates initial line tension for realistic pre-break state

### Animation Details
- **Frame Rate**: ~20 FPS (50ms intervals)
- **Frame Skip**: 2 (for smooth playback)
- **Data Buffer**: 500 points maximum
- **Auto-loop**: Seamless restart at end

## Running the Simulation

### Basic Execution
```bash
python run_dynamic_simulation.py
```

### Expected Output
```
Dynamic Platform Simulator initialized
  - Line break scheduled at t = 5.0 s
  - Total simulation duration: 60.0 s

============================================================
DYNAMIC PLATFORM SIMULATION - Real-time Visualization
============================================================
Setting up realistic platform simulation...
  Initial platform position: (8.0, 6.0) m
  Initial heading: 1.7°
  Line 0 will break at t = 5.0 s
  Computing dynamics for 60.0 s...
  Simulation computed: 1206 time steps
  Maximum displacement: 42.3 m
  Maximum speed: 0.087 m/s

Dynamic simulation ready!
Controls:
  - Play/Pause: Start/stop animation
  - Reset: Return to beginning

Watch for:
  1. Initial equilibrium (slight offset)
  2. Line break event at t = 5.0 s
  3. Dynamic oscillating response
  4. Platform rotation and drift

Press Play to start!
```

### User Interaction
1. **Press Play**: Animation begins showing pre-break phase
2. **Observe Break**: Watch line failure at t=5s
3. **Study Response**: Analyze oscillating motion and force redistribution
4. **Use Controls**: Adjust speed, reset as needed
5. **Multiple Loops**: Let simulation loop to study repeated sequences

## Applications

### Educational Use
- **Offshore Engineering**: Demonstrate platform dynamics principles
- **Mooring Design**: Show importance of redundancy in mooring systems
- **Failure Analysis**: Visualize consequences of line failure

### Engineering Analysis
- **Design Verification**: Check platform response to line failure
- **Parameter Studies**: Modify initial conditions to study different scenarios
- **Safety Assessment**: Evaluate platform stability after component failure

### Research Applications
- **Model Validation**: Compare simulation results with experimental data
- **Sensitivity Analysis**: Study effects of different parameters
- **Scenario Development**: Create test cases for advanced analysis

## Troubleshooting

### Common Issues

#### Animation Not Starting
- **Solution**: Click "Play" button to start animation
- **Check**: Ensure simulation completed successfully

#### Slow Performance
- **Solution**: Reduce speed using "Speed -" button
- **Alternative**: Close other graphics-intensive applications

#### No Platform Motion
- **Explanation**: Platform may start near equilibrium
- **Solution**: Motion becomes more apparent after line break at t=5s

#### Data Plots Not Updating
- **Check**: Ensure animation is playing (not paused)
- **Solution**: Reset simulation and restart

### Performance Tips
- **Optimal Speed**: 1-2x speed provides good balance of detail and viewing time
- **Loop Observation**: Watch multiple loops to understand periodic behavior
- **Focus Areas**: Pay special attention to the break event and subsequent oscillations

## Future Enhancements

### Planned Features
- **Parameter Adjustment**: Real-time modification of simulation parameters
- **Multiple Break Scenarios**: Support for different line break patterns
- **6-DOF Dynamics**: Extension to include heave, roll, and pitch motion
- **Wave Loading**: Addition of environmental forces
- **Advanced Visualization**: 3D rendering capabilities

### Customization Options
- **Break Timing**: Modify when line break occurs
- **Initial Conditions**: Change starting position and orientation
- **Animation Speed**: Adjust default playback speeds
- **Visual Themes**: Custom colors and styling options 