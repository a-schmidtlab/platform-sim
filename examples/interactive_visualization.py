#!/usr/bin/env python3
"""
Interactive visualization example for epsilon-sim

This script demonstrates the interactive visualization capabilities
including speed controls and load adjustment sliders.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Import epsilon-sim modules
try:
    import epsilon_sim as eps
except ImportError:
    print("epsilon-sim package not yet installed. Run 'pip install -e .' first.")
    exit(1)


def main():
    """Run interactive visualization example."""
    
    print("Starting Interactive Visualization Example")
    print("=" * 50)
    
    # Set up output directory
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    # 1. Create simulation with default parameters
    print("Setting up simulation...")
    sim = eps.PlatformSimulator()
    
    # 2. Configure line breaking scenario
    sim.set_line_break(line_id=0, break_time=0.0)
    
    # 3. Run the simulation to get baseline results
    print("Running baseline simulation...")
    results = sim.run(duration=120.0)
    
    print("Simulation completed!")
    print(f"   - Time steps: {len(results.time)}")
    
    # 4. Create interactive visualization
    print("Creating interactive visualization...")
    print("Features:")
    print("   - Looping animation of platform movement")
    print("   - Speed control slider (0.1x to 10x)")
    print("   - Load adjustment sliders for all 4 lines")
    print("   - Real-time force display")
    print("   - Dynamic force diagram")
    
    # Create the interactive animator
    animator = eps.InteractiveAnimator(results)
    
    # Configure interactive features
    config = {
        'speed_control': {
            'enabled': True,
            'min_speed': 0.1,
            'max_speed': 10.0,
            'default_speed': 1.0
        },
        'load_sliders': {
            'enabled': True,
            'min_load_factor': 0.0,
            'max_load_factor': 2.0,
            'default_load_factor': 1.0
        },
        'force_display': {
            'show_bars': True,
            'show_values': True,
            'color_code_lines': True
        },
        'loop_animation': True
    }
    
    # Start interactive display
    animator.create_interactive_display(**config)
    
    print("Interactive visualization launched!")
    print("Controls:")
    print("   - Speed Slider: Adjust animation playback speed")
    print("   - Line Load Sliders: Modify individual line loads")
    print("   - Play/Pause Button: Control animation")
    print("   - Reset Button: Return to original state")
    print("   - Close window to exit")
    
    return animator


def create_load_sensitivity_study():
    """Demonstrate load sensitivity analysis."""
    
    print("\nRunning Load Sensitivity Study...")
    
    # Test different load scenarios
    load_scenarios = [
        [1.0, 1.0, 1.0, 1.0],  # Baseline
        [0.5, 1.0, 1.0, 1.0],  # Reduced load on line 0
        [0.0, 1.0, 1.0, 1.0],  # Line 0 broken
        [0.0, 1.5, 1.5, 1.0],  # Line 0 broken, compensated by lines 1&2
    ]
    
    scenario_names = [
        "Baseline (all lines 100%)",
        "Line 0 at 50% load", 
        "Line 0 broken",
        "Line 0 broken, lines 1&2 increased"
    ]
    
    results_all = []
    
    for i, (loads, name) in enumerate(zip(load_scenarios, scenario_names)):
        print(f"   Scenario {i+1}: {name}")
        
        sim = eps.PlatformSimulator()
        
        # Apply load factors
        for line_id, load_factor in enumerate(loads):
            sim.set_line_load_factor(line_id, load_factor)
        
        # Run simulation
        results = sim.run(duration=60.0)  # Shorter for comparison
        results_all.append((results, name, loads))
    
    # Create comparison plots
    create_comparison_plots(results_all, Path("output"))
    
    print("   Load sensitivity study completed!")
    
    return results_all


def create_comparison_plots(results_all, output_dir):
    """Create comparison plots for different load scenarios."""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    colors = ['blue', 'red', 'green', 'orange']
    
    for i, (results, name, loads) in enumerate(results_all):
        color = colors[i % len(colors)]
        
        # Trajectory comparison
        axes[0, 0].plot(results.x, results.y, color=color, label=name, linewidth=2)
        
        # Position time series
        displacement = np.sqrt(results.x**2 + results.y**2)
        axes[0, 1].plot(results.time, displacement, color=color, label=name, linewidth=2)
        
        # Heading angle
        axes[1, 0].plot(results.time, np.degrees(results.psi), color=color, label=name, linewidth=2)
        
        # Speed
        speed = np.sqrt(results.dx**2 + results.dy**2)
        axes[1, 1].plot(results.time, speed, color=color, label=name, linewidth=2)
    
    # Configure subplots
    axes[0, 0].set_xlabel('X Position [m]')
    axes[0, 0].set_ylabel('Y Position [m]')
    axes[0, 0].set_title('Platform Trajectories')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()
    axes[0, 0].axis('equal')
    
    axes[0, 1].set_xlabel('Time [s]')
    axes[0, 1].set_ylabel('Displacement [m]')
    axes[0, 1].set_title('Platform Displacement from Origin')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()
    
    axes[1, 0].set_xlabel('Time [s]')
    axes[1, 0].set_ylabel('Heading [deg]')
    axes[1, 0].set_title('Platform Heading Angle')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()
    
    axes[1, 1].set_xlabel('Time [s]')
    axes[1, 1].set_ylabel('Speed [m/s]')
    axes[1, 1].set_title('Platform Speed')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / "load_sensitivity_comparison.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    # Run the interactive example
    animator = main()
    
    # Optionally run sensitivity study
    sensitivity_results = create_load_sensitivity_study()
    
    print("\nExample completed!")
    print("Check the output directory for saved plots and videos.") 