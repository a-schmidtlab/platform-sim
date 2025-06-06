#!/usr/bin/env python3
"""
Basic simulation example for epsilon-sim

This script demonstrates the basic usage of the epsilon-sim package
to simulate a floating platform with mooring line breaking.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Import epsilon-sim modules
# Note: These imports will work once the core modules are implemented
try:
    import epsilon_sim as eps
except ImportError:
    print("epsilon-sim package not yet installed. Run 'pip install -e .' first.")
    exit(1)


def main():
    """Run a basic simulation example."""
    
    print("Starting Epsilon-Sim Basic Example")
    print("=" * 50)
    
    # Set up output directory
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    # 1. Create simulation with default parameters
    print("Setting up simulation...")
    sim = eps.PlatformSimulator()
    
    # 2. Configure line breaking scenario
    # Line 0 breaks at t=0 seconds
    sim.set_line_break(line_id=0, break_time=0.0)
    
    # 3. Run the simulation
    print("Running simulation (120 seconds)...")
    results = sim.run(duration=120.0)
    
    print(f"Simulation completed!")
    print(f"   - Time steps: {len(results.time)}")
    print(f"   - Final position: ({results.x[-1]:.2f}, {results.y[-1]:.2f}) m")
    print(f"   - Final heading: {np.degrees(results.psi[-1]):.1f}°")
    
    # 4. Create basic plots
    print("Creating plots...")
    create_basic_plots(results, output_dir)
    
    # 5. Generate interactive animation
    print("Creating interactive animation...")
    animator = eps.InteractiveAnimator(results)
    animator.create_interactive_display(speed_control=True, load_sliders=True)
    
    # Also save static video
    video_path = output_dir / "basic_simulation.mp4"
    animator.save_video(str(video_path), speedup=6, fps=30)
    
    print(f"Results saved to: {output_dir}")
    print(f"   - Animation: {video_path}")
    print(f"   - Plots: {output_dir}/plots_*.png")
    
    return results


def create_basic_plots(results, output_dir):
    """Create basic visualization plots."""
    
    # Plot 1: Platform trajectory
    plt.figure(figsize=(10, 8))
    plt.plot(results.x, results.y, 'b-', linewidth=2, label='Platform trajectory')
    plt.plot(results.x[0], results.y[0], 'go', markersize=8, label='Start')
    plt.plot(results.x[-1], results.y[-1], 'ro', markersize=8, label='End')
    
    # Add anchor positions
    anchors = np.array([[60, 60], [-60, 60], [-60, -60], [60, -60]])
    plt.scatter(anchors[:, 0], anchors[:, 1], c='red', marker='x', s=100, label='Anchors')
    
    plt.xlabel('X Position [m]')
    plt.ylabel('Y Position [m]')
    plt.title('Platform Trajectory (Line 0 Break at t=0)')
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "trajectory.png", dpi=150)
    plt.close()
    
    # Plot 2: Time series
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Position components
    axes[0, 0].plot(results.time, results.x, 'b-', label='X')
    axes[0, 0].plot(results.time, results.y, 'r-', label='Y')
    axes[0, 0].set_xlabel('Time [s]')
    axes[0, 0].set_ylabel('Position [m]')
    axes[0, 0].set_title('Platform Position')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()
    
    # Heading angle
    axes[0, 1].plot(results.time, np.degrees(results.psi), 'g-')
    axes[0, 1].set_xlabel('Time [s]')
    axes[0, 1].set_ylabel('Heading [deg]')
    axes[0, 1].set_title('Platform Heading')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Mooring line forces (if available)
    if hasattr(results, 'line_forces'):
        for i, force in enumerate(results.line_forces):
            if i == 0:  # Broken line
                axes[1, 0].plot(results.time, force, '--', alpha=0.5, label=f'Line {i} (broken)')
            else:
                axes[1, 0].plot(results.time, force, '-', label=f'Line {i}')
        axes[1, 0].set_xlabel('Time [s]')
        axes[1, 0].set_ylabel('Line Force [N]')
        axes[1, 0].set_title('Mooring Line Forces')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].legend()
    
    # Platform speed
    speed = np.sqrt(results.dx**2 + results.dy**2)
    axes[1, 1].plot(results.time, speed, 'purple')
    axes[1, 1].set_xlabel('Time [s]')
    axes[1, 1].set_ylabel('Speed [m/s]')
    axes[1, 1].set_title('Platform Speed')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "time_series.png", dpi=150)
    plt.close()
    
    print("   Plots saved")


if __name__ == "__main__":
    # Run the example
    results = main()
    
    # Optional: Print some statistics
    print("\nSimulation Statistics:")
    print(f"   - Maximum displacement: {np.max(np.sqrt(results.x**2 + results.y**2)):.2f} m")
    print(f"   - Maximum speed: {np.max(np.sqrt(results.dx**2 + results.dy**2)):.3f} m/s")
    print(f"   - Final drift distance: {np.sqrt(results.x[-1]**2 + results.y[-1]**2):.2f} m") 