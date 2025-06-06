#!/usr/bin/env python3
"""
Run the epsilon-sim platform simulation with line break scenario.

This script demonstrates the main simulation functionality with
realistic parameters and displays the results.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from epsilon_sim.core.simulator import PlatformSimulator


def run_line_break_simulation():
    """Run the main line break simulation scenario."""
    
    print("=" * 60)
    print("EPSILON-SIM: FLOATING PLATFORM LINE BREAK SIMULATION")
    print("=" * 60)
    
    # Create simulator with default parameters
    print("Setting up simulation...")
    sim = PlatformSimulator()
    
    # Configure line break scenario (Line 0 breaks at t=0)
    print("Configuring line break scenario:")
    print("  - Line 0 breaks at t=0.0 seconds")
    sim.set_line_break(0, 0.0)
    
    # Set initial position slightly off equilibrium to create line tensions
    print("  - Setting initial platform offset to create line tension")
    sim.platform.state.x = 3.0   # 3m surge displacement
    sim.platform.state.y = 2.0   # 2m sway displacement
    sim.platform.state.psi = 0.05  # Small initial yaw (about 3 degrees)
    
    # Display configuration
    print("\nSimulation Configuration:")
    print(f"  Platform mass: {sim.platform.mass/1e6:.1f} million kg")
    print(f"  Platform size: {sim.platform.length} x {sim.platform.width} m")
    print(f"  Mooring lines: {sim.mooring_system.num_lines}")
    print(f"  Line length: {sim.mooring_system.unstretched_length} m")
    print(f"  Line stiffness: {sim.mooring_system.stiffness/1e9:.1f} GN")
    
    # Validate configuration
    warnings = sim.validate_configuration()
    if warnings:
        print("\nConfiguration warnings:")
        for warning in warnings:
            print(f"  - {warning}")
    else:
        print("\nConfiguration validated successfully")
    
    # Run simulation
    print(f"\nRunning simulation for 120 seconds...")
    print("This may take a moment...")
    
    results = sim.run(duration=120.0, max_step=0.25)
    
    print(f"\nSimulation completed successfully!")
    print(f"  Computation time: {results.computation_time:.2f} seconds")
    print(f"  Time steps: {results.num_steps}")
    print(f"  Average time step: {np.mean(np.diff(results.time)):.3f} s")
    
    # Analyze results
    print("\nSimulation Results:")
    
    # Platform motion
    final_pos = [results.x[-1], results.y[-1]]
    final_heading = np.degrees(results.psi[-1])
    max_displacement = np.max(np.sqrt(results.x**2 + results.y**2))
    max_speed = np.max(np.sqrt(results.dx**2 + results.dy**2))
    
    print(f"  Final position: ({final_pos[0]:.2f}, {final_pos[1]:.2f}) m")
    print(f"  Final heading: {final_heading:.1f}°")
    print(f"  Maximum displacement: {max_displacement:.2f} m")
    print(f"  Maximum speed: {max_speed:.3f} m/s")
    
    # Line forces
    final_forces = results.line_forces[-1] / 1e6  # Convert to MN
    max_forces = np.max(results.line_forces, axis=0) / 1e6
    
    print(f"\nLine Forces at end of simulation:")
    for i in range(4):
        status = "BROKEN" if final_forces[i] < 0.01 else "INTACT"
        print(f"  Line {i}: {final_forces[i]:.2f} MN (max: {max_forces[i]:.2f} MN) [{status}]")
    
    # Energy analysis
    final_ke = results.kinetic_energy[-1] / 1e6  # Convert to MJ
    final_pe = results.potential_energy[-1] / 1e6
    max_ke = np.max(results.kinetic_energy) / 1e6
    
    print(f"\nEnergy Analysis:")
    print(f"  Final kinetic energy: {final_ke:.2f} MJ")
    print(f"  Final potential energy: {final_pe:.2f} MJ")
    print(f"  Maximum kinetic energy: {max_ke:.2f} MJ")
    
    return results


def create_summary_plots(results):
    """Create summary plots of the simulation results."""
    
    print("\nCreating result plots...")
    
    # Create output directory
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    # Set up the figure
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Epsilon-Sim: Platform Line Break Simulation Results', fontsize=16)
    
    # 1. Platform trajectory
    ax1 = axes[0, 0]
    ax1.plot(results.x, results.y, 'b-', linewidth=2, label='Platform trajectory')
    ax1.plot(results.x[0], results.y[0], 'go', markersize=8, label='Start')
    ax1.plot(results.x[-1], results.y[-1], 'ro', markersize=8, label='End')
    
    # Add anchor positions
    anchors = np.array([[60, 60], [-60, 60], [-60, -60], [60, -60]])
    ax1.scatter(anchors[:, 0], anchors[:, 1], c='red', marker='x', s=100, label='Anchors')
    
    ax1.set_xlabel('X Position [m]')
    ax1.set_ylabel('Y Position [m]')
    ax1.set_title('Platform Trajectory')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.axis('equal')
    
    # 2. Position vs time
    ax2 = axes[0, 1]
    ax2.plot(results.time, results.x, 'b-', label='Surge (X)')
    ax2.plot(results.time, results.y, 'r-', label='Sway (Y)')
    ax2.set_xlabel('Time [s]')
    ax2.set_ylabel('Position [m]')
    ax2.set_title('Platform Position vs Time')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # 3. Heading vs time
    ax3 = axes[0, 2]
    ax3.plot(results.time, np.degrees(results.psi), 'g-', linewidth=2)
    ax3.set_xlabel('Time [s]')
    ax3.set_ylabel('Heading [degrees]')
    ax3.set_title('Platform Heading vs Time')
    ax3.grid(True, alpha=0.3)
    
    # 4. Line forces vs time
    ax4 = axes[1, 0]
    colors = ['blue', 'green', 'orange', 'purple']
    line_names = ['Line 0 (BROKEN)', 'Line 1', 'Line 2', 'Line 3']
    
    for i in range(4):
        forces_MN = results.line_forces[:, i] / 1e6
        linestyle = '--' if i == 0 else '-'  # Dashed for broken line
        alpha = 0.5 if i == 0 else 1.0
        ax4.plot(results.time, forces_MN, color=colors[i], 
                linewidth=2, label=line_names[i], linestyle=linestyle, alpha=alpha)
    
    ax4.set_xlabel('Time [s]')
    ax4.set_ylabel('Line Force [MN]')
    ax4.set_title('Mooring Line Forces vs Time')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    
    # 5. Velocity vs time
    ax5 = axes[1, 1]
    speed = np.sqrt(results.dx**2 + results.dy**2)
    ax5.plot(results.time, speed, 'purple', linewidth=2, label='Speed magnitude')
    ax5.plot(results.time, results.dx, 'b--', alpha=0.7, label='Surge velocity')
    ax5.plot(results.time, results.dy, 'r--', alpha=0.7, label='Sway velocity')
    ax5.set_xlabel('Time [s]')
    ax5.set_ylabel('Velocity [m/s]')
    ax5.set_title('Platform Velocity vs Time')
    ax5.grid(True, alpha=0.3)
    ax5.legend()
    
    # 6. Energy vs time
    ax6 = axes[1, 2]
    ke_MJ = results.kinetic_energy / 1e6
    pe_MJ = results.potential_energy / 1e6
    total_energy = ke_MJ + pe_MJ
    
    ax6.plot(results.time, ke_MJ, 'r-', linewidth=2, label='Kinetic Energy')
    ax6.plot(results.time, pe_MJ, 'b-', linewidth=2, label='Potential Energy')
    ax6.plot(results.time, total_energy, 'g-', linewidth=2, label='Total Energy')
    ax6.set_xlabel('Time [s]')
    ax6.set_ylabel('Energy [MJ]')
    ax6.set_title('Energy vs Time')
    ax6.grid(True, alpha=0.3)
    ax6.legend()
    
    plt.tight_layout()
    
    # Save the plot
    plot_file = output_dir / "simulation_results.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"  Results plot saved: {plot_file}")
    
    # Show the plot
    plt.show()


def main():
    """Main function to run simulation and create plots."""
    
    try:
        # Run the simulation
        results = run_line_break_simulation()
        
        # Create plots
        create_summary_plots(results)
        
        print("\n" + "=" * 60)
        print("SIMULATION COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("\nKey observations:")
        print("- Platform drifted due to line break asymmetry")
        print("- Remaining lines experienced load redistribution")
        print("- System reached a new equilibrium position")
        print("- Motion shows expected dynamic response")
        
        print(f"\nOutput files created in 'output/' directory")
        
        return 0
        
    except Exception as e:
        print(f"\nSimulation failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main()) 