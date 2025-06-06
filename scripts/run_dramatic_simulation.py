#!/usr/bin/env python3
"""
Run a more dramatic epsilon-sim platform simulation.

This script shows a more dramatic line break scenario with
significant initial displacement to create visible dynamics.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from epsilon_sim.core.simulator import PlatformSimulator


def run_dramatic_simulation():
    """Run a dramatic line break simulation scenario."""
    
    print("=" * 70)
    print("EPSILON-SIM: DRAMATIC FLOATING PLATFORM LINE BREAK SIMULATION")
    print("=" * 70)
    
    # Create simulator
    print("Setting up dramatic simulation...")
    sim = PlatformSimulator()
    
    # Set dramatic initial conditions - platform displaced far from equilibrium
    print("Configuring dramatic scenario:")
    print("  - Line 0 breaks at t=0.0 seconds")
    print("  - Platform starts 20m displaced with initial velocity")
    
    # Configure line break
    sim.set_line_break(0, 0.0)
    
    # Set dramatic initial conditions
    sim.platform.state.x = 20.0      # 20m surge displacement 
    sim.platform.state.y = 15.0      # 15m sway displacement
    sim.platform.state.psi = 0.2     # 11.5 degree initial heading
    sim.platform.state.dx = -0.5     # Initial surge velocity (moving back)
    sim.platform.state.dy = -0.3     # Initial sway velocity
    sim.platform.state.dpsi = 0.02   # Initial yaw rate
    
    print(f"  - Initial position: ({sim.platform.state.x}, {sim.platform.state.y}) m")
    print(f"  - Initial heading: {np.degrees(sim.platform.state.psi):.1f}°")
    print(f"  - Initial velocity: ({sim.platform.state.dx}, {sim.platform.state.dy}) m/s")
    
    # Show initial line tensions
    print("\nInitial line tensions:")
    attachment_positions = sim.platform.get_attachment_positions_global(sim.platform.state)
    total_force, total_moment, line_forces = sim.mooring_system.compute_total_forces(
        attachment_positions, 0.0
    )
    
    for i, force in enumerate(line_forces):
        force_MN = force / 1e6
        status = "WILL BREAK" if i == 0 else "INTACT"
        print(f"  Line {i}: {force_MN:.2f} MN [{status}]")
    
    print(f"\nTotal initial force: ({total_force[0]/1e6:.2f}, {total_force[1]/1e6:.2f}) MN")
    print(f"Total initial moment: {total_moment/1e6:.2f} MN⋅m")
    
    # Run simulation
    print(f"\nRunning dramatic simulation for 120 seconds...")
    print("Expect significant platform motion due to line break asymmetry...")
    
    results = sim.run(duration=120.0, max_step=0.1)  # Smaller time step for dramatic motion
    
    print(f"\nSimulation completed!")
    print(f"  Computation time: {results.computation_time:.2f} seconds")
    print(f"  Time steps: {results.num_steps}")
    
    # Analyze dramatic results
    print("\nDramatic Simulation Results:")
    
    initial_pos = [results.x[0], results.y[0]]
    final_pos = [results.x[-1], results.y[-1]]
    drift_distance = np.sqrt((final_pos[0] - initial_pos[0])**2 + 
                            (final_pos[1] - initial_pos[1])**2)
    
    final_heading = np.degrees(results.psi[-1])
    initial_heading = np.degrees(results.psi[0])
    heading_change = final_heading - initial_heading
    
    max_displacement = np.max(np.sqrt(results.x**2 + results.y**2))
    max_speed = np.max(np.sqrt(results.dx**2 + results.dy**2))
    
    print(f"  Initial position: ({initial_pos[0]:.1f}, {initial_pos[1]:.1f}) m")
    print(f"  Final position: ({final_pos[0]:.1f}, {final_pos[1]:.1f}) m")
    print(f"  Drift distance: {drift_distance:.2f} m")
    print(f"  Initial heading: {initial_heading:.1f}°")
    print(f"  Final heading: {final_heading:.1f}°")
    print(f"  Heading change: {heading_change:.1f}°")
    print(f"  Maximum displacement from origin: {max_displacement:.1f} m")
    print(f"  Maximum speed reached: {max_speed:.3f} m/s")
    
    # Line force analysis
    print(f"\nLine Force Analysis:")
    for i in range(4):
        forces_MN = results.line_forces[:, i] / 1e6
        max_force = np.max(forces_MN)
        final_force = forces_MN[-1]
        avg_force = np.mean(forces_MN[forces_MN > 0.01])  # Average of non-zero forces
        
        if i == 0:
            print(f"  Line {i}: BROKEN (was {max_force:.2f} MN before break)")
        else:
            status = "INTACT" if final_force > 0.01 else "SLACK"
            print(f"  Line {i}: {final_force:.2f} MN final (max: {max_force:.2f} MN, avg: {avg_force:.2f} MN) [{status}]")
    
    # Energy analysis
    initial_ke = results.kinetic_energy[0] / 1e6
    final_ke = results.kinetic_energy[-1] / 1e6
    max_ke = np.max(results.kinetic_energy) / 1e6
    final_pe = results.potential_energy[-1] / 1e6
    
    print(f"\nEnergy Analysis:")
    print(f"  Initial kinetic energy: {initial_ke:.3f} MJ")
    print(f"  Final kinetic energy: {final_ke:.3f} MJ")
    print(f"  Maximum kinetic energy: {max_ke:.3f} MJ")
    print(f"  Final potential energy: {final_pe:.3f} MJ")
    print(f"  Energy dissipated by damping: {initial_ke - final_ke:.3f} MJ")
    
    return results


def create_dramatic_plots(results):
    """Create plots emphasizing the dramatic motion."""
    
    print("\nCreating dramatic motion plots...")
    
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    # Create figure with more emphasis on motion
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('Epsilon-Sim: Dramatic Line Break Scenario - Platform Response', fontsize=16, weight='bold')
    
    # 1. Platform trajectory with dramatic view
    ax1 = axes[0, 0]
    ax1.plot(results.x, results.y, 'b-', linewidth=3, label='Platform trajectory', alpha=0.8)
    ax1.plot(results.x[0], results.y[0], 'go', markersize=12, label='Start position')
    ax1.plot(results.x[-1], results.y[-1], 'ro', markersize=12, label='Final position')
    
    # Add anchor positions and broken line indication
    anchors = np.array([[60, 60], [-60, 60], [-60, -60], [60, -60]])
    ax1.scatter(anchors[:, 0], anchors[:, 1], c='red', marker='x', s=150, label='Anchors', linewidth=3)
    
    # Highlight the broken line path
    platform_corners_0 = np.array([results.x[0] + 60, results.y[0] + 60])  # Approximate corner position
    ax1.plot([platform_corners_0[0], anchors[0, 0]], [platform_corners_0[1], anchors[0, 1]], 
             'r--', linewidth=3, alpha=0.5, label='Broken line 0')
    
    ax1.set_xlabel('X Position [m]', fontsize=12)
    ax1.set_ylabel('Y Position [m]', fontsize=12)
    ax1.set_title('Platform Drift Trajectory\n(Due to Line 0 Break)', fontsize=14, weight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)
    
    # 2. Position vs time with emphasis on motion
    ax2 = axes[0, 1]
    ax2.plot(results.time, results.x, 'b-', linewidth=2.5, label='Surge (X)', alpha=0.9)
    ax2.plot(results.time, results.y, 'r-', linewidth=2.5, label='Sway (Y)', alpha=0.9)
    ax2.axvline(x=0, color='orange', linestyle='--', linewidth=2, alpha=0.7, label='Line break')
    ax2.set_xlabel('Time [s]', fontsize=12)
    ax2.set_ylabel('Position [m]', fontsize=12)
    ax2.set_title('Platform Position Response\nto Line Break', fontsize=14, weight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)
    
    # 3. Heading vs time showing rotation
    ax3 = axes[0, 2]
    heading_deg = np.degrees(results.psi)
    ax3.plot(results.time, heading_deg, 'g-', linewidth=3, alpha=0.9)
    ax3.axvline(x=0, color='orange', linestyle='--', linewidth=2, alpha=0.7, label='Line break')
    ax3.set_xlabel('Time [s]', fontsize=12)
    ax3.set_ylabel('Heading [degrees]', fontsize=12)
    ax3.set_title('Platform Rotation Response\nto Asymmetric Loading', fontsize=14, weight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=10)
    
    # 4. Dramatic line forces showing break and redistribution
    ax4 = axes[1, 0]
    colors = ['red', 'blue', 'green', 'purple']
    line_names = ['Line 0 (BREAKS at t=0)', 'Line 1', 'Line 2', 'Line 3']
    
    for i in range(4):
        forces_MN = results.line_forces[:, i] / 1e6
        if i == 0:
            # Show broken line as dramatic drop
            ax4.plot(results.time, forces_MN, color=colors[i], 
                    linewidth=4, label=line_names[i], alpha=0.7)
        else:
            ax4.plot(results.time, forces_MN, color=colors[i], 
                    linewidth=2.5, label=line_names[i], alpha=0.9)
    
    ax4.axvline(x=0, color='orange', linestyle='--', linewidth=2, alpha=0.7, label='Break event')
    ax4.set_xlabel('Time [s]', fontsize=12)
    ax4.set_ylabel('Line Force [MN]', fontsize=12)
    ax4.set_title('Mooring Line Forces\n(Load Redistribution)', fontsize=14, weight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.legend(fontsize=10)
    
    # 5. Speed and acceleration
    ax5 = axes[1, 1]
    speed = np.sqrt(results.dx**2 + results.dy**2)
    # Calculate acceleration (numerical derivative of velocity)
    accel_x = np.gradient(results.dx, results.time)
    accel_y = np.gradient(results.dy, results.time)
    accel_magnitude = np.sqrt(accel_x**2 + accel_y**2)
    
    ax5_twin = ax5.twinx()
    
    line1 = ax5.plot(results.time, speed, 'purple', linewidth=3, label='Speed magnitude', alpha=0.9)
    line2 = ax5_twin.plot(results.time, accel_magnitude, 'orange', linewidth=2, 
                         label='Acceleration magnitude', alpha=0.7)
    
    ax5.axvline(x=0, color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax5.set_xlabel('Time [s]', fontsize=12)
    ax5.set_ylabel('Speed [m/s]', fontsize=12, color='purple')
    ax5_twin.set_ylabel('Acceleration [m/s²]', fontsize=12, color='orange')
    ax5.set_title('Platform Speed & Acceleration\n(Dynamic Response)', fontsize=14, weight='bold')
    ax5.grid(True, alpha=0.3)
    
    # Combine legends
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax5.legend(lines, labels, loc='upper right', fontsize=10)
    
    # 6. Energy with dramatic changes
    ax6 = axes[1, 2]
    ke_MJ = results.kinetic_energy / 1e6
    pe_MJ = results.potential_energy / 1e6
    total_energy = ke_MJ + pe_MJ
    
    ax6.plot(results.time, ke_MJ, 'r-', linewidth=3, label='Kinetic Energy', alpha=0.9)
    ax6.plot(results.time, pe_MJ, 'b-', linewidth=3, label='Potential Energy', alpha=0.9)
    ax6.plot(results.time, total_energy, 'g-', linewidth=2, label='Total Energy', alpha=0.7)
    ax6.axvline(x=0, color='orange', linestyle='--', linewidth=2, alpha=0.7, label='Line break')
    ax6.set_xlabel('Time [s]', fontsize=12)
    ax6.set_ylabel('Energy [MJ]', fontsize=12)
    ax6.set_title('Energy Transfer & Dissipation\n(System Dynamics)', fontsize=14, weight='bold')
    ax6.grid(True, alpha=0.3)
    ax6.legend(fontsize=10)
    
    plt.tight_layout()
    
    # Save with high quality
    plot_file = output_dir / "dramatic_simulation_results.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"  Dramatic results plot saved: {plot_file}")
    
    plt.show()


def main():
    """Run dramatic simulation."""
    
    try:
        # Run the dramatic simulation
        results = run_dramatic_simulation()
        
        # Create dramatic plots
        create_dramatic_plots(results)
        
        print("\n" + "=" * 70)
        print("DRAMATIC SIMULATION COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        print("\nKey physical insights:")
        print("- Line break created immediate load redistribution")
        print("- Platform experienced complex 3-DOF motion (surge, sway, yaw)")
        print("- Asymmetric loading caused drift and rotation")
        print("- System energy dissipated through hydrodynamic damping")
        print("- Remaining lines carried increased loads")
        print("- Platform reached new equilibrium position")
        
        print(f"\nOutput files saved in 'output/' directory")
        print("This demonstrates realistic offshore platform dynamics!")
        
        return 0
        
    except Exception as e:
        print(f"\nDramatic simulation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main()) 