#!/usr/bin/env python3
"""
Verification script to test enhanced physics and oscillating motion.
Runs a short simulation and analyzes motion characteristics.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

from src.epsilon_sim.core.simulator import PlatformSimulator


def analyze_oscillations(time, signal, signal_name="Signal"):
    """Analyze oscillation characteristics of a signal."""
    # Remove trend
    signal_detrended = signal - np.mean(signal)
    
    # Find zero crossings
    zero_crossings = 0
    for i in range(1, len(signal_detrended)):
        if signal_detrended[i-1] * signal_detrended[i] < 0:
            zero_crossings += 1
    
    # Calculate statistics
    amplitude = np.std(signal_detrended)
    max_value = np.max(np.abs(signal_detrended))
    oscillations = zero_crossings // 2
    
    # Estimate period from zero crossings
    if zero_crossings > 2:
        period_estimate = 2 * (time[-1] - time[0]) / zero_crossings
    else:
        period_estimate = 0
    
    print(f"\n{signal_name} Analysis:")
    print(f"  RMS Amplitude: {amplitude:.3f}")
    print(f"  Maximum: {max_value:.3f}")
    print(f"  Zero Crossings: {zero_crossings}")
    print(f"  Full Oscillations: {oscillations}")
    print(f"  Estimated Period: {period_estimate:.1f} s")
    
    return {
        'amplitude': amplitude,
        'max_value': max_value,
        'oscillations': oscillations,
        'period': period_estimate
    }


def main():
    """Run verification simulation and analyze results."""
    print("="*60)
    print("ENHANCED PHYSICS VERIFICATION")
    print("="*60)
    
    # Create simulator with enhanced physics
    sim = PlatformSimulator()
    
    # Set dramatic initial conditions
    print("\nSetting enhanced initial conditions...")
    sim.platform.state.x = 45.0      # Large surge displacement
    sim.platform.state.y = 35.0      # Large sway displacement  
    sim.platform.state.psi = 0.25    # 14° heading
    sim.platform.state.dx = -1.8     # Strong surge velocity
    sim.platform.state.dy = 1.2      # Strong sway velocity
    sim.platform.state.dpsi = 0.08   # Significant yaw rate
    
    # Configure wave excitation
    sim.platform.set_wave_excitation(
        amplitude=8.0e6,    # 8 MN force amplitude
        frequency=0.15,     # 0.15 rad/s (42 s period)
        phase=np.pi/4
    )
    
    # Set line break
    sim.set_line_break(0, 10.0)  # Line 0 breaks at t=10s
    
    print(f"Initial position: ({sim.platform.state.x:.1f}, {sim.platform.state.y:.1f}) m")
    print(f"Initial heading: {np.degrees(sim.platform.state.psi):.1f}°")
    print(f"Initial velocities: ({sim.platform.state.dx:.2f}, {sim.platform.state.dy:.2f}) m/s")
    print(f"Wave excitation: 8.0 MN at 0.15 rad/s")
    print(f"Line break at t = 10.0 s")
    
    # Run simulation
    print("\nRunning enhanced physics simulation...")
    duration = 80.0
    results = sim.run(duration=duration, max_step=0.01, rtol=1e-8)
    
    print(f"Simulation completed: {results.num_steps} steps in {results.computation_time:.2f} s")
    
    # Analyze motion characteristics
    print("\n" + "="*60)
    print("MOTION ANALYSIS")
    print("="*60)
    
    # Overall statistics
    max_displacement = np.max(np.sqrt(results.x**2 + results.y**2))
    max_speed = np.max(np.sqrt(results.dx**2 + results.dy**2))
    max_heading = np.max(np.abs(results.psi))
    max_yaw_rate = np.max(np.abs(results.dpsi))
    
    print(f"\nOverall Motion Statistics:")
    print(f"  Maximum displacement: {max_displacement:.1f} m")
    print(f"  Maximum speed: {max_speed:.3f} m/s")
    print(f"  Maximum heading: {np.degrees(max_heading):.1f}°")
    print(f"  Maximum yaw rate: {np.degrees(max_yaw_rate):.3f} °/s")
    
    # Analyze individual motions
    surge_analysis = analyze_oscillations(results.time, results.x, "Surge Motion")
    sway_analysis = analyze_oscillations(results.time, results.y, "Sway Motion")
    yaw_analysis = analyze_oscillations(results.time, np.degrees(results.psi), "Yaw Motion")
    
    # Check for proper oscillating behavior
    print("\n" + "="*60)
    print("OSCILLATION VERIFICATION")
    print("="*60)
    
    # Criteria for realistic oscillating motion
    criteria_met = 0
    total_criteria = 5
    
    # 1. Sufficient displacement amplitude
    if surge_analysis['amplitude'] > 5.0:  # > 5m RMS amplitude
        print("✓ Surge amplitude sufficient for oscillations")
        criteria_met += 1
    else:
        print("✗ Surge amplitude too small")
    
    # 2. Multiple oscillations detected
    if surge_analysis['oscillations'] >= 2:
        print("✓ Multiple surge oscillations detected")
        criteria_met += 1
    else:
        print("✗ Insufficient surge oscillations")
    
    # 3. Realistic oscillation period
    if 20 <= surge_analysis['period'] <= 80:  # 20-80 second period range
        print("✓ Realistic oscillation period")
        criteria_met += 1
    else:
        print("✗ Unrealistic oscillation period")
    
    # 4. Significant yaw motion
    if yaw_analysis['amplitude'] > 2.0:  # > 2° RMS yaw amplitude
        print("✓ Significant yaw oscillations")
        criteria_met += 1
    else:
        print("✗ Insufficient yaw motion")
    
    # 5. Dynamic response to line break
    break_idx = np.argmin(np.abs(results.time - 10.0))
    if break_idx < len(results.time) - 1:
        pre_break_speed = np.mean(np.sqrt(results.dx[:break_idx]**2 + results.dy[:break_idx]**2))
        post_break_speed = np.mean(np.sqrt(results.dx[break_idx:]**2 + results.dy[break_idx:]**2))
        if post_break_speed > 1.5 * pre_break_speed:
            print("✓ Clear dynamic response to line break")
            criteria_met += 1
        else:
            print("✗ Insufficient response to line break")
    else:
        print("✗ Break time not found in simulation")
    
    # Overall assessment
    print(f"\nOverall Assessment: {criteria_met}/{total_criteria} criteria met")
    
    if criteria_met >= 4:
        print("🎉 EXCELLENT: Realistic oscillating motion achieved!")
    elif criteria_met >= 3:
        print("✅ GOOD: Oscillating motion present, may need minor tuning")
    elif criteria_met >= 2:
        print("⚠️  FAIR: Some oscillation, but needs improvement")
    else:
        print("❌ POOR: No realistic oscillating motion detected")
    
    # Create verification plot
    print("\nGenerating verification plots...")
    create_verification_plots(results)
    
    return criteria_met >= 3


def create_verification_plots(results):
    """Create plots to verify oscillating motion."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Enhanced Physics Verification', fontsize=14, fontweight='bold')
    
    # Trajectory plot
    axes[0, 0].plot(results.x, results.y, 'b-', linewidth=2, alpha=0.7)
    axes[0, 0].plot(results.x[0], results.y[0], 'go', markersize=8, label='Start')
    axes[0, 0].plot(results.x[-1], results.y[-1], 'ro', markersize=8, label='End')
    axes[0, 0].set_xlabel('Surge [m]')
    axes[0, 0].set_ylabel('Sway [m]')
    axes[0, 0].set_title('Platform Trajectory')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()
    axes[0, 0].set_aspect('equal')
    
    # Position time series
    axes[0, 1].plot(results.time, results.x, 'b-', label='Surge')
    axes[0, 1].plot(results.time, results.y, 'r-', label='Sway')
    axes[0, 1].axvline(x=10.0, color='k', linestyle='--', alpha=0.7, label='Line Break')
    axes[0, 1].set_xlabel('Time [s]')
    axes[0, 1].set_ylabel('Position [m]')
    axes[0, 1].set_title('Position vs Time')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()
    
    # Velocity time series  
    speed = np.sqrt(results.dx**2 + results.dy**2)
    axes[1, 0].plot(results.time, speed, 'g-', linewidth=2)
    axes[1, 0].axvline(x=10.0, color='k', linestyle='--', alpha=0.7, label='Line Break')
    axes[1, 0].set_xlabel('Time [s]')
    axes[1, 0].set_ylabel('Speed [m/s]')
    axes[1, 0].set_title('Platform Speed')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()
    
    # Yaw motion
    axes[1, 1].plot(results.time, np.degrees(results.psi), 'm-', linewidth=2)
    axes[1, 1].axvline(x=10.0, color='k', linestyle='--', alpha=0.7, label='Line Break')
    axes[1, 1].set_xlabel('Time [s]')
    axes[1, 1].set_ylabel('Heading [°]')
    axes[1, 1].set_title('Platform Heading')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig('verification_plots.png', dpi=150, bbox_inches='tight')
    print("Verification plots saved as 'verification_plots.png'")
    

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1) 