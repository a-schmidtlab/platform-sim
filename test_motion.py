#!/usr/bin/env python3
"""
Quick motion test for corrected 300m line setup.
"""

import numpy as np
import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

from src.epsilon_sim.core.simulator import PlatformSimulator

def test_motion():
    """Test if platform moves with new configuration."""
    print("="*50)
    print("MOTION TEST - 300m Lines + Tugboat Setup")
    print("="*50)
    
    # Create simulator
    sim = PlatformSimulator()
    
    # Extreme initial conditions to force motion
    print("Setting extreme initial conditions:")
    sim.platform.state.x = 80.0      # 80m surge
    sim.platform.state.y = 60.0      # 60m sway
    sim.platform.state.psi = 0.4     # 23° heading
    sim.platform.state.dx = -3.0     # Strong surge velocity
    sim.platform.state.dy = 2.0      # Strong sway velocity
    sim.platform.state.dpsi = 0.15   # Strong yaw rate
    
    print(f"  Initial position: ({sim.platform.state.x:.1f}, {sim.platform.state.y:.1f}) m")
    print(f"  Initial heading: {np.degrees(sim.platform.state.psi):.1f}°")
    print(f"  Initial velocities: ({sim.platform.state.dx:.1f}, {sim.platform.state.dy:.1f}) m/s")
    
    # Set very strong wave forces
    sim.platform.set_wave_excitation(
        amplitude=15.0e6,  # 15 MN
        frequency=0.1,     # 0.1 rad/s
        phase=np.pi/4
    )
    print(f"  Wave excitation: 15.0 MN at 0.1 rad/s")
    
    # Set line break
    sim.set_line_break(0, 5.0)
    print(f"  Line 0 breaks at t = 5.0 s")
    
    # Run short simulation
    print("\nRunning 30-second motion test...")
    results = sim.run(duration=30.0, max_step=0.05)
    
    # Analyze motion
    initial_pos = np.sqrt(results.x[0]**2 + results.y[0]**2)
    final_pos = np.sqrt(results.x[-1]**2 + results.y[-1]**2)
    position_change = abs(final_pos - initial_pos)
    
    max_speed = np.max(np.sqrt(results.dx**2 + results.dy**2))
    max_displacement = np.max(np.sqrt(results.x**2 + results.y**2))
    
    heading_change = abs(results.psi[-1] - results.psi[0])
    
    print(f"\nMotion Analysis:")
    print(f"  Initial distance from origin: {initial_pos:.1f} m")
    print(f"  Final distance from origin: {final_pos:.1f} m")
    print(f"  Position change: {position_change:.1f} m")
    print(f"  Maximum speed: {max_speed:.3f} m/s")
    print(f"  Maximum displacement: {max_displacement:.1f} m")
    print(f"  Heading change: {np.degrees(heading_change):.1f}°")
    
    # Motion criteria
    print(f"\nMotion Check:")
    if position_change > 10.0:
        print("✓ GOOD: Significant position change detected")
    else:
        print("✗ POOR: Insufficient position change")
        
    if max_speed > 0.5:
        print("✓ GOOD: Reasonable speeds achieved")
    else:
        print("✗ POOR: Speeds too low")
        
    if heading_change > 0.1:  # > 6 degrees
        print("✓ GOOD: Platform rotation detected")
    else:
        print("✗ POOR: No significant rotation")
    
    # Overall assessment
    motion_score = sum([
        position_change > 10.0,
        max_speed > 0.5,
        heading_change > 0.1
    ])
    
    print(f"\nOverall Motion Score: {motion_score}/3")
    
    if motion_score >= 2:
        print("🎉 SUCCESS: Platform is moving properly!")
        return True
    else:
        print("❌ FAILED: Platform motion insufficient")
        return False

if __name__ == "__main__":
    success = test_motion()
    exit(0 if success else 1) 