#!/usr/bin/env python3
"""
Test script to verify the simulation implementation.

This script tests the basic functionality without requiring
all dependencies to be installed.
"""

import sys
import os
import numpy as np

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_platform_creation():
    """Test platform object creation and basic functionality."""
    print("Testing Platform creation...")
    
    try:
        from epsilon_sim.core.platform import Platform, PlatformState
        
        # Create platform with default parameters
        platform = Platform()
        print(f"  Platform created: {platform}")
        
        # Test state operations
        state = PlatformState(x=10.0, y=5.0, psi=0.1)
        print(f"  State created: x={state.x}, y={state.y}, psi={state.psi}")
        
        # Test coordinate transformations
        attachment_positions = platform.get_attachment_positions_global(state)
        print(f"  Attachment positions shape: {attachment_positions.shape}")
        
        # Test dynamics calculation
        external_force = np.array([1000.0, 500.0])  # N
        external_moment = 10000.0  # N⋅m
        
        state_derivative = platform.compute_dynamics_rhs(state, external_force, external_moment)
        print(f"  State derivative computed: shape={state_derivative.shape}")
        
        print("  Platform test PASSED")
        return True
        
    except Exception as e:
        print(f"  Platform test FAILED: {e}")
        return False


def test_mooring_system():
    """Test mooring system creation and force calculations."""
    print("Testing MooringSystem creation...")
    
    try:
        from epsilon_sim.physics.mooring import MooringSystem
        
        # Create mooring system
        anchor_positions = [[60, 60], [-60, 60], [-60, -60], [60, -60]]
        mooring = MooringSystem(anchor_positions)
        print(f"  MooringSystem created: {mooring}")
        
        # Test line breaking
        mooring.add_break_scenario(0, 0.0)
        print("  Line break scenario added")
        
        # Test force calculation with displaced platform to create tension
        attachment_positions = np.array([[65, 65], [-55, 65], [-55, -55], [65, -55]])  # Slightly displaced
        total_force, total_moment, line_forces = mooring.compute_total_forces(
            attachment_positions, current_time=1.0
        )
        
        print(f"  Forces computed: total_force={total_force}, total_moment={total_moment}")
        print(f"  Individual line forces: {line_forces}")
        
        # Test load factor adjustment
        mooring.set_line_load_factor(1, 1.5)
        print("  Load factor adjusted")
        
        print("  MooringSystem test PASSED")
        return True
        
    except Exception as e:
        print(f"  MooringSystem test FAILED: {e}")
        return False


def test_simulator():
    """Test main simulator functionality."""
    print("Testing PlatformSimulator...")
    
    try:
        from epsilon_sim.core.simulator import PlatformSimulator
        
        # Create simulator with default config
        sim = PlatformSimulator()
        print(f"  Simulator created: {sim}")
        
        # Add line break scenario
        sim.set_line_break(0, 0.0)
        print("  Line break configured")
        
        # Validate configuration
        warnings = sim.validate_configuration()
        if warnings:
            print(f"  Configuration warnings: {warnings}")
        else:
            print("  Configuration validated successfully")
        
        # Test dynamics function
        initial_state = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        state_derivative = sim.dynamics_rhs(0.0, initial_state)
        print(f"  Dynamics RHS computed: shape={state_derivative.shape}")
        
        print("  PlatformSimulator test PASSED")
        return True
        
    except Exception as e:
        print(f"  PlatformSimulator test FAILED: {e}")
        return False


def test_short_simulation():
    """Test running a very short simulation."""
    print("Testing short simulation run...")
    
    try:
        from epsilon_sim.core.simulator import PlatformSimulator
        
        # Create simulator with initial displacement to create line tension
        sim = PlatformSimulator()
        sim.set_line_break(0, 0.0)
        
        # Set initial position slightly off-center to create initial tension
        sim.platform.state.x = 5.0  # 5m displacement in x
        sim.platform.state.y = 2.0  # 2m displacement in y
        
        print("  Running 5-second simulation...")
        
        # Run very short simulation
        results = sim.run(duration=5.0, max_step=0.1)
        
        print(f"  Simulation completed!")
        print(f"    Duration: {results.final_time} s")
        print(f"    Time steps: {results.num_steps}")
        print(f"    Computation time: {results.computation_time:.3f} s")
        print(f"    Final position: ({results.x[-1]:.3f}, {results.y[-1]:.3f}) m")
        print(f"    Final heading: {np.degrees(results.psi[-1]):.1f}°")
        
        # Check that we have some motion due to line break
        displacement = np.sqrt(results.x[-1]**2 + results.y[-1]**2)
        if displacement > 0.1:
            print(f"    Platform moved {displacement:.3f} m (good!)")
        else:
            print(f"    Warning: Small displacement {displacement:.3f} m")
        
        print("  Short simulation test PASSED")
        return True
        
    except Exception as e:
        print(f"  Short simulation test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_configuration():
    """Test configuration loading."""
    print("Testing configuration system...")
    
    try:
        from epsilon_sim.utils.config import load_default_config, validate_config
        
        # Load default config
        config = load_default_config()
        print(f"  Default config loaded with {len(config)} sections")
        
        # Validate config
        errors = validate_config(config)
        if errors:
            print(f"  Configuration errors: {errors}")
            return False
        else:
            print("  Configuration validation PASSED")
        
        # Test specific values
        assert float(config['platform']['mass']) > 0, "Mass should be positive"
        assert int(config['mooring']['num_lines']) >= 3, "Should have at least 3 lines"
        print("  Configuration values check PASSED")
        
        return True
        
    except Exception as e:
        print(f"  Configuration test FAILED: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("EPSILON-SIM IMPLEMENTATION TEST")
    print("=" * 60)
    
    tests = [
        test_platform_creation,
        test_mooring_system,
        test_simulator,
        test_configuration,
        test_short_simulation,
    ]
    
    passed = 0
    total = len(tests)
    
    for test_func in tests:
        try:
            if test_func():
                passed += 1
            print()
        except Exception as e:
            print(f"  Test {test_func.__name__} CRASHED: {e}")
            print()
    
    print("=" * 60)
    print(f"TEST RESULTS: {passed}/{total} tests passed")
    print("=" * 60)
    
    if passed == total:
        print("All tests PASSED! The simulation implementation is working.")
        print("\nNext steps:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Try the example scripts in examples/")
        print("3. Run interactive visualization")
    else:
        print(f"Some tests failed. Please check the implementation.")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 