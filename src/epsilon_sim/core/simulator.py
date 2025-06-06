"""
Main simulation engine for floating platform dynamics.

This module contains the PlatformSimulator class that coordinates
platform dynamics, mooring system, and numerical integration.
"""

import numpy as np
from scipy.integrate import solve_ivp
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
import time

from .platform import Platform, PlatformState
from ..physics.mooring import MooringSystem
from ..utils.config import load_config


@dataclass
class SimulationResults:
    """Container for simulation results."""
    
    # Time vector
    time: np.ndarray
    
    # Platform state history
    x: np.ndarray           # Surge position [m]
    y: np.ndarray           # Sway position [m]
    psi: np.ndarray         # Yaw angle [rad]
    dx: np.ndarray          # Surge velocity [m/s]
    dy: np.ndarray          # Sway velocity [m/s]
    dpsi: np.ndarray        # Yaw rate [rad/s]
    
    # Force history
    line_forces: np.ndarray      # Individual line forces [N] (time x lines)
    total_force: np.ndarray      # Total force vector [N] (time x 2)
    total_moment: np.ndarray     # Total moment [N⋅m] (time,)
    
    # Energy history
    kinetic_energy: np.ndarray   # Kinetic energy [J]
    potential_energy: np.ndarray # Potential energy [J]
    
    # Simulation metadata
    computation_time: float      # Wall clock time [s]
    num_steps: int              # Number of integration steps
    final_time: float           # Final simulation time [s]
    
    def get_platform_trajectory(self) -> np.ndarray:
        """Get platform trajectory as Nx2 array."""
        return np.column_stack([self.x, self.y])
    
    def get_displacement_magnitude(self) -> np.ndarray:
        """Get magnitude of displacement from origin."""
        return np.sqrt(self.x**2 + self.y**2)
    
    def get_speed_magnitude(self) -> np.ndarray:
        """Get magnitude of platform speed."""
        return np.sqrt(self.dx**2 + self.dy**2)


class PlatformSimulator:
    """
    Main simulation engine for floating platform dynamics.
    
    Integrates platform dynamics with mooring system forces using
    numerical integration. Handles line breaking scenarios and
    provides comprehensive results.
    """
    
    def __init__(self, config_path: Optional[str] = None, **kwargs):
        """
        Initialize simulator with configuration.
        
        Args:
            config_path: Path to YAML configuration file
            **kwargs: Override configuration parameters
        """
        # Load configuration
        if config_path is None:
            # Use default configuration
            self.config = self._get_default_config()
        else:
            self.config = load_config(config_path)
        
        # Override with any provided kwargs
        self._update_config_with_kwargs(kwargs)
        
        # Initialize components
        self._initialize_platform()
        self._initialize_mooring_system()
        
        # Simulation state
        self.current_time = 0.0
        self.is_initialized = True
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration parameters."""
        return {
            'platform': {
                'mass': 1.25e7,
                'length': 120.0,
                'width': 120.0,
                'inertia_z': 8.0e9,
                'attachment_points': [
                    [60.0, 60.0], [-60.0, 60.0], 
                    [-60.0, -60.0], [60.0, -60.0]
                ]
            },
            'mooring': {
                'num_lines': 4,
                'length': 300.0,  # Corrected to 300m
                'stiffness': 1.2e9,
                'anchor_positions': [
                    [280.0, 280.0], [-280.0, 280.0],
                    [-280.0, -280.0], [280.0, -280.0]
                ]
            },
            'damping': {
                'linear_coeff': 3.5e6,
                'angular_coeff': 1.75e8
            },
            'simulation': {
                'duration': 120.0,
                'max_timestep': 0.25,
                'tolerance': 1.0e-6,
                'initial_position': [0.0, 0.0, 0.0],
                'initial_velocity': [0.0, 0.0, 0.0]
            }
        }
    
    def _update_config_with_kwargs(self, kwargs: Dict[str, Any]) -> None:
        """Update configuration with keyword arguments."""
        for key, value in kwargs.items():
            if '.' in key:
                # Handle nested keys like 'platform.mass'
                parts = key.split('.')
                config_section = self.config
                for part in parts[:-1]:
                    if part not in config_section:
                        config_section[part] = {}
                    config_section = config_section[part]
                config_section[parts[-1]] = value
            else:
                self.config[key] = value
    
    def _initialize_platform(self) -> None:
        """Initialize platform from configuration."""
        platform_config = self.config['platform']
        damping_config = self.config['damping']
        
        self.platform = Platform(
            mass=platform_config['mass'],
            length=platform_config['length'],
            width=platform_config['width'],
            inertia_z=platform_config['inertia_z'],
            linear_damping=damping_config['linear_coeff'],
            angular_damping=damping_config['angular_coeff'],
            attachment_points=platform_config.get('attachment_points')
        )
        
        # Set initial conditions
        sim_config = self.config['simulation']
        initial_pos = sim_config['initial_position']
        initial_vel = sim_config['initial_velocity']
        
        self.platform.state = PlatformState(
            x=initial_pos[0], y=initial_pos[1], psi=initial_pos[2],
            dx=initial_vel[0], dy=initial_vel[1], dpsi=initial_vel[2]
        )
    
    def _initialize_mooring_system(self) -> None:
        """Initialize mooring system from configuration."""
        mooring_config = self.config['mooring']
        
        self.mooring_system = MooringSystem(
            anchor_positions=mooring_config['anchor_positions'],
            unstretched_length=mooring_config['length'],
            stiffness=mooring_config['stiffness']
        )
    
    def set_line_break(self, line_id: int, break_time: float) -> None:
        """
        Configure a line breaking scenario.
        
        Args:
            line_id: ID of line to break (0-based)
            break_time: Time when line breaks [s]
        """
        self.mooring_system.add_break_scenario(line_id, break_time)
    
    def set_line_load_factor(self, line_id: int, load_factor: float) -> None:
        """
        Set load scaling factor for interactive control.
        
        Args:
            line_id: ID of line (0-based)
            load_factor: Load scaling factor (1.0 = normal)
        """
        self.mooring_system.set_line_load_factor(line_id, load_factor)
    
    def dynamics_rhs(self, t: float, state_vector: np.ndarray) -> np.ndarray:
        """
        Right-hand side of the dynamics equations for integration.
        
        Args:
            t: Current time [s]
            state_vector: State vector [x, y, psi, dx, dy, dpsi]
            
        Returns:
            State derivative [dx, dy, dpsi, ddx, ddy, ddpsi]
        """
        # Convert state vector to PlatformState
        state = PlatformState.from_array(state_vector)
        
        # Get attachment positions in global frame
        attachment_positions = self.platform.get_attachment_positions_global(state)
        
        # Compute mooring forces
        total_force, total_moment, _ = self.mooring_system.compute_total_forces(
            attachment_positions, t
        )
        
        # Compute platform dynamics with enhanced physics
        state_derivative = self.platform.compute_dynamics_rhs(
            state, total_force, total_moment, t
        )
        
        return state_derivative
    
    def run(
        self, 
        duration: Optional[float] = None,
        max_step: Optional[float] = None,
        rtol: Optional[float] = None
    ) -> SimulationResults:
        """
        Run the simulation.
        
        Args:
            duration: Simulation duration [s] (overrides config)
            max_step: Maximum integration step [s] (overrides config)
            rtol: Relative tolerance (overrides config)
            
        Returns:
            SimulationResults object with all results
        """
        start_time = time.time()
        
        # Get simulation parameters
        sim_config = self.config['simulation']
        duration = duration or sim_config['duration']
        max_step = max_step or sim_config['max_timestep']
        rtol = rtol or sim_config['tolerance']
        
        # Initial state
        initial_state = self.platform.state.to_array()
        
        # Time span
        t_span = (0.0, duration)
        
        print(f"Running simulation for {duration} seconds...")
        print(f"Platform: {self.platform}")
        print(f"Mooring: {self.mooring_system}")
        
        # Solve ODE
        solution = solve_ivp(
            self.dynamics_rhs,
            t_span,
            initial_state,
            method='DOP853',  # High-order Runge-Kutta
            max_step=max_step,
            rtol=rtol,
            atol=1e-9,
            dense_output=True
        )
        
        if not solution.success:
            raise RuntimeError(f"Integration failed: {solution.message}")
        
        # Extract results
        time_vector = solution.t
        state_history = solution.y
        
        # Reset mooring system status for clean force history computation
        self.mooring_system.reset_line_status()
        
        # Compute force history
        print("Computing force history...")
        line_forces_history = []
        total_forces_history = []
        total_moments_history = []
        kinetic_energy_history = []
        potential_energy_history = []
        
        for i, t in enumerate(time_vector):
            # Get state at this time
            state = PlatformState.from_array(state_history[:, i])
            
            # Update platform state (for energy calculation)
            self.platform.update_state(state_history[:, i])
            
            # Get attachment positions
            attachment_positions = self.platform.get_attachment_positions_global(state)
            
            # Compute forces
            total_force, total_moment, line_forces = self.mooring_system.compute_total_forces(
                attachment_positions, t
            )
            
            # Store results
            line_forces_history.append(line_forces)
            total_forces_history.append(total_force)
            total_moments_history.append(total_moment)
            kinetic_energy_history.append(self.platform.get_kinetic_energy())
            
            # Compute potential energy (elastic energy in lines)
            extensions = self.mooring_system.get_line_extensions(attachment_positions)
            potential_energy = 0.5 * self.mooring_system.stiffness * sum(ext**2 for ext in extensions)
            potential_energy_history.append(potential_energy)
        
        computation_time = time.time() - start_time
        
        print(f"Simulation completed in {computation_time:.2f} seconds")
        print(f"Number of time steps: {len(time_vector)}")
        
        # Create results object
        results = SimulationResults(
            time=time_vector,
            x=state_history[0, :],
            y=state_history[1, :],
            psi=state_history[2, :],
            dx=state_history[3, :],
            dy=state_history[4, :],
            dpsi=state_history[5, :],
            line_forces=np.array(line_forces_history),
            total_force=np.array(total_forces_history),
            total_moment=np.array(total_moments_history),
            kinetic_energy=np.array(kinetic_energy_history),
            potential_energy=np.array(potential_energy_history),
            computation_time=computation_time,
            num_steps=len(time_vector),
            final_time=duration
        )
        
        return results
    
    def get_equilibrium_position(self) -> PlatformState:
        """
        Compute equilibrium position with all lines intact.
        
        Returns:
            Equilibrium platform state
        """
        # Simple iterative solver for equilibrium
        # Start from current position
        state = PlatformState()
        
        for iteration in range(100):
            # Get attachment positions
            attachment_positions = self.platform.get_attachment_positions_global(state)
            
            # Compute forces
            total_force, total_moment, _ = self.mooring_system.compute_total_forces(
                attachment_positions, 0.0
            )
            
            # Check convergence
            force_magnitude = np.linalg.norm(total_force)
            if force_magnitude < 1.0 and abs(total_moment) < 1.0:
                break
            
            # Simple update (could be improved with Newton-Raphson)
            alpha = 1e-8  # Step size
            state.x -= alpha * total_force[0]
            state.y -= alpha * total_force[1]
            state.psi -= alpha * total_moment / self.platform.inertia_z
        
        return state
    
    def validate_configuration(self) -> List[str]:
        """
        Validate simulation configuration.
        
        Returns:
            List of validation warnings/errors
        """
        warnings = []
        
        # Check platform parameters
        if self.platform.mass <= 0:
            warnings.append("Platform mass must be positive")
        
        # Check mooring system
        if self.mooring_system.num_lines < 3:
            warnings.append("Insufficient mooring lines for stability")
        
        # Check time step stability
        sim_config = self.config['simulation']
        max_step = sim_config['max_timestep']
        
        # Estimate system natural frequency
        stiffness_matrix = self.mooring_system.get_system_stiffness_matrix(
            self.platform.attachment_points
        )
        if np.any(np.diag(stiffness_matrix) > 0):
            omega_max = np.sqrt(np.max(np.diag(stiffness_matrix)) / self.platform.mass)
            stable_dt = 0.1 / omega_max
            if max_step > stable_dt:
                warnings.append(f"Time step may be too large for stability. "
                               f"Consider max_step < {stable_dt:.3f} s")
        
        return warnings
    
    def __repr__(self) -> str:
        """String representation of simulator."""
        return (f"PlatformSimulator(platform={self.platform}, "
                f"mooring={self.mooring_system})") 