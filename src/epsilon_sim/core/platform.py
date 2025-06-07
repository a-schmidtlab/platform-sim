"""
Platform dynamics model for floating platform simulation.

This module contains the Platform class that represents a floating platform
with 3 degrees of freedom: surge (x), sway (y), and yaw (psi).
Enhanced with realistic hydrodynamic effects for proper oscillating motion.
"""

import numpy as np
from typing import Tuple, List, Optional
from dataclasses import dataclass


@dataclass
class PlatformState:
    """State vector for the platform dynamics."""
    
    # Position and orientation
    x: float = 0.0          # Surge position [m]
    y: float = 0.0          # Sway position [m] 
    psi: float = 0.0        # Yaw angle [rad]
    
    # Velocities
    dx: float = 0.0         # Surge velocity [m/s]
    dy: float = 0.0         # Sway velocity [m/s]
    dpsi: float = 0.0       # Yaw rate [rad/s]
    
    def to_array(self) -> np.ndarray:
        """Convert state to numpy array for integration."""
        return np.array([self.x, self.y, self.psi, self.dx, self.dy, self.dpsi])
    
    @classmethod
    def from_array(cls, state_array: np.ndarray) -> 'PlatformState':
        """Create state from numpy array."""
        return cls(
            x=state_array[0], y=state_array[1], psi=state_array[2],
            dx=state_array[3], dy=state_array[4], dpsi=state_array[5]
        )
    
    def get_position(self) -> np.ndarray:
        """Get position vector [x, y]."""
        return np.array([self.x, self.y])
    
    def get_velocity(self) -> np.ndarray:
        """Get velocity vector [dx, dy]."""
        return np.array([self.dx, self.dy])


class Platform:
    """
    Floating platform with 3-DOF dynamics.
    
    Models a semi-ballasted floating platform with surge, sway, and yaw motion.
    Includes enhanced hydrodynamic effects for realistic oscillating motion:
    - Added mass effects
    - Nonlinear damping
    - Wave-structure interaction
    - Restoring forces
    """
    
    def __init__(
        self, 
        mass: float = 1.25e7,
        length: float = 120.0,
        width: float = 120.0, 
        inertia_z: float = 8.0e9,
        linear_damping: float = None,    # Will be calculated based on literature
        angular_damping: float = None,   # Will be calculated based on literature
        attachment_points: Optional[List[List[float]]] = None
    ):
        """
        Initialize platform parameters with realistic hydrodynamic damping.
        
        Based on marine engineering literature for floating platforms:
        - Faltinsen (2005): "Sea Loads on Ships and Offshore Structures"
        - Newman (1977): "Marine Hydrodynamics" 
        - DNV-GL standards for offshore platforms
        
        Args:
            mass: Platform mass [kg]
            length: Platform length [m]
            width: Platform width [m]
            inertia_z: Yaw moment of inertia [kg⋅m²]
            linear_damping: Linear damping coefficient [N⋅s/m] (calculated if None)
            angular_damping: Angular damping coefficient [N⋅m⋅s/rad] (calculated if None)
            attachment_points: Mooring line attachment points in body frame [[x,y], ...]
        """
        # Mass properties
        self.mass = mass
        self.length = length
        self.width = width
        self.inertia_z = inertia_z
        
        # Calculate realistic hydrodynamic damping coefficients based on literature
        if linear_damping is None:
            # Linear damping calculation based on platform geometry and fluid mechanics
            # For a large floating platform in water:
            # B_surge/sway ≈ 0.5 * ρ * C_d * A * U_typical
            # Where: ρ = water density (1025 kg/m³)
            #        C_d = drag coefficient (≈ 1.2 for square platform)
            #        A = projected area (length × draft, assume 10m draft)
            #        U_typical = typical velocity for linearization (≈ 1 m/s)
            
            rho_water = 1025.0  # kg/m³
            drag_coefficient = 1.2  # For square/rectangular platform
            draft = 10.0  # m, assumed platform draft
            projected_area = length * draft  # m²
            typical_velocity = 1.0  # m/s for linearization
            
            # Linear damping coefficient from viscous drag
            linear_damping = 0.5 * rho_water * drag_coefficient * projected_area * typical_velocity
            
            # Add radiation damping (wave-making resistance)
            # B_radiation ≈ k * √(ρ * g * A_waterplane) for surge/sway
            # Where A_waterplane is the waterplane area
            g = 9.81  # m/s²
            waterplane_area = length * width  # m²
            radiation_damping_factor = 0.3  # Typical for semi-submersible platforms
            radiation_damping = radiation_damping_factor * np.sqrt(rho_water * g * waterplane_area)
            
            # Total linear damping
            linear_damping = linear_damping + radiation_damping
            
            # Scale up for large platform - empirical correction factor
            scale_factor = 3.0  # Based on model test correlations for large platforms
            linear_damping *= scale_factor
            
        if angular_damping is None:
            # Angular damping calculation for yaw motion
            # B_yaw ≈ 0.5 * ρ * C_d * L⁴ * ω_typical
            # Where: L is characteristic length
            #        ω_typical is typical angular velocity (≈ 0.1 rad/s)
            
            char_length = max(length, width)  # m
            typical_angular_velocity = 0.1  # rad/s for linearization
            
            # Viscous yaw damping
            angular_damping = 0.5 * rho_water * drag_coefficient * (char_length**4) * typical_angular_velocity
            
            # Add rotational wave damping
            rotational_wave_damping = 0.2 * rho_water * g * (waterplane_area * char_length**2) / 100
            angular_damping += rotational_wave_damping
            
            # Scale up for realistic platform response
            angular_scale_factor = 2.5
            angular_damping *= angular_scale_factor
        
        # Store calculated damping coefficients
        self.linear_damping = linear_damping
        self.angular_damping = angular_damping
        
        # Print calculated values for verification
        print(f"Calculated hydrodynamic damping coefficients:")
        print(f"  Linear damping: {linear_damping:.2e} N⋅s/m")
        print(f"  Angular damping: {angular_damping:.2e} N⋅m⋅s/rad")
        print(f"  Critical damping ratio (surge): {linear_damping / (2 * np.sqrt(mass * 1e5)):.3f}")
        
        # Added mass coefficients (typical for semi-submersible platforms)
        self.added_mass_surge = 0.15 * mass    # 15% of platform mass
        self.added_mass_sway = 0.15 * mass     # 15% of platform mass  
        self.added_inertia_yaw = 0.10 * inertia_z  # 10% of platform inertia
        
        # Total mass matrix including added mass effects
        self.mass_matrix = np.array([
            [mass + self.added_mass_surge, 0.0, 0.0],
            [0.0, mass + self.added_mass_sway, 0.0],
            [0.0, 0.0, inertia_z + self.added_inertia_yaw]
        ])
        
        # Base damping matrix  
        self.damping_matrix = np.array([
            [linear_damping, 0.0, 0.0],
            [0.0, linear_damping, 0.0],
            [0.0, 0.0, angular_damping]
        ])
        
        # Hydrostatic restoring coefficients (for stability)
        self.restoring_stiffness = np.array([
            [0.0, 0.0, 0.0],          # No surge restoring force
            [0.0, 0.0, 0.0],          # No sway restoring force  
            [0.0, 0.0, 2.5e8]         # Yaw restoring moment
        ])
        
        # Default attachment points (corners of platform square)
        if attachment_points is None:
            half_l, half_w = length/2, width/2
            self.attachment_points = np.array([
                [half_l, half_w],      # Top-right corner
                [-half_l, half_w],     # Top-left corner
                [-half_l, -half_w],    # Bottom-left corner
                [half_l, -half_w]      # Bottom-right corner
            ])
        else:
            self.attachment_points = np.array(attachment_points)
        
        # Current state
        self.state = PlatformState()
        
        # Wave excitation parameters
        self.wave_amplitude = 0.0    # Can be set for wave effects
        self.wave_frequency = 0.0
        self.wave_phase = 0.0
    
    def set_wave_excitation(self, amplitude: float, frequency: float, phase: float = 0.0):
        """
        Set wave excitation parameters for oscillating motion.
        
        Args:
            amplitude: Wave force amplitude [N]
            frequency: Wave frequency [rad/s]
            phase: Wave phase [rad]
        """
        self.wave_amplitude = amplitude
        self.wave_frequency = frequency
        self.wave_phase = phase
    
    def compute_nonlinear_damping(self, velocity: np.ndarray) -> np.ndarray:
        """
        Compute velocity-dependent nonlinear damping forces.
        
        Args:
            velocity: Velocity vector [dx, dy, dpsi]
            
        Returns:
            Nonlinear damping forces [Fx, Fy, Mz]
        """
        # Quadratic damping (proportional to |v|*v)
        # Reduced coefficient since linear damping is now much higher
        quad_damping_coeff = 0.05  # Further reduced since linear damping increased
        
        # Linear velocities
        vel_surge = velocity[0]
        vel_sway = velocity[1] 
        vel_yaw = velocity[2]
        
        # Quadratic damping forces
        damping_surge = -quad_damping_coeff * self.linear_damping * abs(vel_surge) * vel_surge
        damping_sway = -quad_damping_coeff * self.linear_damping * abs(vel_sway) * vel_sway
        damping_yaw = -quad_damping_coeff * self.angular_damping * abs(vel_yaw) * vel_yaw
        
        return np.array([damping_surge, damping_sway, damping_yaw])
    
    def compute_wave_forces(self, current_time: float) -> np.ndarray:
        """
        Compute realistic wave excitation forces for continuous oscillating motion.
        
        Args:
            current_time: Current simulation time [s]
            
        Returns:
            Wave forces [Fx, Fy, Mz]
        """
        if self.wave_amplitude == 0.0:
            return np.zeros(3)
        
        # Multi-frequency wave spectrum for realistic excitation
        # Primary wave frequency
        omega1 = self.wave_frequency
        phase1 = omega1 * current_time + self.wave_phase
        
        # Secondary frequencies for irregular waves
        omega2 = 0.8 * omega1  # Lower frequency component
        omega3 = 1.3 * omega1  # Higher frequency component
        
        phase2 = omega2 * current_time + np.pi/3
        phase3 = omega3 * current_time + np.pi/6
        
        # Combined wave forces with realistic spectrum
        force_surge = (self.wave_amplitude * np.sin(phase1) + 
                      0.4 * self.wave_amplitude * np.sin(phase2) +
                      0.2 * self.wave_amplitude * np.sin(phase3))
        
        force_sway = (0.6 * self.wave_amplitude * np.sin(phase1 + np.pi/4) + 
                     0.3 * self.wave_amplitude * np.sin(phase2 + np.pi/2) +
                     0.1 * self.wave_amplitude * np.sin(phase3 + np.pi/8))
        
        # Yaw moment from wave directionality and platform asymmetry
        moment_yaw = (0.15 * self.wave_amplitude * self.length * np.sin(phase1 + np.pi/6) +
                     0.08 * self.wave_amplitude * self.length * np.sin(phase2 + np.pi/3))
        
        return np.array([force_surge, force_sway, moment_yaw])
    
    def compute_memory_effects(self, state: PlatformState, current_time: float) -> np.ndarray:
        """
        Compute fluid memory effects (simplified radiation damping).
        
        Args:
            state: Current platform state
            current_time: Current simulation time [s]
            
        Returns:
            Memory effect forces [Fx, Fy, Mz]
        """
        # Simplified radiation damping based on acceleration history
        # This creates additional phase lag and oscillation characteristics
        
        velocity = np.array([state.dx, state.dy, state.dpsi])
        
        # Memory damping coefficients
        memory_coeff = 0.05
        
        memory_forces = -memory_coeff * self.mass_matrix @ velocity
        
        return memory_forces
        
    def get_rotation_matrix(self, psi: float) -> np.ndarray:
        """
        Get 2D rotation matrix for coordinate transformation.
        
        Args:
            psi: Yaw angle [rad]
            
        Returns:
            2x2 rotation matrix from body to global frame
        """
        cos_psi = np.cos(psi)
        sin_psi = np.sin(psi)
        
        return np.array([
            [cos_psi, -sin_psi],
            [sin_psi, cos_psi]
        ])
    
    def body_to_global(self, body_points: np.ndarray, state: PlatformState) -> np.ndarray:
        """
        Transform points from body frame to global frame.
        
        Args:
            body_points: Points in body frame (Nx2 array)
            state: Current platform state
            
        Returns:
            Points in global frame (Nx2 array)
        """
        R = self.get_rotation_matrix(state.psi)
        position = state.get_position()
        
        # Handle single point or array of points
        if body_points.ndim == 1:
            return R @ body_points + position
        else:
            return (R @ body_points.T).T + position
    
    def global_to_body(self, global_points: np.ndarray, state: PlatformState) -> np.ndarray:
        """
        Transform points from global frame to body frame.
        
        Args:
            global_points: Points in global frame (Nx2 array)
            state: Current platform state
            
        Returns:
            Points in body frame (Nx2 array)
        """
        R = self.get_rotation_matrix(-state.psi)  # Inverse rotation
        position = state.get_position()
        
        # Handle single point or array of points
        if global_points.ndim == 1:
            return R @ (global_points - position)
        else:
            return (R @ (global_points - position).T).T
    
    def get_attachment_positions_global(self, state: PlatformState) -> np.ndarray:
        """
        Get mooring line attachment positions in global frame.
        
        Args:
            state: Current platform state
            
        Returns:
            Attachment positions in global frame (Nx2 array)
        """
        return self.body_to_global(self.attachment_points, state)
    
    def compute_dynamics_rhs(
        self, 
        state: PlatformState, 
        external_forces: np.ndarray, 
        external_moment: float,
        current_time: float = 0.0
    ) -> np.ndarray:
        """
        Compute right-hand side of dynamics equations with enhanced physics.
        
        Args:
            state: Current platform state
            external_forces: External forces [Fx, Fy] in global frame [N]
            external_moment: External moment about z-axis [N⋅m]
            current_time: Current simulation time [s]
            
        Returns:
            State derivative [dx, dy, dpsi, ddx, ddy, ddpsi]
        """
        # Current velocity vector
        velocity = np.array([state.dx, state.dy, state.dpsi])
        
        # Current position vector for restoring forces
        position = np.array([state.x, state.y, state.psi])
        
        # External force vector
        external_force_vector = np.array([
            external_forces[0], 
            external_forces[1], 
            external_moment
        ])
        
        # Linear damping forces
        linear_damping_forces = -self.damping_matrix @ velocity
        
        # Nonlinear damping forces  
        nonlinear_damping_forces = self.compute_nonlinear_damping(velocity)
        
        # Hydrostatic restoring forces
        restoring_forces = -self.restoring_stiffness @ position
        
        # Wave excitation forces
        wave_forces = self.compute_wave_forces(current_time)
        
        # Fluid memory effects
        memory_forces = self.compute_memory_effects(state, current_time)
        
        # Total forces
        total_forces = (external_force_vector + 
                       linear_damping_forces + 
                       nonlinear_damping_forces + 
                       restoring_forces + 
                       wave_forces + 
                       memory_forces)
        
        # Solve for accelerations: M * a = F
        try:
            accelerations = np.linalg.solve(self.mass_matrix, total_forces)
        except np.linalg.LinAlgError:
            # Fallback if matrix is singular
            accelerations = np.zeros(3)
        
        # Return state derivative
        return np.array([
            state.dx,           # dx/dt = dx
            state.dy,           # dy/dt = dy  
            state.dpsi,         # dpsi/dt = dpsi
            accelerations[0],   # ddx/dt = ddx
            accelerations[1],   # ddy/dt = ddy
            accelerations[2]    # ddpsi/dt = ddpsi
        ])

    def update_state(self, new_state_array: np.ndarray) -> None:
        """Update platform state from array."""
        self.state = PlatformState.from_array(new_state_array)

    def get_kinetic_energy(self) -> float:
        """Compute platform kinetic energy."""
        velocity = np.array([self.state.dx, self.state.dy, self.state.dpsi])
        return 0.5 * velocity.T @ self.mass_matrix @ velocity

    def get_platform_corners(self, state: Optional[PlatformState] = None) -> np.ndarray:
        """
        Get platform corner positions in global frame.
        
        Args:
            state: Platform state (uses current state if None)
            
        Returns:
            Corner positions in global frame (4x2 array)
        """
        if state is None:
            state = self.state
        
        # Corner positions in body frame
        half_l, half_w = self.length/2, self.width/2
        corners_body = np.array([
            [-half_l, -half_w],  # Rear-left
            [half_l, -half_w],   # Rear-right  
            [half_l, half_w],    # Front-right
            [-half_l, half_w]    # Front-left
        ])
        
        return self.body_to_global(corners_body, state)

    def __repr__(self) -> str:
        """String representation."""
        return (f"Platform(mass={self.mass:.1e}, length={self.length}, "
                f"width={self.width}, state=({self.state.x:.1f}, "
                f"{self.state.y:.1f}, {np.degrees(self.state.psi):.1f}°))") 