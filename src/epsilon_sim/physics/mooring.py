"""
Mooring system dynamics for floating platform simulation.

This module contains classes for modeling mooring lines as linear springs
with breaking capabilities, force calculations, and impact load analysis.
"""

import numpy as np
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass
from enum import Enum


class LineStatus(Enum):
    """Status of a mooring line."""
    INTACT = "intact"
    BROKEN = "broken"
    SLACK = "slack"
    OVERLOADED = "overloaded"  # New status for overload detection


@dataclass
class LineImpactData:
    """Impact load analysis data for a mooring line."""
    
    # Load characteristics
    current_force: float = 0.0           # Current force [N]
    peak_force: float = 0.0              # Peak force since last break [N]
    working_load: float = 0.0            # Normal working load [N]
    ultimate_strength: float = 0.0       # Ultimate breaking load [N]
    
    # Safety factors
    safety_factor: float = 0.0           # Current safety factor
    min_safety_factor: float = 0.0       # Minimum safety factor since break
    
    # Impact analysis
    force_increase_rate: float = 0.0     # Rate of force increase [N/s]
    time_to_failure: float = float('inf') # Estimated time to failure [s]
    overload_risk: str = "LOW"           # Risk level: LOW, MEDIUM, HIGH, CRITICAL
    
    # Historical data
    force_history: List[float] = None    # Force history for impact analysis
    time_history: List[float] = None     # Time history
    
    def __post_init__(self):
        """Initialize lists if not provided."""
        if self.force_history is None:
            self.force_history = []
        if self.time_history is None:
            self.time_history = []


@dataclass
class MooringLine:
    """Individual mooring line properties and state."""
    
    # Line properties
    unstretched_length: float     # Unstretched length [m]
    stiffness: float             # Axial stiffness EA [N]
    anchor_position: np.ndarray  # Anchor position in global frame [m]
    attachment_id: int           # ID of attachment point on platform
    
    # Line state
    status: LineStatus = LineStatus.INTACT
    break_time: Optional[float] = None
    load_factor: float = 1.0     # Load scaling factor (for interactive control)
    
    # Current values
    current_length: float = 0.0
    current_force: float = 0.0
    force_vector: np.ndarray = None
    
    # Impact load analysis
    working_load_limit: float = 8.0e6    # Working load limit [N] (8 MN)
    ultimate_strength: float = 12.0e6    # Ultimate breaking load [N] (12 MN) 
    impact_data: LineImpactData = None
    
    def __post_init__(self):
        """Initialize computed values."""
        if self.force_vector is None:
            self.force_vector = np.zeros(2)
        if self.impact_data is None:
            self.impact_data = LineImpactData()
            self.impact_data.working_load = self.working_load_limit
            self.impact_data.ultimate_strength = self.ultimate_strength


class MooringSystem:
    """
    Mooring system with multiple lines and impact load analysis.
    
    Manages all mooring lines, computes forces, handles line breaking,
    and analyzes impact loads and overload risks.
    """
    
    def __init__(
        self,
        anchor_positions: List[List[float]],
        unstretched_length: float = 300.0,  # Corrected to 300m
        stiffness: float = 1.2e9,
        attachment_ids: Optional[List[int]] = None
    ):
        """
        Initialize mooring system with impact load analysis.
        
        Args:
            anchor_positions: List of anchor positions [[x, y], ...]
            unstretched_length: Unstretched line length [m]
            stiffness: Line axial stiffness EA [N]
            attachment_ids: Platform attachment point IDs for each line
        """
        self.unstretched_length = unstretched_length
        self.stiffness = stiffness
        
        # Create mooring lines
        self.lines: List[MooringLine] = []
        
        if attachment_ids is None:
            attachment_ids = list(range(len(anchor_positions)))
        
        for i, (anchor_pos, attach_id) in enumerate(zip(anchor_positions, attachment_ids)):
            line = MooringLine(
                unstretched_length=unstretched_length,
                stiffness=stiffness,
                anchor_position=np.array(anchor_pos),
                attachment_id=attach_id
            )
            self.lines.append(line)
        
        self.num_lines = len(self.lines)
        
        # Break scenarios
        self.break_scenarios: List[Dict] = []
        
        # Impact load tracking
        self.last_break_time: Optional[float] = None
        self.pre_break_forces: Optional[np.ndarray] = None
        self.impact_analysis_active: bool = False
    
    def add_break_scenario(self, line_id: int, break_time: float) -> None:
        """
        Add a line breaking scenario.
        
        Args:
            line_id: ID of line to break (0-based)
            break_time: Time when line breaks [s]
        """
        self.break_scenarios.append({
            'line_id': line_id,
            'break_time': break_time
        })
        
        # Also store in the line itself
        if 0 <= line_id < self.num_lines:
            self.lines[line_id].break_time = break_time
    
    def set_line_load_factor(self, line_id: int, load_factor: float) -> None:
        """
        Set load scaling factor for a line (for interactive control).
        
        Args:
            line_id: ID of line (0-based)
            load_factor: Load scaling factor (1.0 = normal, 0.0 = no load)
        """
        if 0 <= line_id < self.num_lines:
            self.lines[line_id].load_factor = max(0.0, load_factor)
    
    def update_line_status(self, current_time: float) -> None:
        """
        Update line status based on break scenarios and overload conditions.
        
        Args:
            current_time: Current simulation time [s]
        """
        for i, line in enumerate(self.lines):
            # Check for scheduled breaks
            if (line.break_time is not None and 
                current_time >= line.break_time and 
                line.status == LineStatus.INTACT):
                
                # Store pre-break forces for impact analysis
                if self.pre_break_forces is None:
                    self.pre_break_forces = np.array([l.current_force for l in self.lines])
                
                line.status = LineStatus.BROKEN
                self.last_break_time = current_time
                self.impact_analysis_active = True
                print(f"  Line {i} broke at t={current_time:.2f}s")
            
            # Check for overload conditions
            if (line.status == LineStatus.INTACT and 
                line.current_force > line.working_load_limit):
                line.status = LineStatus.OVERLOADED
                print(f"  WARNING: Line {i} overloaded! Force: {line.current_force/1e6:.2f} MN")
            
            # Check for potential failure due to extreme overload
            if (line.status in [LineStatus.INTACT, LineStatus.OVERLOADED] and
                line.current_force > line.ultimate_strength):
                line.status = LineStatus.BROKEN
                print(f"  CRITICAL: Line {i} failed due to overload! Force: {line.current_force/1e6:.2f} MN")
    
    def analyze_impact_loads(self, current_time: float) -> Dict:
        """
        Analyze impact loads and overload risks after line breaks.
        
        Args:
            current_time: Current simulation time [s]
            
        Returns:
            Dictionary with impact analysis results
        """
        if not self.impact_analysis_active or self.last_break_time is None:
            return {}
        
        analysis = {
            'time_since_break': current_time - self.last_break_time,
            'intact_lines': [],
            'overloaded_lines': [],
            'force_redistribution': {},
            'total_force_increase': 0.0,
            'max_overload_ratio': 0.0,
            'critical_lines': [],
            'estimated_failures': []
        }
        
        total_current_force = 0.0
        total_pre_break_force = 0.0 if self.pre_break_forces is None else np.sum(self.pre_break_forces)
        
        for i, line in enumerate(self.lines):
            if line.status == LineStatus.BROKEN:
                continue
                
            # Update impact data
            impact = line.impact_data
            impact.current_force = line.current_force
            impact.peak_force = max(impact.peak_force, line.current_force)
            
            # Calculate safety factor
            if line.ultimate_strength > 0:
                impact.safety_factor = line.ultimate_strength / max(line.current_force, 1.0)
                impact.min_safety_factor = min(impact.min_safety_factor, impact.safety_factor) if impact.min_safety_factor > 0 else impact.safety_factor
            
            # Force increase analysis
            if self.pre_break_forces is not None:
                force_increase = line.current_force - self.pre_break_forces[i]
                analysis['force_redistribution'][f'line_{i}'] = {
                    'pre_break': self.pre_break_forces[i] / 1e6,  # MN
                    'current': line.current_force / 1e6,         # MN
                    'increase': force_increase / 1e6,            # MN
                    'increase_percent': (force_increase / max(self.pre_break_forces[i], 1.0)) * 100
                }
            
            # Overload assessment
            working_overload = line.current_force / line.working_load_limit
            ultimate_overload = line.current_force / line.ultimate_strength
            analysis['max_overload_ratio'] = max(analysis['max_overload_ratio'], ultimate_overload)
            
            # Risk assessment
            if ultimate_overload > 0.9:  # >90% of ultimate strength
                impact.overload_risk = "CRITICAL"
                analysis['critical_lines'].append(i)
            elif ultimate_overload > 0.75:  # >75% of ultimate strength
                impact.overload_risk = "HIGH"
            elif working_overload > 1.2:   # >120% of working load
                impact.overload_risk = "MEDIUM"
            else:
                impact.overload_risk = "LOW"
            
            # Track intact vs overloaded
            if line.status == LineStatus.OVERLOADED:
                analysis['overloaded_lines'].append(i)
            else:
                analysis['intact_lines'].append(i)
            
            total_current_force += line.current_force
        
        # Calculate total force redistribution
        if total_pre_break_force > 0:
            analysis['total_force_increase'] = (total_current_force - total_pre_break_force) / 1e6  # MN
        
        return analysis
    
    def get_impact_summary(self) -> str:
        """
        Get a formatted summary of impact load analysis.
        
        Returns:
            Formatted string with impact analysis summary
        """
        if not self.impact_analysis_active:
            return "No line breaks detected - impact analysis inactive"
        
        intact_lines = [i for i, line in enumerate(self.lines) if line.status != LineStatus.BROKEN]
        overloaded_lines = [i for i, line in enumerate(self.lines) if line.status == LineStatus.OVERLOADED]
        
        summary = f"IMPACT LOAD ANALYSIS\n"
        summary += f"Intact lines: {len(intact_lines)}/{self.num_lines}\n"
        summary += f"Overloaded lines: {len(overloaded_lines)}\n"
        
        for i in intact_lines:
            line = self.lines[i]
            impact = line.impact_data
            summary += f"Line {i}: {impact.current_force/1e6:.2f} MN (SF: {impact.safety_factor:.2f}, Risk: {impact.overload_risk})\n"
        
        return summary

    def compute_line_force(
        self, 
        line: MooringLine, 
        attachment_position: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """
        Compute force from a single mooring line.
        
        Args:
            line: Mooring line object
            attachment_position: Current attachment position in global frame [m]
            
        Returns:
            force_vector: Force vector [Fx, Fy] in global frame [N]
            force_magnitude: Force magnitude [N]
        """
        # Check if line is broken
        if line.status == LineStatus.BROKEN:
            return np.zeros(2), 0.0
        
        # Vector from attachment to anchor
        line_vector = line.anchor_position - attachment_position
        line_length = np.linalg.norm(line_vector)
        
        # Update line state
        line.current_length = line_length
        
        # Check for slack condition
        if line_length <= line.unstretched_length:
            line.status = LineStatus.SLACK
            return np.zeros(2), 0.0
        else:
            if line.status == LineStatus.SLACK:
                line.status = LineStatus.INTACT
        
        # Compute extension and force magnitude
        extension = line_length - line.unstretched_length
        force_magnitude = (line.stiffness / line.unstretched_length) * extension
        
        # Apply load factor for interactive control
        force_magnitude *= line.load_factor
        
        # Force direction (unit vector along line)
        if line_length > 1e-12:  # Avoid division by zero
            force_direction = line_vector / line_length
        else:
            force_direction = np.zeros(2)
        
        # Force vector (tension pulls toward anchor)
        force_vector = force_magnitude * force_direction
        
        # Update line state
        line.current_force = force_magnitude
        line.force_vector = force_vector.copy()
        
        return force_vector, force_magnitude
    
    def compute_total_forces(
        self, 
        attachment_positions: np.ndarray,
        current_time: float = 0.0
    ) -> Tuple[np.ndarray, float, List[float]]:
        """
        Compute total forces and moment from all mooring lines with impact analysis.
        
        Args:
            attachment_positions: Attachment positions in global frame [Nx2]
            current_time: Current simulation time [s]
            
        Returns:
            total_force: Total force vector [Fx, Fy] [N]
            total_moment: Total moment about platform center [N⋅m]
            line_forces: List of individual line force magnitudes [N]
        """
        total_force = np.zeros(2)
        total_moment = 0.0
        line_forces = []
        
        # First compute all line forces before updating status
        for i, line in enumerate(self.lines):
            if i < len(attachment_positions):
                attachment_pos = attachment_positions[i]
                
                # Compute line force
                force_vector, force_magnitude = self.compute_line_force(line, attachment_pos)
                
                # Add to totals
                total_force += force_vector
                
                # Compute moment arm (vector from platform center to attachment point)
                # Assuming platform center is at (0,0) in body frame
                moment_arm = attachment_pos  # This should be relative to platform center
                
                # Moment contribution (cross product in 2D)
                moment_contribution = np.cross(moment_arm, force_vector)
                total_moment += moment_contribution
                
                line_forces.append(force_magnitude)
            else:
                line_forces.append(0.0)
        
        # Update line status after computing forces
        self.update_line_status(current_time)
        
        return total_force, total_moment, line_forces
    
    def get_line_extensions(self, attachment_positions: np.ndarray) -> List[float]:
        """
        Get current extensions of all lines.
        
        Args:
            attachment_positions: Attachment positions in global frame (Nx2 array)
            
        Returns:
            List of line extensions [m]
        """
        extensions = []
        
        for line in self.lines:
            if line.status == LineStatus.BROKEN:
                extensions.append(0.0)
            else:
                attach_pos = attachment_positions[line.attachment_id]
                line_length = np.linalg.norm(line.anchor_position - attach_pos)
                extension = max(0.0, line_length - line.unstretched_length)
                extensions.append(extension)
        
        return extensions
    
    def get_system_stiffness_matrix(
        self, 
        attachment_positions: np.ndarray
    ) -> np.ndarray:
        """
        Compute system stiffness matrix for stability analysis.
        
        Args:
            attachment_positions: Attachment positions in global frame (Nx2 array)
            
        Returns:
            3x3 stiffness matrix for surge, sway, yaw
        """
        K = np.zeros((3, 3))
        
        for line in self.lines:
            if line.status == LineStatus.BROKEN:
                continue
                
            attach_pos = attachment_positions[line.attachment_id]
            line_vector = line.anchor_position - attach_pos
            line_length = np.linalg.norm(line_vector)
            
            if line_length <= line.unstretched_length:
                continue  # Slack line contributes no stiffness
            
            # Unit vector along line
            if line_length > 1e-12:
                unit_vector = line_vector / line_length
            else:
                continue
            
            # Line stiffness
            k_line = line.stiffness / line.unstretched_length
            
            # Contribution to surge-sway stiffness
            K[0, 0] += k_line * unit_vector[0]**2  # Surge-surge
            K[0, 1] += k_line * unit_vector[0] * unit_vector[1]  # Surge-sway
            K[1, 0] += k_line * unit_vector[1] * unit_vector[0]  # Sway-surge
            K[1, 1] += k_line * unit_vector[1]**2  # Sway-sway
            
            # Contribution to yaw stiffness (simplified)
            moment_arm_length = np.linalg.norm(attach_pos)
            K[2, 2] += k_line * moment_arm_length**2
        
        return K
    
    def get_line_info(self) -> Dict:
        """
        Get comprehensive information about all lines.
        
        Returns:
            Dictionary with line information
        """
        info = {
            'num_lines': self.num_lines,
            'lines': []
        }
        
        for i, line in enumerate(self.lines):
            line_info = {
                'id': i,
                'status': line.status.value,
                'anchor_position': line.anchor_position.tolist(),
                'attachment_id': line.attachment_id,
                'current_length': line.current_length,
                'current_force': line.current_force,
                'force_vector': line.force_vector.tolist(),
                'load_factor': line.load_factor,
                'break_time': line.break_time
            }
            info['lines'].append(line_info)
        
        return info
    
    def reset_line_status(self) -> None:
        """Reset all lines to intact status."""
        for line in self.lines:
            line.status = LineStatus.INTACT
            line.current_force = 0.0
            line.force_vector = np.zeros(2)
    
    def __repr__(self) -> str:
        """String representation of mooring system."""
        intact_lines = sum(1 for line in self.lines if line.status == LineStatus.INTACT)
        return (f"MooringSystem({self.num_lines} lines, "
                f"{intact_lines} intact, "
                f"L0={self.unstretched_length} m, "
                f"EA={self.stiffness:.2e} N)") 