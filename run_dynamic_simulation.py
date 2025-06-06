#!/usr/bin/env python3
"""
Enhanced dynamic real-time simulation with realistic oscillating motion.

This script provides a real-time dynamic simulation that shows:
- Platform as a square representation with realistic oscillating motion
- Mooring lines as visual connections
- Line break event in real-time
- Proper oscillating platform response with swinging motion
- Real-time data visualization
- Looping simulation with enhanced physics
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Button, Slider
from matplotlib.patches import Polygon
import time
from typing import Optional, Dict, Any

# Add the project root to Python path
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

from src.epsilon_sim.core.simulator import PlatformSimulator


class DynamicPlatformSimulator:
    """
    Real-time dynamic platform simulation with enhanced oscillating motion.
    """
    
    def __init__(self):
        """Initialize enhanced dynamic simulator."""
        # Simulation state
        self.simulator = None
        self.results = None
        self.current_frame = 0
        self.is_playing = False
        
        # Enhanced timing for dramatic motion
        self.break_time = 8.0  # Line breaks at t=8s to show equilibrium first
        self.total_duration = 120.0  # Longer simulation to see oscillations develop
        self.frame_skip = 1  # Real-time animation
        self.animation_speed = 1.0  # Speed multiplier for control
        
        # UI elements
        self.fig = None
        self.ax_main = None
        self.ax_force = None
        self.ax_position = None
        
        # Visual elements
        self.platform_square = None
        self.mooring_lines = []
        self.anchor_points = None
        self.force_lines = []  # Changed from force_bars to force_lines
        self.position_lines = []
        
        # Data storage for real-time plotting
        self.time_history = []
        self.position_history = {'x': [], 'y': [], 'psi': []}
        self.force_history = {'line0': [], 'line1': [], 'line2': [], 'line3': []}
        
        print("Enhanced Dynamic Platform Simulator initialized")
        print(f"  - Line break scheduled at t = {self.break_time} s")
        print(f"  - Total simulation duration: {self.total_duration} s")
        print("  - Enhanced physics with realistic oscillating motion")
    
    def setup_simulation(self):
        """Setup the enhanced simulation with realistic physics and dramatic initial conditions."""
        print("Setting up enhanced platform simulation with realistic oscillating motion...")
        
        # Create simulator with enhanced configuration
        self.simulator = PlatformSimulator()
        
        # Set dramatic initial conditions for visible oscillating motion
        # Platform starts significantly off-center to create strong restoring forces
        print("Setting dramatic initial conditions:")
        
        # Moderate initial displacement for proper oscillations (not too extreme)
        initial_surge = 30.0    # 30m surge displacement (moderate for oscillations)
        initial_sway = 20.0     # 20m sway displacement  
        initial_heading = 0.2   # 11° initial heading 
        
        # Moderate initial velocities for oscillating motion
        initial_surge_vel = -1.0  # Moderate motion back toward center
        initial_sway_vel = 0.8    # Moderate cross motion
        initial_yaw_vel = 0.05    # Moderate initial yaw rate
        
        self.simulator.platform.state.x = initial_surge
        self.simulator.platform.state.y = initial_sway  
        self.simulator.platform.state.psi = initial_heading
        self.simulator.platform.state.dx = initial_surge_vel
        self.simulator.platform.state.dy = initial_sway_vel
        self.simulator.platform.state.dpsi = initial_yaw_vel
        
        # Configure moderate wave excitation for sustainable oscillations
        print("Setting up wave excitation for enhanced oscillations:")
        wave_amplitude = 5.0e6     # 5.0 MN wave force amplitude (moderate for oscillations)
        wave_frequency = 0.12      # 0.12 rad/s (period ≈ 52 seconds)
        wave_phase = np.pi/4       # Phase offset
        
        self.simulator.platform.set_wave_excitation(
            amplitude=wave_amplitude,
            frequency=wave_frequency, 
            phase=wave_phase
        )
        
        # Configure line break at enhanced timing
        self.simulator.set_line_break(0, self.break_time)
        
        print(f"  Initial platform position: ({self.simulator.platform.state.x:.1f}, {self.simulator.platform.state.y:.1f}) m")
        print(f"  Initial heading: {np.degrees(self.simulator.platform.state.psi):.1f}°")
        print(f"  Initial velocities: ({self.simulator.platform.state.dx:.2f}, {self.simulator.platform.state.dy:.2f}) m/s")
        print(f"  Wave excitation: {wave_amplitude/1e6:.1f} MN at {wave_frequency:.2f} rad/s")
        print(f"  Line 0 will break at t = {self.break_time} s")
        
        # Run the full enhanced simulation with fine time stepping
        print(f"  Computing enhanced dynamics for {self.total_duration} s...")
        self.results = self.simulator.run(duration=self.total_duration, max_step=0.01, rtol=1e-8)
        
        # Analyze results for oscillation characteristics
        max_displacement = np.max(np.sqrt(self.results.x**2 + self.results.y**2))
        max_speed = np.max(np.sqrt(self.results.dx**2 + self.results.dy**2))
        max_heading = np.max(np.abs(self.results.psi))
        max_yaw_rate = np.max(np.abs(self.results.dpsi))
        
        print(f"  Simulation completed: {self.results.num_steps} time steps")
        print(f"  Maximum displacement: {max_displacement:.1f} m")
        print(f"  Maximum speed: {max_speed:.3f} m/s")
        print(f"  Maximum heading: {np.degrees(max_heading):.1f}°")
        print(f"  Maximum yaw rate: {np.degrees(max_yaw_rate):.3f} °/s")
        
        # Check for oscillating motion
        surge_oscillations = self._count_oscillations(self.results.x, self.results.time)
        sway_oscillations = self._count_oscillations(self.results.y, self.results.time)
        yaw_oscillations = self._count_oscillations(self.results.psi, self.results.time)
        
        print(f"  Detected oscillations - Surge: {surge_oscillations}, Sway: {sway_oscillations}, Yaw: {yaw_oscillations}")
        
        # Setup display parameters 
        self.frame_skip = max(1, len(self.results.time) // 2000)  # Limit to ~2000 frames
        print(f"  Frame skip: {self.frame_skip} (total frames: {len(self.results.time) // self.frame_skip})")
        
        # Position of anchors/tugboats - set early for force calculations
        self.anchor_positions = np.array([
            [280, 280],   # NE
            [-280, 280],  # NW  
            [-280, -280], # SW
            [280, -280]   # SE
        ])
        
        print("  Dynamic simulation setup complete!")
        
    def _count_oscillations(self, signal: np.ndarray, time: np.ndarray, min_period: float = 5.0) -> int:
        """Count number of oscillations in a signal (zero crossings)."""
        # Remove trend
        signal_detrended = signal - np.mean(signal)
        
        # Find zero crossings
        zero_crossings = 0
        for i in range(1, len(signal_detrended)):
            if signal_detrended[i-1] * signal_detrended[i] < 0:
                zero_crossings += 1
        
        # Each full oscillation has 2 zero crossings
        return zero_crossings // 2
    
    def _calculate_current_forces(self, frame_idx: int, current_time: float):
        """Calculate mooring line forces at current platform state in MN."""
        try:
            # Get current platform state
            x = self.results.x[frame_idx]
            y = self.results.y[frame_idx] 
            psi = self.results.psi[frame_idx]
            
            # Platform attachment points in body frame (corners)
            platform_size = 60.0
            attachments_body = np.array([
                [platform_size, platform_size],    # Line 0 - NE corner
                [-platform_size, platform_size],   # Line 1 - NW corner  
                [-platform_size, -platform_size],  # Line 2 - SW corner
                [platform_size, -platform_size]    # Line 3 - SE corner
            ])
            
            # Transform to global frame
            cos_psi, sin_psi = np.cos(psi), np.sin(psi)
            R = np.array([[cos_psi, -sin_psi], [sin_psi, cos_psi]])
            attachments_global = (R @ attachments_body.T).T + np.array([x, y])
            
            # Mooring line parameters
            L0 = 300.0  # Unstretched length [m]
            EA = 1.2e9  # Axial stiffness [N]
            
            forces_MN = []
            
            # Calculate force for each line
            for i in range(4):
                if i == 0 and current_time >= self.break_time:
                    # Line 0 is broken after break time
                    forces_MN.append(0.0)
                else:
                    # Get attachment and anchor positions
                    attach_pos = attachments_global[i]
                    anchor_pos = self.anchor_positions[i]
                    
                    # Calculate line vector and current length
                    line_vec = attach_pos - anchor_pos
                    current_length = np.linalg.norm(line_vec)
                    
                    # Calculate extension (strain)
                    extension = max(0.0, current_length - L0)
                    
                    # Calculate force using Hooke's law: F = (EA/L0) * extension
                    force_N = (EA / L0) * extension
                    force_MN = force_N / 1e6
                    
                    forces_MN.append(force_MN)
            
            return forces_MN
            
        except Exception as e:
            print(f"Error calculating forces: {e}")
            return [0.0, 0.0, 0.0, 0.0]
    
    def create_dynamic_visualization(self):
        """Create the enhanced dynamic real-time visualization."""
        print("Creating enhanced dynamic visualization interface...")
        
        # Create figure with improved layout
        self.fig = plt.figure(figsize=(16, 10))
        self.fig.suptitle('Floating Platform Dynamic Analysis: 120m×120m Semi-Ballasted Structure | 4-Line Diagonal Mooring System | Line Break Response at t=8.0s', 
                         fontsize=13, fontweight='bold')
        
        # Main platform view (larger, better positioned)
        self.ax_main = plt.subplot2grid((2, 3), (0, 0), colspan=2, rowspan=2)
        self._setup_main_visualization()
        
        # Force display (top right)
        self.ax_force = plt.subplot2grid((2, 3), (0, 2))
        self._setup_force_display()
        
        # Position tracking (bottom right)
        self.ax_position = plt.subplot2grid((2, 3), (1, 2))
        self._setup_position_tracking()
        
        # Control buttons
        self._setup_controls()
        
        plt.tight_layout()
        print("  Enhanced dynamic visualization created")
    
    def _setup_main_visualization(self):
        """Setup the main platform visualization with enhanced view for oscillations."""
        # Much larger plot limits to show entire scenario (tugboats at ±250m + platform drift)
        plot_limit = 350  # Large enough for tugboats + platform motion
        self.ax_main.set_xlim(-plot_limit, plot_limit)
        self.ax_main.set_ylim(-plot_limit, plot_limit)
        self.ax_main.set_aspect('equal')
        self.ax_main.grid(True, alpha=0.3)
        self.ax_main.set_xlabel('X Position [m]')
        self.ax_main.set_ylabel('Y Position [m]')
        self.ax_main.set_title('Platform Oscillating Motion (Real-time)')
        
        # Add reference circles for oscillation amplitude visualization
        for radius in [25, 50, 75, 100]:
            circle = plt.Circle((0, 0), radius, fill=False, color='gray', alpha=0.2, linestyle='--')
            self.ax_main.add_patch(circle)
        
        # Create platform square as polygon (larger for visibility)
        platform_size = 60.0  # Half-size for drawing
        square_corners = np.array([
            [-platform_size, -platform_size],
            [platform_size, -platform_size],
            [platform_size, platform_size],
            [-platform_size, platform_size]
        ])
        
        self.platform_square = Polygon(square_corners, 
                                     facecolor='darkblue', 
                                     alpha=0.8, 
                                     edgecolor='black',
                                     linewidth=2)
        self.ax_main.add_patch(self.platform_square)
        
        # Create mooring lines with enhanced visualization
        self.mooring_lines = []
        line_colors = ['red', 'blue', 'green', 'purple']
        for i in range(4):
            line, = self.ax_main.plot([], [], 
                                     color=line_colors[i], 
                                     linewidth=4, alpha=0.9, 
                                     label=f'Line {i}')
            self.mooring_lines.append(line)
        
        # Tugboat positions (300m from platform corners) - enhanced markers  
        tugboat_distance = 280  # Distance to tugboats from origin (diagonal)
        anchors = np.array([
            [tugboat_distance, tugboat_distance],       # NE tugboat (for top-right corner)
            [-tugboat_distance, tugboat_distance],      # NW tugboat (for top-left corner)
            [-tugboat_distance, -tugboat_distance],     # SW tugboat (for bottom-left corner)
            [tugboat_distance, -tugboat_distance]       # SE tugboat (for bottom-right corner)
        ])
        self.anchor_positions = anchors
        self.anchor_points = []
        tugboat_colors = ['red', 'blue', 'green', 'purple']
        tugboat_labels = ['NE Tugboat', 'NW Tugboat', 'SW Tugboat', 'SE Tugboat']
        for i in range(4):
            tugboat = self.ax_main.scatter(
                anchors[i, 0], anchors[i, 1],
                c=tugboat_colors[i], marker='s', s=300,
                linewidth=2, label=tugboat_labels[i], zorder=10
            )
            self.anchor_points.append(tugboat)
        
        # Enhanced status displays
        self.time_text = self.ax_main.text(
            0.02, 0.98, '', transform=self.ax_main.transAxes,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9),
            verticalalignment='top', fontsize=11, fontweight='bold'
        )
        
        self.status_text = self.ax_main.text(
            0.02, 0.88, '', transform=self.ax_main.transAxes,
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.9),
            verticalalignment='top', fontsize=10, fontweight='bold'
        )
        
        # Oscillation info text
        self.oscillation_text = self.ax_main.text(
            0.02, 0.78, '', transform=self.ax_main.transAxes,
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.9),
            verticalalignment='top', fontsize=9, fontweight='bold'
        )
        
        # Impact analysis text (bottom left)
        self.impact_text = self.ax_main.text(
            0.02, 0.25, '', transform=self.ax_main.transAxes,
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9),
            verticalalignment='top', fontsize=8, fontweight='bold'
        )
        
        self.ax_main.legend(loc='upper right')
    
    def _setup_force_display(self):
        """Setup real-time force plot with live force history."""
        self.ax_force.set_title('Mooring Line Forces (Live Plot)')
        self.ax_force.set_xlabel('Time [s]')
        self.ax_force.set_ylabel('Force [MN]')
        self.ax_force.grid(True, alpha=0.3)
        
        # Create force lines for live plotting
        line_colors = ['red', 'blue', 'green', 'purple']
        line_names = ['Line 0', 'Line 1', 'Line 2', 'Line 3']
        self.force_lines = []
        
        for i, (color, name) in enumerate(zip(line_colors, line_names)):
            line, = self.ax_force.plot([], [], color=color, linewidth=2, 
                                      alpha=0.9, label=name)
            self.force_lines.append(line)
        
        # Add working load and ultimate strength reference lines
        working_load = 50.0  # 50 MN working load limit (adjusted for high-force scenario)
        ultimate_strength = 100.0  # 100 MN ultimate strength (adjusted for high-force scenario)
        
        self.ax_force.axhline(y=working_load, color='orange', linestyle='--', alpha=0.7, label='Working Load (50 MN)')
        self.ax_force.axhline(y=ultimate_strength, color='red', linestyle='-', alpha=0.7, label='Ultimate Strength (100 MN)')
        self.ax_force.legend(loc='upper right', fontsize=8)
        
        self.ax_force.set_xlim(0, self.total_duration)
        self.ax_force.set_ylim(0, 15)  # Allow for overload visualization
    
    def _setup_position_tracking(self):
        """Setup real-time position tracking."""
        self.ax_position.set_title('Platform Position')
        self.ax_position.set_xlabel('Time [s]')
        self.ax_position.set_ylabel('Position [m]')
        self.ax_position.grid(True, alpha=0.3)
        
        # Create position lines
        self.position_lines = []
        labels = ['Surge (X)', 'Sway (Y)', 'Heading [°/10]']
        colors = ['red', 'blue', 'green']  # Fixed order: red for X, blue for Y, green for heading
        
        for i, (label, color) in enumerate(zip(labels, colors)):
            line, = self.ax_position.plot([], [], color=color, linewidth=2, label=label)
            self.position_lines.append(line)
        
        self.ax_position.legend()
        self.ax_position.set_xlim(0, self.total_duration)
        self.ax_position.set_ylim(-10, 80)  # Expanded to cover surge up to 80m
    
    def _setup_controls(self):
        """Setup control buttons and speed slider at bottom."""
        # Controls positioned at bottom, horizontally arranged
        bottom_margin = 0.02
        button_width = 0.08
        button_height = 0.04
        slider_width = 0.15
        slider_height = 0.03
        
        # Play/Pause button (left)
        ax_play = plt.axes([0.15, bottom_margin, button_width, button_height])
        self.btn_play = Button(ax_play, 'Play')
        self.btn_play.on_clicked(self._toggle_play)
        
        # Reset button (center-left)
        ax_reset = plt.axes([0.25, bottom_margin, button_width, button_height])
        self.btn_reset = Button(ax_reset, 'Reset')
        self.btn_reset.on_clicked(self._reset_simulation)
        
        # Speed control slider (center-right)
        ax_speed = plt.axes([0.40, bottom_margin, slider_width, slider_height])
        self.speed_slider = Slider(ax_speed, 'Speed', 0.1, 5.0, valinit=1.0, valstep=0.1)
        self.speed_slider.on_changed(self._update_speed)
        
        # Speed display text (right)
        self.speed_text = plt.figtext(0.57, bottom_margin + 0.01, 'Speed: 1.0x', fontsize=10, ha='left')
    
    def _update_frame(self, frame_num):
        """Update single animation frame."""
        if not self.is_playing or self.results is None:
            return []
        
        # Calculate actual frame considering skip
        actual_frame = self.current_frame * self.frame_skip
        if actual_frame >= len(self.results.time):
            actual_frame = len(self.results.time) - 1
        
        # Get current time and state
        current_time = self.results.time[actual_frame]
        x = self.results.x[actual_frame]
        y = self.results.y[actual_frame]
        psi = self.results.psi[actual_frame]
        
        # Update platform square position and rotation
        self._update_platform_square(x, y, psi)
        
        # Update mooring lines
        self._update_mooring_lines(x, y, psi, current_time)
        
        # Update time and status displays
        self._update_status_displays(current_time)
        
        # Update real-time data plots
        self._update_data_plots(current_time, actual_frame)
        
        # Advance frame by speed factor
        frame_advance = max(1, int(self.animation_speed))
        self.current_frame += frame_advance
        max_frames = len(self.results.time) // self.frame_skip
        if self.current_frame >= max_frames:
            self.current_frame = 0  # Loop
            self._clear_data_history()
            print("  Animation loop restarted")
        
        return []
    
    def _update_platform_square(self, x: float, y: float, psi: float):
        """Update platform square position and rotation."""
        # Platform corners in body frame
        platform_size = 60.0
        corners_body = np.array([
            [-platform_size, -platform_size],
            [platform_size, -platform_size],
            [platform_size, platform_size],
            [-platform_size, platform_size]
        ])
        
        # Rotation matrix
        cos_psi, sin_psi = np.cos(psi), np.sin(psi)
        R = np.array([[cos_psi, -sin_psi], [sin_psi, cos_psi]])
        
        # Transform corners to global frame
        corners_global = (R @ corners_body.T).T + np.array([x, y])
        
        # Update platform polygon
        self.platform_square.set_xy(corners_global)
    
    def _update_mooring_lines(self, x: float, y: float, psi: float, current_time: float):
        """Update mooring line and tugboat visualizations."""
        # Platform attachment points in global frame (corners)
        platform_size = 60.0
        attachments_body = np.array([
            [platform_size, platform_size],    # Top-right corner
            [-platform_size, platform_size],   # Top-left corner
            [-platform_size, -platform_size],  # Bottom-left corner
            [platform_size, -platform_size]    # Bottom-right corner
        ])
        
        cos_psi, sin_psi = np.cos(psi), np.sin(psi)
        R = np.array([[cos_psi, -sin_psi], [sin_psi, cos_psi]])
        attachments_global = (R @ attachments_body.T).T + np.array([x, y])
        
        # Update each mooring line and tugboat
        for i, (line, tugboat_point) in enumerate(zip(self.mooring_lines, self.anchor_points)):
            attach_pos = attachments_global[i]
            tugboat_pos = self.anchor_positions[i]
            
            # Check if line is broken
            line_broken = (i == 0 and current_time >= self.break_time)
            
            if line_broken:
                # Hide broken line and tugboat completely
                line.set_data([], [])
                tugboat_point.set_offsets(np.empty((0, 2)))  # Hide tugboat
            else:
                # Show intact line to tugboat
                line.set_data([attach_pos[0], tugboat_pos[0]], 
                             [attach_pos[1], tugboat_pos[1]])
                line_colors = ['red', 'blue', 'green', 'purple']
                line.set_color(line_colors[i])
                line.set_alpha(0.9)
                line.set_linestyle('-')
                line.set_linewidth(4)
                
                # Show tugboat
                tugboat_point.set_offsets([tugboat_pos])
    
    def _update_status_displays(self, current_time: float):
        """Update time, status, and oscillation information displays."""
        # Time display
        self.time_text.set_text(f'Time: {current_time:.1f} s\nSpeed: {self.animation_speed:.1f}x')
        
        # Status display
        if current_time < self.break_time:
            status = f'PRE-BREAK\nAll lines intact\nBreak in: {self.break_time - current_time:.1f} s'
            self.status_text.set_bbox(dict(boxstyle='round', facecolor='lightgreen', alpha=0.9))
        elif current_time < self.break_time + 1.0:
            status = 'LINE BREAK!\nLine 0 FAILED'
            self.status_text.set_bbox(dict(boxstyle='round', facecolor='red', alpha=0.9))
        else:
            status = 'POST-BREAK\nOscillating response\nLine 0 broken'
            self.status_text.set_bbox(dict(boxstyle='round', facecolor='orange', alpha=0.9))
        
        self.status_text.set_text(status)
        
        # Oscillation information (current motion characteristics)
        if hasattr(self, 'results') and self.results is not None:
            frame_idx = min(self.current_frame * self.frame_skip, len(self.results.time) - 1)
            current_x = self.results.x[frame_idx]
            current_y = self.results.y[frame_idx]
            current_heading = np.degrees(self.results.psi[frame_idx])
            current_speed = np.sqrt(self.results.dx[frame_idx]**2 + self.results.dy[frame_idx]**2)
            
            displacement = np.sqrt(current_x**2 + current_y**2)
            
            oscillation_info = (f"Displacement: {displacement:.1f} m\n"
                               f"Heading: {current_heading:.1f}°\n"
                               f"Speed: {current_speed:.2f} m/s")
            
            self.oscillation_text.set_text(oscillation_info)
            
        # Update impact analysis display
        if current_time >= self.break_time:
            impact_analysis = self.simulator.mooring_system.analyze_impact_loads(current_time)
            if impact_analysis:
                time_since_break = impact_analysis['time_since_break']
                intact_lines = len(impact_analysis['intact_lines'])
                overloaded_lines = len(impact_analysis['overloaded_lines'])
                max_overload = impact_analysis['max_overload_ratio']
                
                impact_info = (f"IMPACT ANALYSIS\n"
                             f"Time since break: {time_since_break:.1f}s\n"
                             f"Intact lines: {intact_lines}/4\n"
                             f"Overloaded: {overloaded_lines}\n"
                             f"Max overload: {max_overload:.2f}")
                
                # Color code based on risk level
                if max_overload > 0.9:
                    self.impact_text.set_bbox(dict(boxstyle='round', facecolor='red', alpha=0.9))
                elif max_overload > 0.75:
                    self.impact_text.set_bbox(dict(boxstyle='round', facecolor='orange', alpha=0.9))
                elif overloaded_lines > 0:
                    self.impact_text.set_bbox(dict(boxstyle='round', facecolor='yellow', alpha=0.9))
                else:
                    self.impact_text.set_bbox(dict(boxstyle='round', facecolor='lightgreen', alpha=0.9))
                
                self.impact_text.set_text(impact_info)
            else:
                self.impact_text.set_text('')
        else:
            self.impact_text.set_text('')
    
    def _update_data_plots(self, current_time: float, frame_idx: int):
        """Update real-time data visualization."""
        # Update data history
        self.time_history.append(current_time)
        self.position_history['x'].append(self.results.x[frame_idx])
        self.position_history['y'].append(self.results.y[frame_idx])
        self.position_history['psi'].append(np.degrees(self.results.psi[frame_idx]))
        
        # Calculate forces in real-time from current platform state
        forces_MN = self._calculate_current_forces(frame_idx, current_time)
        
        # Store force history
        for i in range(4):
            self.force_history[f'line{i}'].append(forces_MN[i])
        
        # Debug output occasionally
        if len(self.time_history) % 50 == 1:  # Every 50 frames
            print(f"t={current_time:.1f}s: Forces = {[f'{f:.2f}' for f in forces_MN]} MN")
        
        # Limit data buffer size
        buffer_size = 500
        if len(self.time_history) > buffer_size:
            self.time_history = self.time_history[-buffer_size:]
            for key in self.position_history:
                self.position_history[key] = self.position_history[key][-buffer_size:]
            for key in self.force_history:
                self.force_history[key] = self.force_history[key][-buffer_size:]
        
        # Update live force lines
        time_data = np.array(self.time_history)
        working_load = 50.0  # MN - Adjusted for this high-force scenario
        ultimate_strength = 100.0  # MN - Adjusted for this high-force scenario
        
        # Define base colors for each line
        base_colors = ['red', 'blue', 'green', 'purple']
        line_names = ['Line 0 (Broken)', 'Line 1', 'Line 2', 'Line 3']
        
        for i, line in enumerate(self.force_lines):
            # Get force data for this line
            if f'line{i}' in self.force_history and len(self.force_history[f'line{i}']) > 0:
                force_data = np.array(self.force_history[f'line{i}'])
                
                # Ensure time_data and force_data have same length
                min_length = min(len(time_data), len(force_data))
                if min_length > 0:
                    line.set_data(time_data[-min_length:], force_data[-min_length:])
                
                # Get current force for this line
                current_force = forces_MN[i] if i < len(forces_MN) else 0.0
                
                # Apply color and style based on line status
                if i == 0:
                    # Line 0 is always broken (force = 0)
                    line.set_color('gray')
                    line.set_alpha(0.6)
                    line.set_linestyle('--')
                    line.set_linewidth(2)
                    line.set_label('Line 0 (Broken)')
                elif current_force > ultimate_strength:
                    # Critical overload - use darkened base color + thick line
                    critical_colors = ['darkred', 'darkblue', 'darkgreen', 'purple']
                    line.set_color(critical_colors[i])
                    line.set_alpha(1.0)
                    line.set_linewidth(4)
                    line.set_linestyle('-')
                    line.set_label(f'Line {i} (CRITICAL: {current_force:.1f} MN)')
                elif current_force > working_load:
                    # Overloaded - use bright base color + medium thick line  
                    overload_colors = ['red', 'cyan', 'lime', 'magenta']
                    line.set_color(overload_colors[i])
                    line.set_alpha(0.9)
                    line.set_linewidth(3)
                    line.set_linestyle('-')
                    line.set_label(f'Line {i} (OVERLOAD: {current_force:.1f} MN)')
                else:
                    # Normal operation - original color
                    line.set_color(base_colors[i])
                    line.set_alpha(0.9)
                    line.set_linewidth(2)
                    line.set_linestyle('-')
                    line.set_label(f'Line {i} ({current_force:.1f} MN)')
            else:
                # No data yet, hide line
                line.set_data([], [])
                line.set_label(f'Line {i} (No Data)')
        
        # Update position tracking (reuse time_data from above)
        self.position_lines[0].set_data(time_data, self.position_history['x'])
        self.position_lines[1].set_data(time_data, self.position_history['y'])
        self.position_lines[2].set_data(time_data, np.array(self.position_history['psi'])/10)  # Scale for visibility
        
        # Auto-adjust axis limits for better visibility
        if len(time_data) > 1:
            # Update time axis
            self.ax_position.set_xlim(max(0, current_time - 60), current_time + 5)  # Show last 60s + 5s ahead
            self.ax_force.set_xlim(max(0, current_time - 60), current_time + 5)
            
            # Update force axis limits if needed
            if len(self.force_history['line0']) > 0:
                max_force = max([max(self.force_history[f'line{i}']) for i in range(4) if self.force_history[f'line{i}']])
                self.ax_force.set_ylim(0, max(15, max_force * 1.1))
        
        # Update force plot legend to show current values
        if len(time_data) > 1:
            self.ax_force.legend(loc='upper right', fontsize=8, framealpha=0.9)
        
        # Mark break time on both plots
        if current_time >= self.break_time and len(time_data) > 1:
            # Add vertical line at break time (only once)
            if not hasattr(self, '_break_line_added'):
                self.ax_position.axvline(x=self.break_time, color='red', linestyle='--', alpha=0.7, label='Line Break')
                self.ax_force.axvline(x=self.break_time, color='red', linestyle='--', alpha=0.7, label='Line 0 Break')
                self._break_line_added = True
    
    def _clear_data_history(self):
        """Clear data history for loop restart."""
        self.time_history = []
        for key in self.position_history:
            self.position_history[key] = []
        for key in self.force_history:
            self.force_history[key] = []
        
        # Remove break line markers
        if hasattr(self, '_break_line_added'):
            delattr(self, '_break_line_added')
    
    def _toggle_play(self, event):
        """Toggle play/pause."""
        self.is_playing = not self.is_playing
        self.btn_play.label.set_text('Pause' if self.is_playing else 'Play')
    
    def _reset_simulation(self, event):
        """Reset simulation to beginning."""
        self.current_frame = 0
        self.is_playing = False
        self.btn_play.label.set_text('Play')
        self._clear_data_history()
        print("  Simulation reset to beginning")
    
    def _update_speed(self, val):
        """Update animation speed from slider."""
        self.animation_speed = val
        self.speed_text.set_text(f'Speed: {val:.1f}x')
        
        # Update animation interval if animation exists
        if hasattr(self, 'animation') and self.animation is not None:
            # Base interval is 50ms, adjust by speed
            new_interval = max(10, int(50 / val))  # Minimum 10ms interval
            self.animation.event_source.interval = new_interval
    
    def run(self):
        """Run the complete dynamic simulation."""
        print("\n" + "="*80)
        print("FLOATING PLATFORM DYNAMIC ANALYSIS - Marine Engineering Simulation")
        print("120m×120m Semi-Ballasted Structure | 4-Line Diagonal Mooring | Line Break Analysis")
        print("="*80)
        
        try:
            # Setup simulation
            self.setup_simulation()
            
            # Create visualization
            self.create_dynamic_visualization()
            
            # Create animation
            max_frames = len(self.results.time) // self.frame_skip
            self.animation = animation.FuncAnimation(
                self.fig, self._update_frame, frames=max_frames,
                interval=50, blit=False, repeat=True
            )
            
            print("\nDynamic simulation ready!")
            print("Controls:")
            print("  - Play/Pause: Start/stop animation")
            print("  - Reset: Return to beginning")
            print("\nWatch for:")
            print("  1. Initial equilibrium (slight offset)")
            print(f"  2. Line break event at t = {self.break_time} s")
            print("  3. Dynamic oscillating response")
            print("  4. Platform rotation and drift")
            print("\nPress Play to start!")
            
            # Show the plot
            plt.show()
            
            return True
            
        except Exception as e:
            print(f"\nDynamic simulation failed: {e}")
            import traceback
            traceback.print_exc()
            return False


def main():
    """Run the dynamic platform simulation."""
    simulator = DynamicPlatformSimulator()
    success = simulator.run()
    
    if success:
        print("\nDynamic simulation completed successfully!")
    else:
        print("\nDynamic simulation failed!")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 