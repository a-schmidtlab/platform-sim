"""
Dynamic real-time simulation visualization for platform dynamics.

This module provides a real-time dynamic simulation that shows:
- Platform as a square representation
- Mooring lines as visual connections
- Line break event in real-time
- Oscillating platform response
- Real-time data visualization
- Looping simulation
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Button
from matplotlib.patches import Rectangle
import time
from typing import Optional, Dict, Any
from pathlib import Path

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))
from src.epsilon_sim.core.simulator import PlatformSimulator


class DynamicPlatformSimulator:
    """
    Real-time dynamic platform simulation with visual break event and oscillations.
    
    Shows the complete sequence:
    1. Platform in equilibrium with slight offset (pre-break)
    2. Line break event at specified time
    3. Dynamic oscillating response with turning motion
    4. Real-time data visualization
    5. Looping animation
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize dynamic simulator.
        
        Args:
            config: Configuration dictionary for simulation parameters
        """
        self.config = config or self._get_default_config()
        
        # Simulation state
        self.simulator = None
        self.results = None
        self.current_frame = 0
        self.is_playing = False
        self.is_looping = True
        
        # Timing
        self.break_time = 5.0  # Line breaks at t=5s to show before/after
        self.total_duration = 60.0  # Total simulation time
        self.animation_speed = 2.0  # Animation speed multiplier
        
        # UI elements
        self.fig = None
        self.ax_main = None
        self.ax_force = None
        self.ax_position = None
        self.ax_energy = None
        
        # Visual elements
        self.platform_square = None
        self.mooring_lines = []
        self.anchor_points = None
        self.force_bars = None
        self.position_lines = []
        self.energy_lines = []
        
        # Data storage for real-time plotting
        self.time_history = []
        self.position_history = {'x': [], 'y': [], 'psi': []}
        self.force_history = {'line0': [], 'line1': [], 'line2': [], 'line3': []}
        self.energy_history = {'kinetic': [], 'potential': [], 'total': []}
        
        # Animation timer
        self.animation_timer = None
        
        print("Dynamic Platform Simulator initialized")
        print(f"  - Line break scheduled at t = {self.break_time} s")
        print(f"  - Total simulation duration: {self.total_duration} s")
        print(f"  - Will show oscillating platform response")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for dynamic simulation."""
        return {
            'platform_size': 120.0,  # Platform size for visualization
            'platform_color': 'darkblue',
            'platform_alpha': 0.8,
            'line_colors': ['red', 'blue', 'green', 'purple'],
            'broken_line_color': 'gray',
            'broken_line_alpha': 0.3,
            'anchor_color': 'red',
            'anchor_size': 150,
            'animation_interval': 50,  # ms between frames
            'data_buffer_size': 1000,  # Number of data points to keep
            'plot_limits': 80,  # Plot limits in meters
        }
    
    def setup_simulation(self):
        """Setup the simulation with realistic initial conditions."""
        print("Setting up realistic platform simulation...")
        
        # Create simulator
        self.simulator = PlatformSimulator()
        
        # Set realistic initial conditions (small offset for pre-tension)
        # Platform starts slightly off-center to create initial line tension
        self.simulator.platform.state.x = 8.0   # 8m initial surge offset
        self.simulator.platform.state.y = 6.0   # 6m initial sway offset  
        self.simulator.platform.state.psi = 0.03 # Small initial heading (1.7°)
        self.simulator.platform.state.dx = 0.0   # Start at rest
        self.simulator.platform.state.dy = 0.0
        self.simulator.platform.state.dpsi = 0.0
        
        # Configure line break at t=5 seconds (not t=0)
        self.simulator.set_line_break(0, self.break_time)
        
        print(f"  Initial platform position: ({self.simulator.platform.state.x:.1f}, {self.simulator.platform.state.y:.1f}) m")
        print(f"  Initial heading: {np.degrees(self.simulator.platform.state.psi):.1f}°")
        print(f"  Line 0 will break at t = {self.break_time} s")
        
        # Run the full simulation
        print(f"  Computing dynamics for {self.total_duration} s...")
        self.results = self.simulator.run(duration=self.total_duration, max_step=0.05)
        
        print(f"  Simulation computed: {self.results.num_steps} time steps")
        print(f"  Maximum displacement: {np.max(np.sqrt(self.results.x**2 + self.results.y**2)):.1f} m")
        print(f"  Maximum speed: {np.max(np.sqrt(self.results.dx**2 + self.results.dy**2)):.3f} m/s")
        
        # Analyze oscillations
        break_idx = np.where(self.results.time >= self.break_time)[0]
        if len(break_idx) > 0:
            post_break_x = self.results.x[break_idx[0]:]
            post_break_y = self.results.y[break_idx[0]:]
            post_break_psi = self.results.psi[break_idx[0]:]
            
            print(f"  Post-break motion analysis:")
            print(f"    X oscillation range: {np.ptp(post_break_x):.1f} m")
            print(f"    Y oscillation range: {np.ptp(post_break_y):.1f} m")
            print(f"    Heading change: {np.degrees(np.ptp(post_break_psi)):.1f}°")
    
    def create_dynamic_visualization(self):
        """Create the dynamic real-time visualization."""
        print("Creating dynamic visualization interface...")
        
        # Create figure with subplots
        self.fig = plt.figure(figsize=(16, 10))
        self.fig.suptitle('Dynamic Platform Simulation - Real-time Mooring Line Break Analysis', 
                         fontsize=14, fontweight='bold')
        
        # Main platform view (top left - largest)
        self.ax_main = plt.subplot2grid((3, 4), (0, 0), colspan=2, rowspan=2)
        self._setup_main_visualization()
        
        # Force display (top right)
        self.ax_force = plt.subplot2grid((3, 4), (0, 2), colspan=2)
        self._setup_force_display()
        
        # Position tracking (bottom left)
        self.ax_position = plt.subplot2grid((3, 4), (2, 0), colspan=2)
        self._setup_position_tracking()
        
        # Energy display (bottom right)
        self.ax_energy = plt.subplot2grid((3, 4), (2, 2), colspan=2)
        self._setup_energy_display()
        
        # Control buttons
        self._setup_controls()
        
        plt.tight_layout()
        print("  Dynamic visualization created")
    
    def _setup_main_visualization(self):
        """Setup the main platform visualization with square and lines."""
        self.ax_main.set_xlim(-self.config['plot_limits'], self.config['plot_limits'])
        self.ax_main.set_ylim(-self.config['plot_limits'], self.config['plot_limits'])
        self.ax_main.set_aspect('equal')
        self.ax_main.grid(True, alpha=0.3)
        self.ax_main.set_xlabel('X Position [m]')
        self.ax_main.set_ylabel('Y Position [m]')
        self.ax_main.set_title('Platform Motion (Real-time)')
        
        # Create platform square (will be updated in animation)
        platform_size = 60.0  # Half-size for drawing
        self.platform_square = Rectangle(
            (-platform_size, -platform_size), 2*platform_size, 2*platform_size,
            facecolor=self.config['platform_color'], 
            alpha=self.config['platform_alpha'],
            edgecolor='black',
            linewidth=2
        )
        self.ax_main.add_patch(self.platform_square)
        
        # Create mooring lines
        self.mooring_lines = []
        for i in range(4):
            line, = self.ax_main.plot([], [], 
                                     color=self.config['line_colors'][i], 
                                     linewidth=3, alpha=0.9, 
                                     label=f'Line {i}')
            self.mooring_lines.append(line)
        
        # Anchor positions (static)
        anchors = np.array([[60, 60], [-60, 60], [-60, -60], [60, -60]])
        self.anchor_points = self.ax_main.scatter(
            anchors[:, 0], anchors[:, 1], 
            c=self.config['anchor_color'], 
            marker='x', s=self.config['anchor_size'], 
            linewidth=3, label='Anchors', zorder=10
        )
        
        # Time and status text
        self.time_text = self.ax_main.text(
            0.02, 0.98, '', transform=self.ax_main.transAxes,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
            verticalalignment='top', fontsize=10, fontweight='bold'
        )
        
        self.status_text = self.ax_main.text(
            0.02, 0.88, '', transform=self.ax_main.transAxes,
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8),
            verticalalignment='top', fontsize=9, fontweight='bold'
        )
        
        self.ax_main.legend(loc='upper right')
    
    def _setup_force_display(self):
        """Setup real-time force bar chart."""
        self.ax_force.set_title('Mooring Line Forces (Real-time)')
        self.ax_force.set_xlabel('Line Number')
        self.ax_force.set_ylabel('Force [MN]')
        self.ax_force.grid(True, alpha=0.3)
        
        # Create force bars
        line_names = ['Line 0', 'Line 1', 'Line 2', 'Line 3']
        self.force_bars = self.ax_force.bar(
            range(4), [0, 0, 0, 0], 
            color=self.config['line_colors'],
            alpha=0.8
        )
        self.ax_force.set_xticks(range(4))
        self.ax_force.set_xticklabels(line_names)
        self.ax_force.set_ylim(0, 5)  # Will be updated dynamically
    
    def _setup_position_tracking(self):
        """Setup real-time position tracking."""
        self.ax_position.set_title('Platform Position (Real-time)')
        self.ax_position.set_xlabel('Time [s]')
        self.ax_position.set_ylabel('Position [m]')
        self.ax_position.grid(True, alpha=0.3)
        
        # Create position lines
        self.position_lines = []
        labels = ['Surge (X)', 'Sway (Y)', 'Heading [°/10]']
        colors = ['blue', 'red', 'green']
        
        for i, (label, color) in enumerate(zip(labels, colors)):
            line, = self.ax_position.plot([], [], color=color, linewidth=2, label=label)
            self.position_lines.append(line)
        
        self.ax_position.legend()
        self.ax_position.set_xlim(0, self.total_duration)
        self.ax_position.set_ylim(-30, 30)
    
    def _setup_energy_display(self):
        """Setup real-time energy tracking."""
        self.ax_energy.set_title('System Energy (Real-time)')
        self.ax_energy.set_xlabel('Time [s]')
        self.ax_energy.set_ylabel('Energy [MJ]')
        self.ax_energy.grid(True, alpha=0.3)
        
        # Create energy lines
        self.energy_lines = []
        labels = ['Kinetic', 'Potential', 'Total']
        colors = ['red', 'blue', 'green']
        
        for i, (label, color) in enumerate(zip(labels, colors)):
            line, = self.ax_energy.plot([], [], color=color, linewidth=2, label=label)
            self.energy_lines.append(line)
        
        self.ax_energy.legend()
        self.ax_energy.set_xlim(0, self.total_duration)
        self.ax_energy.set_ylim(0, 1)  # Will be updated dynamically
    
    def _setup_controls(self):
        """Setup control buttons."""
        # Play/Pause button
        ax_play = plt.axes([0.02, 0.02, 0.08, 0.04])
        self.btn_play = Button(ax_play, 'Play')
        self.btn_play.on_clicked(self._toggle_play)
        
        # Reset button
        ax_reset = plt.axes([0.12, 0.02, 0.08, 0.04])
        self.btn_reset = Button(ax_reset, 'Reset')
        self.btn_reset.on_clicked(self._reset_simulation)
        
        # Speed control
        ax_speed_up = plt.axes([0.22, 0.02, 0.08, 0.04])
        self.btn_speed_up = Button(ax_speed_up, 'Speed +')
        self.btn_speed_up.on_clicked(self._increase_speed)
        
        ax_speed_down = plt.axes([0.32, 0.02, 0.08, 0.04])
        self.btn_speed_down = Button(ax_speed_down, 'Speed -')
        self.btn_speed_down.on_clicked(self._decrease_speed)
    
    def _update_frame(self):
        """Update single animation frame."""
        if not self.is_playing or self.results is None:
            return
        
        # Get current time and state
        current_time = self.results.time[self.current_frame]
        x = self.results.x[self.current_frame]
        y = self.results.y[self.current_frame]
        psi = self.results.psi[self.current_frame]
        
        # Update platform square position and rotation
        self._update_platform_square(x, y, psi)
        
        # Update mooring lines
        self._update_mooring_lines(x, y, psi, current_time)
        
        # Update time and status displays
        self._update_status_displays(current_time)
        
        # Update real-time data plots
        self._update_data_plots(current_time)
        
        # Advance frame
        self.current_frame += 1
        if self.current_frame >= len(self.results.time):
            if self.is_looping:
                self.current_frame = 0
                self._clear_data_history()
                print("  Animation loop restarted")
            else:
                self.is_playing = False
                self.btn_play.label.set_text('Play')
    
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
        
        # Update rectangle (matplotlib Rectangle doesn't support rotation easily,
        # so we'll use a polygon-like approach by updating the patch)
        self.platform_square.set_xy(corners_global[0])  # Bottom-left corner
        
        # For proper rotation, we'll update using a transform
        import matplotlib.transforms as transforms
        t = transforms.Affine2D().rotate_around(x, y, psi) + self.ax_main.transData
        self.platform_square.set_transform(t)
        
        # Simpler approach: update the rectangle center and angle
        self.platform_square.set_x(x - platform_size)
        self.platform_square.set_y(y - platform_size)
    
    def _update_mooring_lines(self, x: float, y: float, psi: float, current_time: float):
        """Update mooring line visualizations."""
        # Platform attachment points in global frame
        platform_size = 60.0
        attachments_body = np.array([
            [platform_size, platform_size],    # Corner 0
            [-platform_size, platform_size],   # Corner 1
            [-platform_size, -platform_size],  # Corner 2
            [platform_size, -platform_size]    # Corner 3
        ])
        
        cos_psi, sin_psi = np.cos(psi), np.sin(psi)
        R = np.array([[cos_psi, -sin_psi], [sin_psi, cos_psi]])
        attachments_global = (R @ attachments_body.T).T + np.array([x, y])
        
        # Anchor positions
        anchors = np.array([[60, 60], [-60, 60], [-60, -60], [60, -60]])
        
        # Update each mooring line
        for i, line in enumerate(self.mooring_lines):
            attach_pos = attachments_global[i]
            anchor_pos = anchors[i]
            
            # Check if line is broken
            line_broken = (i == 0 and current_time >= self.break_time)
            
            if line_broken:
                # Show broken line
                line.set_data([attach_pos[0], anchor_pos[0]], 
                             [attach_pos[1], anchor_pos[1]])
                line.set_color(self.config['broken_line_color'])
                line.set_alpha(self.config['broken_line_alpha'])
                line.set_linestyle('--')
                line.set_linewidth(2)
            else:
                # Show intact line
                line.set_data([attach_pos[0], anchor_pos[0]], 
                             [attach_pos[1], anchor_pos[1]])
                line.set_color(self.config['line_colors'][i])
                line.set_alpha(0.9)
                line.set_linestyle('-')
                line.set_linewidth(3)
    
    def _update_status_displays(self, current_time: float):
        """Update time and status text displays."""
        # Time display
        self.time_text.set_text(f'Time: {current_time:.1f} s\nSpeed: {self.animation_speed:.1f}x')
        
        # Status display
        if current_time < self.break_time:
            status = f'PRE-BREAK\nAll lines intact\nBreak in: {self.break_time - current_time:.1f} s'
            self.status_text.set_bbox(dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        elif current_time < self.break_time + 0.5:
            status = 'LINE BREAK!\nLine 0 FAILED'
            self.status_text.set_bbox(dict(boxstyle='round', facecolor='red', alpha=0.8))
        else:
            status = 'POST-BREAK\nOscillating response\nLine 0 broken'
            self.status_text.set_bbox(dict(boxstyle='round', facecolor='orange', alpha=0.8))
        
        self.status_text.set_text(status)
    
    def _update_data_plots(self, current_time: float):
        """Update real-time data visualization."""
        frame_idx = self.current_frame
        
        # Update data history
        self.time_history.append(current_time)
        self.position_history['x'].append(self.results.x[frame_idx])
        self.position_history['y'].append(self.results.y[frame_idx])
        self.position_history['psi'].append(np.degrees(self.results.psi[frame_idx]))
        
        # Force data
        forces_MN = self.results.line_forces[frame_idx] / 1e6
        for i, force in enumerate(forces_MN):
            self.force_history[f'line{i}'].append(force)
        
        # Energy data
        ke_MJ = self.results.kinetic_energy[frame_idx] / 1e6
        pe_MJ = self.results.potential_energy[frame_idx] / 1e6
        self.energy_history['kinetic'].append(ke_MJ)
        self.energy_history['potential'].append(pe_MJ)
        self.energy_history['total'].append(ke_MJ + pe_MJ)
        
        # Limit data buffer size
        buffer_size = self.config['data_buffer_size']
        if len(self.time_history) > buffer_size:
            self.time_history = self.time_history[-buffer_size:]
            for key in self.position_history:
                self.position_history[key] = self.position_history[key][-buffer_size:]
            for key in self.force_history:
                self.force_history[key] = self.force_history[key][-buffer_size:]
            for key in self.energy_history:
                self.energy_history[key] = self.energy_history[key][-buffer_size:]
        
        # Update force bars
        max_force = max(max(forces_MN), 1.0)
        self.ax_force.set_ylim(0, max_force * 1.1)
        for i, bar in enumerate(self.force_bars):
            bar.set_height(forces_MN[i])
            if i == 0 and current_time >= self.break_time:
                bar.set_color('gray')
                bar.set_alpha(0.3)
            else:
                bar.set_color(self.config['line_colors'][i])
                bar.set_alpha(0.8)
        
        # Update position tracking
        time_data = np.array(self.time_history)
        self.position_lines[0].set_data(time_data, self.position_history['x'])
        self.position_lines[1].set_data(time_data, self.position_history['y'])
        self.position_lines[2].set_data(time_data, np.array(self.position_history['psi'])/10)  # Scale for visibility
        
        # Update energy tracking
        max_energy = max(max(self.energy_history['total']), 0.1)
        self.ax_energy.set_ylim(0, max_energy * 1.1)
        self.energy_lines[0].set_data(time_data, self.energy_history['kinetic'])
        self.energy_lines[1].set_data(time_data, self.energy_history['potential'])
        self.energy_lines[2].set_data(time_data, self.energy_history['total'])
        
        # Mark break time on plots
        if current_time >= self.break_time and len(time_data) > 1:
            # Add vertical line at break time (only once)
            if not hasattr(self, '_break_line_added'):
                self.ax_position.axvline(x=self.break_time, color='red', linestyle='--', alpha=0.7, label='Line Break')
                self.ax_energy.axvline(x=self.break_time, color='red', linestyle='--', alpha=0.7, label='Line Break')
                self._break_line_added = True
    
    def _clear_data_history(self):
        """Clear data history for loop restart."""
        self.time_history = []
        for key in self.position_history:
            self.position_history[key] = []
        for key in self.force_history:
            self.force_history[key] = []
        for key in self.energy_history:
            self.energy_history[key] = []
        
        # Remove break line markers
        if hasattr(self, '_break_line_added'):
            delattr(self, '_break_line_added')
    
    def _toggle_play(self, event):
        """Toggle play/pause."""
        self.is_playing = not self.is_playing
        self.btn_play.label.set_text('Pause' if self.is_playing else 'Play')
        
        if self.is_playing:
            self._start_animation()
        else:
            self._stop_animation()
    
    def _reset_simulation(self, event):
        """Reset simulation to beginning."""
        self.current_frame = 0
        self.is_playing = False
        self.btn_play.label.set_text('Play')
        self._clear_data_history()
        self._stop_animation()
        print("  Simulation reset to beginning")
    
    def _increase_speed(self, event):
        """Increase animation speed."""
        self.animation_speed = min(self.animation_speed * 1.5, 10.0)
        print(f"  Animation speed: {self.animation_speed:.1f}x")
        if self.is_playing:
            self._restart_animation()
    
    def _decrease_speed(self, event):
        """Decrease animation speed."""
        self.animation_speed = max(self.animation_speed / 1.5, 0.1)
        print(f"  Animation speed: {self.animation_speed:.1f}x")
        if self.is_playing:
            self._restart_animation()
    
    def _start_animation(self):
        """Start the animation timer."""
        interval = int(self.config['animation_interval'] / self.animation_speed)
        self.animation_timer = self.fig.canvas.new_timer(interval=interval)
        self.animation_timer.add_callback(self._update_frame)
        self.animation_timer.start()
    
    def _stop_animation(self):
        """Stop the animation timer."""
        if self.animation_timer:
            self.animation_timer.stop()
    
    def _restart_animation(self):
        """Restart animation with new speed."""
        self._stop_animation()
        if self.is_playing:
            self._start_animation()
    
    def run(self):
        """Run the complete dynamic simulation."""
        print("\n" + "="*60)
        print("DYNAMIC PLATFORM SIMULATION - Real-time Visualization")
        print("="*60)
        
        try:
            # Setup simulation
            self.setup_simulation()
            
            # Create visualization
            self.create_dynamic_visualization()
            
            print("\nDynamic simulation ready!")
            print("Controls:")
            print("  - Play/Pause: Start/stop animation")
            print("  - Reset: Return to beginning")
            print("  - Speed +/-: Adjust playback speed")
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