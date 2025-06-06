"""
Animation and visualization system for platform simulation.

This module provides classes for creating both static timelapse videos
and interactive real-time visualizations with controls.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Slider, Button
from typing import Optional, Dict, Any, Callable
from pathlib import Path

from ..core.simulator import SimulationResults


class TimelapseAnimator:
    """
    Creates timelapse animations from simulation results.
    
    Generates MP4 videos showing platform motion over time
    with customizable speed and styling.
    """
    
    def __init__(self, results: SimulationResults, config: Optional[Dict[str, Any]] = None):
        """
        Initialize animator with simulation results.
        
        Args:
            results: Simulation results object
            config: Animation configuration dictionary
        """
        self.results = results
        self.config = config or self._get_default_config()
        
        # Extract key data
        self.time = results.time
        self.x = results.x
        self.y = results.y
        self.psi = results.psi
        self.line_forces = results.line_forces
        
        # Animation state
        self.fig = None
        self.ax = None
        self.ani = None
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default animation configuration."""
        return {
            'speedup_factor': 6.0,
            'fps': 30,
            'plot_limits': [-250, 250],
            'platform_color': 'black',
            'line_colors': ['blue', 'blue', 'blue', 'blue'],
            'broken_line_color': 'red',
            'anchor_color': 'red',
            'anchor_marker': 'x',
            'anchor_size': 100,
            'force_display': {
                'show_force_bars': True,
                'color_code_lines': True,
                'force_scale_factor': 1.0e-6
            }
        }
    
    def create_video(
        self, 
        output_path: str, 
        speedup: Optional[float] = None,
        fps: Optional[int] = None,
        show_forces: bool = True
    ) -> None:
        """
        Create timelapse video animation.
        
        Args:
            output_path: Output video file path
            speedup: Animation speed multiplier (overrides config)
            fps: Frames per second (overrides config)
            show_forces: Whether to show force visualization
        """
        speedup = speedup or self.config['speedup_factor']
        fps = fps or self.config['fps']
        
        print(f"Creating timelapse animation...")
        print(f"  Output: {output_path}")
        print(f"  Speedup: {speedup}x")
        print(f"  FPS: {fps}")
        
        # Calculate frame skip based on speedup
        dt_sim = np.mean(np.diff(self.time))  # Average simulation time step
        dt_video = 1.0 / fps  # Video time step
        dt_real = dt_video * speedup  # Real time per video frame
        frame_skip = max(1, int(dt_real / dt_sim))
        
        # Create figure and subplots
        if show_forces:
            self.fig, (self.ax, self.ax_force) = plt.subplots(1, 2, figsize=(16, 8))
        else:
            self.fig, self.ax = plt.subplots(1, 1, figsize=(10, 8))
        
        self._setup_main_plot()
        
        if show_forces:
            self._setup_force_plot()
        
        # Create animation
        frames = range(0, len(self.time), frame_skip)
        
        def animate(frame_idx):
            return self._update_frame(frames[frame_idx], show_forces)
        
        def init():
            return self._init_animation(show_forces)
        
        self.ani = animation.FuncAnimation(
            self.fig, animate, frames=len(frames),
            init_func=init, blit=True, interval=1000/fps
        )
        
        # Save animation
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        writer = animation.FFMpegWriter(fps=fps, bitrate=5000)
        self.ani.save(str(output_file), writer=writer)
        
        print(f"Video saved: {output_path}")
        
        plt.close(self.fig)
    
    def _setup_main_plot(self) -> None:
        """Setup the main platform visualization plot."""
        limits = self.config['plot_limits']
        
        self.ax.set_xlim(limits[0], limits[1])
        self.ax.set_ylim(limits[0], limits[1])
        self.ax.set_aspect('equal')
        self.ax.grid(True, alpha=0.3)
        self.ax.set_xlabel('X Position [m]')
        self.ax.set_ylabel('Y Position [m]')
        self.ax.set_title('Platform Motion (Line Break Scenario)')
        
        # Initialize plot elements
        self.platform_line, = self.ax.plot([], [], 
                                          color=self.config['platform_color'], 
                                          linewidth=3, label='Platform')
        
        # Mooring lines
        self.mooring_lines = []
        for i in range(4):  # Assuming 4 lines
            line, = self.ax.plot([], [], 
                               color=self.config['line_colors'][i], 
                               linewidth=2, alpha=0.8)
            self.mooring_lines.append(line)
        
        # Anchors (static)
        anchor_positions = np.array([[60, 60], [-60, 60], [-60, -60], [60, -60]])
        self.ax.scatter(anchor_positions[:, 0], anchor_positions[:, 1], 
                       c=self.config['anchor_color'], 
                       marker=self.config['anchor_marker'],
                       s=self.config['anchor_size'], 
                       label='Anchors', zorder=10)
        
        # Trajectory trail
        self.trajectory_line, = self.ax.plot([], [], 'gray', alpha=0.5, linewidth=1)
        
        # Time text
        self.time_text = self.ax.text(0.02, 0.95, '', transform=self.ax.transAxes, 
                                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        self.ax.legend(loc='upper right')
    
    def _setup_force_plot(self) -> None:
        """Setup the force visualization plot."""
        self.ax_force.set_ylim(0, 1.0)  # Will be updated dynamically
        self.ax_force.set_xlabel('Mooring Line')
        self.ax_force.set_ylabel('Force [MN]')
        self.ax_force.set_title('Mooring Line Forces')
        self.ax_force.grid(True, alpha=0.3)
        
        # Force bars
        self.force_bars = self.ax_force.bar(range(4), [0, 0, 0, 0], 
                                           color=self.config['line_colors'],
                                           alpha=0.8)
        self.ax_force.set_xticks(range(4))
        self.ax_force.set_xticklabels(['Line 0', 'Line 1', 'Line 2', 'Line 3'])
    
    def _init_animation(self, show_forces: bool) -> list:
        """Initialize animation elements."""
        self.platform_line.set_data([], [])
        self.trajectory_line.set_data([], [])
        
        for line in self.mooring_lines:
            line.set_data([], [])
        
        self.time_text.set_text('')
        
        elements = [self.platform_line, self.trajectory_line, self.time_text] + self.mooring_lines
        
        if show_forces:
            for bar in self.force_bars:
                bar.set_height(0)
            elements.extend(self.force_bars)
        
        return elements
    
    def _update_frame(self, frame_idx: int, show_forces: bool) -> list:
        """Update animation frame."""
        # Current time and state
        t = self.time[frame_idx]
        x, y, psi = self.x[frame_idx], self.y[frame_idx], self.psi[frame_idx]
        
        # Platform corners in global frame
        corners = self._get_platform_corners(x, y, psi)
        platform_loop = np.vstack([corners, corners[0]])  # Close the loop
        self.platform_line.set_data(platform_loop[:, 0], platform_loop[:, 1])
        
        # Trajectory trail (last 50 points)
        trail_start = max(0, frame_idx - 50)
        self.trajectory_line.set_data(self.x[trail_start:frame_idx+1], 
                                     self.y[trail_start:frame_idx+1])
        
        # Mooring lines
        anchor_positions = np.array([[60, 60], [-60, 60], [-60, -60], [60, -60]])
        
        for i, line in enumerate(self.mooring_lines):
            attach_pos = corners[i]
            anchor_pos = anchor_positions[i]
            
            # Check if line is broken (force = 0)
            force = self.line_forces[frame_idx, i] if frame_idx < len(self.line_forces) else 0
            
            if force > 1.0:  # Line is active
                line.set_data([attach_pos[0], anchor_pos[0]], 
                             [attach_pos[1], anchor_pos[1]])
                # Color code by force if enabled
                if self.config['force_display']['color_code_lines']:
                    # Normalize force for color coding
                    max_force = np.max(self.line_forces) if len(self.line_forces) > 0 else 1.0
                    intensity = min(1.0, force / max_force)
                    color = plt.cm.viridis(intensity)
                    line.set_color(color)
                else:
                    line.set_color(self.config['line_colors'][i])
            else:
                # Broken line - show as dashed red or hide
                line.set_data([attach_pos[0], anchor_pos[0]], 
                             [attach_pos[1], anchor_pos[1]])
                line.set_color(self.config['broken_line_color'])
                line.set_linestyle('--')
                line.set_alpha(0.3)
        
        # Update time text
        self.time_text.set_text(f'Time: {t:.1f} s')
        
        elements = [self.platform_line, self.trajectory_line, self.time_text] + self.mooring_lines
        
        # Update force bars if enabled
        if show_forces and frame_idx < len(self.line_forces):
            forces_MN = self.line_forces[frame_idx] * self.config['force_display']['force_scale_factor']
            max_force = np.max(forces_MN) * 1.1 if np.max(forces_MN) > 0 else 1.0
            
            self.ax_force.set_ylim(0, max_force)
            
            for i, bar in enumerate(self.force_bars):
                bar.set_height(forces_MN[i])
                # Color broken lines differently
                if forces_MN[i] < 0.01:  # Essentially zero
                    bar.set_color('red')
                    bar.set_alpha(0.3)
                else:
                    bar.set_color(self.config['line_colors'][i])
                    bar.set_alpha(0.8)
            
            elements.extend(self.force_bars)
        
        return elements
    
    def _get_platform_corners(self, x: float, y: float, psi: float) -> np.ndarray:
        """Get platform corner positions in global frame."""
        # Platform dimensions (assuming square platform)
        half_length = 60.0  # Half platform length
        
        # Corner positions in body frame
        corners_body = np.array([
            [half_length, half_length],    # Front-right
            [-half_length, half_length],   # Front-left
            [-half_length, -half_length],  # Rear-left
            [half_length, -half_length]    # Rear-right
        ])
        
        # Rotation matrix
        cos_psi, sin_psi = np.cos(psi), np.sin(psi)
        R = np.array([[cos_psi, -sin_psi], [sin_psi, cos_psi]])
        
        # Transform to global frame
        corners_global = (R @ corners_body.T).T + np.array([x, y])
        
        return corners_global


class InteractiveAnimator:
    """
    Interactive real-time animation with controls.
    
    Provides speed control sliders, load adjustment sliders,
    and real-time force visualization.
    """
    
    def __init__(self, results: SimulationResults, config: Optional[Dict[str, Any]] = None):
        """
        Initialize interactive animator.
        
        Args:
            results: Simulation results object
            config: Animation configuration dictionary
        """
        self.results = results
        self.config = config or self._get_default_config()
        
        # Animation state
        self.current_frame = 0
        self.is_playing = False
        self.speed_multiplier = 1.0
        self.load_factors = [1.0, 1.0, 1.0, 1.0]  # For interactive load control
        
        # UI elements
        self.fig = None
        self.ax_main = None
        self.ax_force = None
        self.sliders = {}
        self.buttons = {}
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default interactive animation configuration."""
        return {
            'speedup_factor': 1.0,
            'fps': 30,
            'plot_limits': [-250, 250],
            'platform_color': 'black',
            'line_colors': ['blue', 'green', 'orange', 'purple'],
            'broken_line_color': 'red',
            'anchor_color': 'red',
            'anchor_marker': 'x',
            'anchor_size': 100,
            'speed_control': {
                'enabled': True,
                'min_speed': 0.1,
                'max_speed': 10.0,
                'default_speed': 1.0
            },
            'load_control': {
                'enabled': True,
                'min_load_factor': 0.0,
                'max_load_factor': 2.0,
                'default_load_factor': 1.0
            },
            'force_display': {
                'show_force_bars': True,
                'show_force_values': True,
                'color_code_lines': True,
                'force_scale_factor': 1.0e-6
            }
        }
    
    def create_interactive_display(self, **kwargs) -> None:
        """
        Create interactive display with controls.
        
        Args:
            **kwargs: Override configuration options
        """
        # Update config with any provided options
        for key, value in kwargs.items():
            if key in self.config:
                if isinstance(self.config[key], dict) and isinstance(value, dict):
                    self.config[key].update(value)
                else:
                    self.config[key] = value
        
        print("Creating interactive visualization...")
        print("Controls available:")
        if self.config['speed_control']['enabled']:
            print("  - Speed control slider")
        if self.config['load_control']['enabled']:
            print("  - Load adjustment sliders for each line")
        print("  - Play/Pause button")
        print("  - Reset button")
        
        self._setup_interactive_figure()
        self._create_control_widgets()
        self._update_display()
        
        plt.show()
    
    def _setup_interactive_figure(self) -> None:
        """Setup the interactive figure layout."""
        # Create figure with subplots
        self.fig = plt.figure(figsize=(18, 12))
        
        # Main animation plot (top)
        self.ax_main = plt.subplot2grid((3, 4), (0, 0), colspan=4, rowspan=2)
        
        # Force display plot (bottom left)
        self.ax_force = plt.subplot2grid((3, 4), (2, 0), colspan=2)
        
        # Control panel area (bottom right)
        self.ax_controls = plt.subplot2grid((3, 4), (2, 2), colspan=2)
        self.ax_controls.set_xticks([])
        self.ax_controls.set_yticks([])
        self.ax_controls.set_title('Interactive Controls')
        
        # Setup plots
        self._setup_main_plot()
        self._setup_force_plot()
    
    def _setup_main_plot(self) -> None:
        """Setup main platform visualization."""
        # Similar to TimelapseAnimator but for interactive use
        limits = self.config['plot_limits']
        
        self.ax_main.set_xlim(limits[0], limits[1])
        self.ax_main.set_ylim(limits[0], limits[1])
        self.ax_main.set_aspect('equal')
        self.ax_main.grid(True, alpha=0.3)
        self.ax_main.set_xlabel('X Position [m]')
        self.ax_main.set_ylabel('Y Position [m]')
        self.ax_main.set_title('Interactive Platform Simulation')
        
        # Plot elements (similar setup as TimelapseAnimator)
        self.platform_line, = self.ax_main.plot([], [], 
                                               color=self.config['platform_color'], 
                                               linewidth=3, label='Platform')
        
        self.mooring_lines = []
        for i in range(4):
            line, = self.ax_main.plot([], [], 
                                    color=self.config['line_colors'][i], 
                                    linewidth=3, alpha=0.8, label=f'Line {i}')
            self.mooring_lines.append(line)
        
        # Anchors
        anchor_positions = np.array([[60, 60], [-60, 60], [-60, -60], [60, -60]])
        self.ax_main.scatter(anchor_positions[:, 0], anchor_positions[:, 1], 
                           c=self.config['anchor_color'], 
                           marker=self.config['anchor_marker'],
                           s=self.config['anchor_size'], 
                           label='Anchors', zorder=10)
        
        self.trajectory_line, = self.ax_main.plot([], [], 'gray', alpha=0.5, linewidth=1)
        
        self.time_text = self.ax_main.text(0.02, 0.95, '', transform=self.ax_main.transAxes, 
                                         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        self.ax_main.legend(loc='upper right')
    
    def _setup_force_plot(self) -> None:
        """Setup force visualization."""
        self.ax_force.set_xlabel('Mooring Line')
        self.ax_force.set_ylabel('Force [MN]')
        self.ax_force.set_title('Real-time Line Forces')
        self.ax_force.grid(True, alpha=0.3)
        
        self.force_bars = self.ax_force.bar(range(4), [0, 0, 0, 0], 
                                          color=self.config['line_colors'],
                                          alpha=0.8)
        self.ax_force.set_xticks(range(4))
        self.ax_force.set_xticklabels(['Line 0', 'Line 1', 'Line 2', 'Line 3'])
    
    def _create_control_widgets(self) -> None:
        """Create interactive control widgets."""
        # Speed control slider
        if self.config['speed_control']['enabled']:
            ax_speed = plt.axes([0.55, 0.20, 0.35, 0.03])
            self.sliders['speed'] = Slider(
                ax_speed, 'Speed', 
                self.config['speed_control']['min_speed'],
                self.config['speed_control']['max_speed'],
                valinit=self.config['speed_control']['default_speed'],
                valfmt='%.1fx'
            )
            self.sliders['speed'].on_changed(self._on_speed_change)
        
        # Load control sliders
        if self.config['load_control']['enabled']:
            for i in range(4):
                ax_load = plt.axes([0.55, 0.12 - i*0.025, 0.35, 0.02])
                self.sliders[f'load_{i}'] = Slider(
                    ax_load, f'Line {i}', 
                    self.config['load_control']['min_load_factor'],
                    self.config['load_control']['max_load_factor'],
                    valinit=self.config['load_control']['default_load_factor'],
                    valfmt='%.1f'
                )
                self.sliders[f'load_{i}'].on_changed(
                    lambda val, line_id=i: self._on_load_change(line_id, val)
                )
        
        # Play/Pause button
        ax_play = plt.axes([0.55, 0.05, 0.15, 0.04])
        self.buttons['play'] = Button(ax_play, 'Play')
        self.buttons['play'].on_clicked(self._toggle_play)
        
        # Reset button
        ax_reset = plt.axes([0.75, 0.05, 0.15, 0.04])
        self.buttons['reset'] = Button(ax_reset, 'Reset')
        self.buttons['reset'].on_clicked(self._reset_animation)
    
    def _on_speed_change(self, val: float) -> None:
        """Handle speed slider change."""
        self.speed_multiplier = val
    
    def _on_load_change(self, line_id: int, val: float) -> None:
        """Handle load slider change."""
        self.load_factors[line_id] = val
        self._update_display()  # Immediate visual feedback
    
    def _toggle_play(self, event) -> None:
        """Toggle play/pause state."""
        self.is_playing = not self.is_playing
        self.buttons['play'].label.set_text('Pause' if self.is_playing else 'Play')
        
        if self.is_playing:
            self._start_animation()
    
    def _reset_animation(self, event) -> None:
        """Reset animation to beginning."""
        self.current_frame = 0
        self.is_playing = False
        self.buttons['play'].label.set_text('Play')
        self._update_display()
    
    def _start_animation(self) -> None:
        """Start the animation loop."""
        if hasattr(self, '_timer'):
            self._timer.stop()
        
        interval = 1000 / (self.config['fps'] * self.speed_multiplier)
        self._timer = self.fig.canvas.new_timer(interval=interval)
        self._timer.add_callback(self._animation_step)
        self._timer.start()
    
    def _animation_step(self) -> None:
        """Single animation step."""
        if not self.is_playing:
            return
        
        self.current_frame += 1
        if self.current_frame >= len(self.results.time):
            if self.config.get('loop_animation', True):
                self.current_frame = 0  # Loop back to start
            else:
                self.is_playing = False
                self.buttons['play'].label.set_text('Play')
                return
        
        self._update_display()
    
    def _update_display(self) -> None:
        """Update the display with current frame and settings."""
        if self.current_frame >= len(self.results.time):
            return
        
        # Get current state
        t = self.results.time[self.current_frame]
        x = self.results.x[self.current_frame]
        y = self.results.y[self.current_frame]
        psi = self.results.psi[self.current_frame]
        
        # Update platform visualization (similar to TimelapseAnimator)
        corners = self._get_platform_corners(x, y, psi)
        platform_loop = np.vstack([corners, corners[0]])
        self.platform_line.set_data(platform_loop[:, 0], platform_loop[:, 1])
        
        # Update trajectory
        trail_start = max(0, self.current_frame - 50)
        self.trajectory_line.set_data(
            self.results.x[trail_start:self.current_frame+1], 
            self.results.y[trail_start:self.current_frame+1]
        )
        
        # Update mooring lines with load factors
        anchor_positions = np.array([[60, 60], [-60, 60], [-60, -60], [60, -60]])
        
        for i, line in enumerate(self.mooring_lines):
            attach_pos = corners[i]
            anchor_pos = anchor_positions[i]
            
            # Apply load factor
            if self.current_frame < len(self.results.line_forces):
                original_force = self.results.line_forces[self.current_frame, i]
                modified_force = original_force * self.load_factors[i]
            else:
                modified_force = 0
            
            if modified_force > 1.0:  # Line is active
                line.set_data([attach_pos[0], anchor_pos[0]], 
                             [attach_pos[1], anchor_pos[1]])
                line.set_color(self.config['line_colors'][i])
                line.set_alpha(0.8)
                line.set_linewidth(3)
            else:
                # Broken or slack line
                line.set_data([attach_pos[0], anchor_pos[0]], 
                             [attach_pos[1], anchor_pos[1]])
                line.set_color(self.config['broken_line_color'])
                line.set_alpha(0.3)
                line.set_linewidth(1)
                line.set_linestyle('--')
        
        # Update time display
        self.time_text.set_text(f'Time: {t:.1f} s')
        
        # Update force bars
        if self.current_frame < len(self.results.line_forces):
            forces = self.results.line_forces[self.current_frame] * self.load_factors
            forces_MN = forces * self.config['force_display']['force_scale_factor']
            max_force = np.max(forces_MN) * 1.1 if np.max(forces_MN) > 0 else 1.0
            
            self.ax_force.set_ylim(0, max_force)
            
            for i, bar in enumerate(self.force_bars):
                bar.set_height(forces_MN[i])
                if forces_MN[i] < 0.01:
                    bar.set_color('red')
                    bar.set_alpha(0.3)
                else:
                    bar.set_color(self.config['line_colors'][i])
                    bar.set_alpha(0.8)
        
        # Redraw
        self.fig.canvas.draw_idle()
    
    def _get_platform_corners(self, x: float, y: float, psi: float) -> np.ndarray:
        """Get platform corner positions (same as TimelapseAnimator)."""
        half_length = 60.0
        corners_body = np.array([
            [half_length, half_length], [-half_length, half_length],
            [-half_length, -half_length], [half_length, -half_length]
        ])
        
        cos_psi, sin_psi = np.cos(psi), np.sin(psi)
        R = np.array([[cos_psi, -sin_psi], [sin_psi, cos_psi]])
        
        return (R @ corners_body.T).T + np.array([x, y])
    
    def save_video(self, output_path: str, **kwargs) -> None:
        """Save current animation settings as static video."""
        # Create a TimelapseAnimator with current settings and save
        timelapse = TimelapseAnimator(self.results, self.config)
        timelapse.create_video(output_path, **kwargs) 