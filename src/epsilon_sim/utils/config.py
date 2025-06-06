"""
Configuration management utilities.

This module provides functions for loading and validating
configuration files for the simulation.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List
import os


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        Configuration dictionary
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is invalid YAML
    """
    config_file = Path(config_path)
    
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Invalid YAML in configuration file: {e}")
    
    return config


def get_default_config_path() -> str:
    """
    Get path to default configuration file.
    
    Returns:
        Path to default config file
    """
    # Get package directory
    package_dir = Path(__file__).parent.parent.parent.parent
    default_config = package_dir / "config" / "default.yaml"
    
    return str(default_config)


def load_default_config() -> Dict[str, Any]:
    """
    Load default configuration.
    
    Returns:
        Default configuration dictionary
    """
    default_path = get_default_config_path()
    
    if Path(default_path).exists():
        return load_config(default_path)
    else:
        # Fallback to hardcoded defaults if file doesn't exist
        return _get_hardcoded_defaults()


def _get_hardcoded_defaults() -> Dict[str, Any]:
    """Get hardcoded default configuration as fallback."""
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
                [60.0, 60.0], [-60.0, 60.0],
                [-60.0, -60.0], [60.0, -60.0]
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
        },
        'animation': {
            'speedup_factor': 6.0,
            'fps': 30,
            'resolution': [1920, 1080],
            'loop_animation': True,
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
            'plot_limits': [-250, 250],
            'platform_color': "black",
            'line_colors': ["blue", "blue", "blue", "blue"],
            'broken_line_color': "red",
            'anchor_color': "red",
            'anchor_marker': "x",
            'anchor_size': 100,
            'force_display': {
                'show_force_bars': True,
                'show_force_values': True,
                'color_code_lines': True,
                'force_scale_factor': 1.0e-6
            }
        },
        'output': {
            'results_dir': "output",
            'video_filename': "simulation.mp4",
            'data_filename': "results.csv",
            'save_trajectory': True,
            'save_forces': True,
            'save_energy': True,
            'log_interval': 0.1
        }
    }


def validate_config(config: Dict[str, Any]) -> List[str]:
    """
    Validate configuration parameters.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        List of validation error messages
    """
    errors = []
    
    # Validate platform section
    if 'platform' in config:
        platform = config['platform']
        
        if 'mass' in platform:
            try:
                mass = float(platform['mass'])
                if mass <= 0:
                    errors.append("Platform mass must be positive")
            except (ValueError, TypeError):
                errors.append("Platform mass must be a valid number")
            
        if 'length' in platform:
            try:
                length = float(platform['length'])
                if length <= 0:
                    errors.append("Platform length must be positive")
            except (ValueError, TypeError):
                errors.append("Platform length must be a valid number")
            
        if 'width' in platform:
            try:
                width = float(platform['width'])
                if width <= 0:
                    errors.append("Platform width must be positive")
            except (ValueError, TypeError):
                errors.append("Platform width must be a valid number")
            
        if 'inertia_z' in platform:
            try:
                inertia_z = float(platform['inertia_z'])
                if inertia_z <= 0:
                    errors.append("Platform inertia_z must be positive")
            except (ValueError, TypeError):
                errors.append("Platform inertia_z must be a valid number")
    
    # Validate mooring section
    if 'mooring' in config:
        mooring = config['mooring']
        
        if 'num_lines' in mooring:
            try:
                num_lines = int(mooring['num_lines'])
                if num_lines < 3:
                    errors.append("At least 3 mooring lines required for stability")
            except (ValueError, TypeError):
                errors.append("num_lines must be a valid integer")
            
        if 'length' in mooring:
            try:
                length = float(mooring['length'])
                if length <= 0:
                    errors.append("Mooring line length must be positive")
            except (ValueError, TypeError):
                errors.append("Mooring line length must be a valid number")
            
        if 'stiffness' in mooring:
            try:
                stiffness = float(mooring['stiffness'])
                if stiffness <= 0:
                    errors.append("Mooring line stiffness must be positive")
            except (ValueError, TypeError):
                errors.append("Mooring line stiffness must be a valid number")
        
        if 'anchor_positions' in mooring:
            anchors = mooring['anchor_positions']
            num_lines = mooring.get('num_lines', 4)
            if len(anchors) != num_lines:
                errors.append("Number of anchor positions must match num_lines")
    
    # Validate simulation section
    if 'simulation' in config:
        sim = config['simulation']
        
        if 'duration' in sim:
            try:
                duration = float(sim['duration'])
                if duration <= 0:
                    errors.append("Simulation duration must be positive")
            except (ValueError, TypeError):
                errors.append("Simulation duration must be a valid number")
            
        if 'max_timestep' in sim:
            try:
                max_timestep = float(sim['max_timestep'])
                if max_timestep <= 0:
                    errors.append("Maximum timestep must be positive")
            except (ValueError, TypeError):
                errors.append("Maximum timestep must be a valid number")
            
        if 'tolerance' in sim:
            try:
                tolerance = float(sim['tolerance'])
                if tolerance <= 0:
                    errors.append("Integration tolerance must be positive")
            except (ValueError, TypeError):
                errors.append("Integration tolerance must be a valid number")
    
    return errors


def save_config(config: Dict[str, Any], output_path: str) -> None:
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary
        output_path: Output file path
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)


def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge two configuration dictionaries.
    
    Args:
        base_config: Base configuration
        override_config: Override configuration
        
    Returns:
        Merged configuration
    """
    merged = base_config.copy()
    
    for key, value in override_config.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            # Recursively merge nested dictionaries
            merged[key] = merge_configs(merged[key], value)
        else:
            # Override value
            merged[key] = value
    
    return merged


def get_config_template() -> str:
    """
    Get a YAML template for configuration.
    
    Returns:
        YAML configuration template as string
    """
    template = """# Epsilon-Sim Configuration Template

platform:
  # Physical properties of the semi-ballasted platform
  mass: 1.25e7                 # kg, total mass
  length: 120.0                # m, platform length
  width: 120.0                 # m, platform width
  inertia_z: 8.0e9            # kg⋅m², moment of inertia about vertical axis
  
  # Platform geometry - attachment points relative to center
  attachment_points:           # m, [x, y] coordinates in body frame
    - [60.0, 60.0]            # Corner 1 (front-right)
    - [-60.0, 60.0]           # Corner 2 (front-left)  
    - [-60.0, -60.0]          # Corner 3 (rear-left)
    - [60.0, -60.0]           # Corner 4 (rear-right)

mooring:
  # Mooring line configuration
  num_lines: 4
  length: 400.0                # m, unstretched line length
  stiffness: 1.2e9            # N, axial stiffness (EA)
  
  # Anchor positions in global coordinate system
  anchor_positions:           # m, [x, y] coordinates
    - [60.0, 60.0]            # Anchor 1
    - [-60.0, 60.0]           # Anchor 2
    - [-60.0, -60.0]          # Anchor 3
    - [60.0, -60.0]           # Anchor 4

damping:
  # Hydrodynamic damping coefficients
  linear_coeff: 3.5e6         # N⋅s/m, linear damping in surge/sway
  angular_coeff: 1.75e8       # N⋅m⋅s/rad, angular damping in yaw
  
simulation:
  # Integration parameters
  duration: 120.0             # s, total simulation time
  max_timestep: 0.25          # s, maximum integration timestep
  tolerance: 1.0e-6           # relative tolerance for integration
  
  # Initial conditions
  initial_position: [0.0, 0.0, 0.0]    # [x, y, psi] in [m, m, rad]
  initial_velocity: [0.0, 0.0, 0.0]    # [dx, dy, dpsi] in [m/s, m/s, rad/s]

animation:
  # Animation and visualization settings
  speedup_factor: 6.0         # Animation speed multiplier
  fps: 30                     # Frames per second
  resolution: [1920, 1080]    # [width, height] in pixels
  loop_animation: true        # Enable continuous looping
  
  # Interactive controls
  speed_control:
    enabled: true
    min_speed: 0.1            # Minimum playback speed multiplier
    max_speed: 10.0           # Maximum playback speed multiplier
    default_speed: 1.0        # Default playback speed
  
  load_control:
    enabled: true
    min_load_factor: 0.0      # Minimum load factor (0% of original)
    max_load_factor: 2.0      # Maximum load factor (200% of original)
    default_load_factor: 1.0  # Default load factor (100% of original)
  
  # Plot limits and styling
  plot_limits: [-250, 250]    # m, symmetric plot limits
  platform_color: "black"
  line_colors: ["blue", "blue", "blue", "blue"]
  broken_line_color: "red"
  anchor_color: "red"
  anchor_marker: "x"
  anchor_size: 100
  
  # Force visualization
  force_display:
    show_force_bars: true     # Show real-time force bar chart
    show_force_values: true   # Show numerical force readouts
    color_code_lines: true    # Color-code lines by force magnitude
    force_scale_factor: 1.0e-6 # Scale factor for force display (MN)

output:
  # Output file settings
  results_dir: "output"
  video_filename: "simulation.mp4"
  data_filename: "results.csv"
  
  # Data logging
  save_trajectory: true
  save_forces: true
  save_energy: true
  log_interval: 0.1           # s, time interval for data logging
"""
    return template 