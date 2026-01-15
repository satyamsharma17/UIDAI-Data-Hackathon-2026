"""
Configuration Manager
Loads and validates configuration from YAML file
"""

import yaml
import os
from pathlib import Path
from typing import Dict, Any
import logging


class ConfigManager:
    """
    Manages application configuration from YAML file
    """
    
    def __init__(self, config_path: str = 'config.yaml'):
        """
        Initialize configuration manager
        
        Args:
            config_path: Path to YAML configuration file
        """
        self.config_path = config_path
        self.config = self._load_config()
        self._validate_config()
        self._setup_logging()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML configuration: {e}")
    
    def _validate_config(self):
        """Validate required configuration sections exist"""
        required_sections = ['project', 'data', 'processing', 'analysis']
        
        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"Missing required configuration section: {section}")
    
    def _setup_logging(self):
        """Setup logging based on configuration"""
        log_config = self.config.get('logging', {})
        
        # Create logs directory if it doesn't exist
        log_file = log_config.get('file', 'logs/analysis.log')
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        # Configure logging
        logging.basicConfig(
            level=getattr(logging, log_config.get('level', 'INFO')),
            format=log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s'),
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Configuration loaded from {self.config_path}")
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation
        
        Args:
            key_path: Dot-separated path to config value (e.g., 'data.input.enrolment')
            default: Default value if key not found
        
        Returns:
            Configuration value or default
        """
        keys = key_path.split('.')
        value = self.config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value
    
    def get_data_paths(self) -> Dict[str, str]:
        """Get all data paths"""
        return {
            'input': self.config['data']['input'],
            'output': self.config['data']['output']
        }
    
    def get_processing_params(self) -> Dict[str, Any]:
        """Get processing parameters"""
        return self.config.get('processing', {})
    
    def get_analysis_params(self) -> Dict[str, Any]:
        """Get analysis parameters"""
        return self.config.get('analysis', {})
    
    def get_visualization_settings(self) -> Dict[str, Any]:
        """Get visualization settings"""
        return self.config.get('visualization', {})
    
    def get_metric_config(self, metric_name: str) -> Dict[str, Any]:
        """Get configuration for specific metric"""
        return self.config.get('metrics', {}).get(metric_name, {})
    
    def get_roi_params(self) -> Dict[str, Any]:
        """Get ROI calculation parameters"""
        return self.config.get('roi', {})
    
    def get_quality_thresholds(self) -> Dict[str, float]:
        """Get data quality thresholds"""
        return self.config.get('quality', {})
    
    def update_config(self, key_path: str, value: Any):
        """
        Update configuration value
        
        Args:
            key_path: Dot-separated path to config value
            value: New value to set
        """
        keys = key_path.split('.')
        config_section = self.config
        
        # Navigate to the correct section
        for key in keys[:-1]:
            if key not in config_section:
                config_section[key] = {}
            config_section = config_section[key]
        
        # Set the value
        config_section[keys[-1]] = value
        self.logger.info(f"Updated config: {key_path} = {value}")
    
    def save_config(self, output_path: str = None):
        """
        Save current configuration to YAML file
        
        Args:
            output_path: Path to save config (defaults to original path)
        """
        output_path = output_path or self.config_path
        
        with open(output_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False, sort_keys=False)
        
        self.logger.info(f"Configuration saved to {output_path}")


# Global config instance
_config_instance = None


def get_config(config_path: str = 'config.yaml') -> ConfigManager:
    """
    Get global configuration instance (singleton pattern)
    
    Args:
        config_path: Path to configuration file
    
    Returns:
        ConfigManager instance
    """
    global _config_instance
    
    if _config_instance is None:
        _config_instance = ConfigManager(config_path)
    
    return _config_instance


if __name__ == "__main__":
    # Test configuration loading
    config = ConfigManager('config.yaml')
    
    print("Configuration loaded successfully!")
    print("\nProject Info:")
    print(f"  Name: {config.get('project.name')}")
    print(f"  Version: {config.get('project.version')}")
    
    print("\nData Paths:")
    paths = config.get_data_paths()
    print(f"  Input: {paths['input']}")
    print(f"  Output: {paths['output']}")
    
    print("\nProcessing Params:")
    params = config.get_processing_params()
    print(f"  Sample Rate: {params.get('sample_rate')}")
    print(f"  Chunk Size: {params.get('chunk_size')}")
    
    print("\nROI Parameters:")
    roi = config.get_roi_params()
    print(f"  Implementation Cost: ₹{roi['costs']['implementation']:,}")
    print(f"  Annual Maintenance: ₹{roi['costs']['annual_maintenance']:,}")
