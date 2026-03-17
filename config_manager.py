import json
import os

class ConfigManager:
    """Algorithm hyperparameter manager."""
    
    _instance = None
    _config = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ConfigManager, cls).__new__(cls)
            cls._instance._load_config()
        return cls._instance

    def _load_config(self):
        config_path = os.path.join(os.path.dirname(__file__), 'config.json')
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    self._config = json.load(f)
            except Exception as e:
                print(f"[ERROR] Failed to load config.json: {e}")
                self._config = {}
        else:
            print("[WARNING] config.json not found. Using default internal values.")
            self._config = {}

    def get(self, section, key, default):
        """Retrieve a specific parameter value."""
        return self._config.get(section, {}).get(key, default)

# Singleton instance for easy access
config = ConfigManager()
