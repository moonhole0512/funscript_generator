import json
import os
import threading

class ConfigManager:
    """Algorithm hyperparameter manager."""
    
    _instance = None
    _config = {}
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(ConfigManager, cls).__new__(cls)
                cls._instance._load_config()
        return cls._instance

    def _load_config(self):
        config_path = self._get_config_path()
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

    def _get_config_path(self):
        return os.path.join(os.path.dirname(__file__), 'config.json')

    def get(self, section, key, default):
        """Retrieve a specific parameter value."""
        with self._lock:
            return self._config.get(section, {}).get(key, default)

    def set(self, section, key, value):
        """Update a specific parameter and persist to disk."""
        with self._lock:
            if section not in self._config:
                self._config[section] = {}
            self._config[section][key] = value
            
            # Persist to disk
            config_path = self._get_config_path()
            try:
                with open(config_path, 'w') as f:
                    json.dump(self._config, f, indent=2)
            except Exception as e:
                print(f"[ERROR] Failed to save config.json: {e}")

    def reload(self):
        """Reload configuration from disk."""
        with self._lock:
            self._load_config()

# Singleton instance for easy access
config = ConfigManager()
