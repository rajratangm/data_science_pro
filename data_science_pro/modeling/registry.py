import os
import joblib

class Registry:
    def __init__(self, registry_dir='model_registry'):
        self.registry_dir = registry_dir
        os.makedirs(self.registry_dir, exist_ok=True)

    def _get_model_path(self, name, version):
        return os.path.join(self.registry_dir, f"{name}_v{version}.pkl")

    def save_model(self, model, name, version):
        """
        Save model with name and version.
        """
        path = self._get_model_path(name, version)
        joblib.dump(model, path)
        return path

    def load_model(self, name, version):
        """
        Load model by name and version.
        """
        path = self._get_model_path(name, version)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model {name} v{version} not found.")
        return joblib.load(path)

    def get_latest_version(self, name):
        """
        Get the latest version number for a model name.
        """
        files = [f for f in os.listdir(self.registry_dir) if f.startswith(name + '_v') and f.endswith('.pkl')]
        versions = [int(f.split('_v')[1].split('.pkl')[0]) for f in files]
        return max(versions) if versions else None
