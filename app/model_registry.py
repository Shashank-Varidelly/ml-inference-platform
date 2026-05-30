"""
Handles loading both model versions and A/B traffic routing.
50/50 split by default — tweak AB_V2_RATIO to change traffic.
"""

import joblib
import numpy as np
import os
import random
from typing import Tuple

AB_V2_RATIO = 0.5  # 50% traffic goes to v2

class ModelRegistry:
    def __init__(self):
        self.models = {}
        self._load_models()

    def _load_models(self):
        model_dir = os.environ.get("MODEL_DIR", "model")
        for version in ["v1", "v2"]:
            path = os.path.join(model_dir, f"model_{version}.pkl")
            if os.path.exists(path):
                self.models[version] = joblib.load(path)
                print(f"Loaded model {version} from {path}")
            else:
                print(f"WARNING: Model {version} not found at {path}. Run model/train.py first.")

    def predict(self, features: np.ndarray, version: str = "auto") -> Tuple[float, str]:
        """
        Returns (fraud_probability, model_version_used).
        If version='auto', routes via A/B split.
        """
        if version == "auto":
            version = "v2" if random.random() < AB_V2_RATIO else "v1"

        if version not in self.models:
            # Fallback to whichever model is available
            available = list(self.models.keys())
            if not available:
                raise RuntimeError("No models loaded. Run model/train.py first.")
            version = available[0]

        model = self.models[version]
        prob = model.predict_proba(features)[0][1]
        return float(prob), version

    def loaded_versions(self) -> dict:
        return {v: "loaded" for v in self.models}


# Singleton — shared across requests
registry = ModelRegistry()
