import os
import numpy as np
import yaml
from dotenv import load_dotenv

load_dotenv()

def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    cfg["database"]["url"] = os.path.expandvars(cfg["database"]["url"])
    return cfg

config = load_config()

def get_source_points() -> np.ndarray:
    return np.array(config["speed"]["source_points"], dtype=np.float32)

def get_target_points() -> np.ndarray:
    w = config["speed"]["target_width_m"]
    h = config["speed"]["target_length_m"]
    return np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
