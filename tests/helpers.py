from dal.config import default_config
from dal.env import ScenarioData


def small_cfg(window: int = 32, max_steps: int = 30) -> dict:
    cfg = default_config()
    cfg["filter"]["window"] = window
    cfg["filter"]["max_steps"] = max_steps
    return cfg


def synthetic_data(**kw) -> ScenarioData:
    return ScenarioData.synthetic(**kw)
