import logging
from pathlib import Path
import json
from typing import Any

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent
CONFIG_PATH = BASE_DIR / "config.json"


def _read_config() -> dict:
    try:
        with open(CONFIG_PATH, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        logger.warning("config.json not found at %s, using empty config.", CONFIG_PATH)
        return {}
    except json.JSONDecodeError as e:
        logger.error("config.json is malformed: %s", e)
        return {}


def _write_config(config: dict) -> None:
    try:
        with open(CONFIG_PATH, "w") as f:
            json.dump(config, f, indent=4)
    except OSError as e:
        logger.error("Failed to write config: %s", e)


class Config:
    def __init__(self):
        self.config = _read_config()

    def get(self, key: str) -> Any:
        return self.config.get(key)

    def set(self, key: str, value: Any) -> None:
        self.config[key] = value
        _write_config(self.config)
