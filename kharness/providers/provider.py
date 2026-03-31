import os
import logging
from abc import ABC, abstractmethod
from ..config.config import Config

logger = logging.getLogger(__name__)


class Provider(ABC):
    def __init__(self, name: str):
        self.name = name
        self.config = Config()

    @abstractmethod
    def chat(self, messages: list[dict], **kwargs) -> str:
        """Send a list of messages and return the model's text response."""

    def set_model_dir(self, model_dir: str) -> None:
        if os.path.exists(model_dir) and os.path.isdir(model_dir):
            self.config.set(f"{self.name}.models_dir", model_dir)
        else:
            raise FileNotFoundError(f"Model directory '{model_dir}' does not exist.")

    def set_default_model(self, model_name: str) -> None:
        self.config.set(f"{self.name}.default_model", model_name)

    def get_model_dir(self) -> str:
        return self.config.get(f"{self.name}.models_dir")

    def get_default_model(self) -> str:
        return self.config.get(f"{self.name}.default_model")
