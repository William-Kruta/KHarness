

from agentharness.providers.llamacpp import LlamaCpp
from agentharness.providers.ollama import Ollama

MODELS_DIR = "/mnt/machine_learning/AI_APPS/LLM_MODELS"
MODEL_NAME = "Nemotron-Cascade-2-30B-A3B-Q4_K_M.gguf"
llamacpp = LlamaCpp()
ollama = Ollama()

llamacpp.set_model_dir(MODELS_DIR)
llamacpp.set_default_model(MODEL_NAME)
ollama.set_default_model("qwen3.5:2b")
print(llamacpp.get_model_dir())
print(llamacpp.get_default_model())

