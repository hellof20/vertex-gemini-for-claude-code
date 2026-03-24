import os
import logging
from pathlib import Path

import yaml
from google import genai

logger = logging.getLogger(__name__)

# Load config file
_config_path = Path(__file__).parent / "config.yaml"
with open(_config_path) as f:
    _cfg = yaml.safe_load(f)

# Env vars override config file
VERTEX_PROJECT_ID = os.environ.get("VERTEX_PROJECT_ID") or _cfg.get("vertex_project_id", "")
VERTEX_REGION = os.environ.get("VERTEX_REGION") or _cfg.get("vertex_region", "global")
# Support both VERTEX_API_KEY and GOOGLE_CLOUD_API_KEY
VERTEX_API_KEY = os.environ.get("VERTEX_API_KEY") or os.environ.get("GOOGLE_CLOUD_API_KEY") or _cfg.get("vertex_api_key", "")
SERVER_PORT = int(os.environ.get("SERVER_PORT") or _cfg.get("server_port", 8765))

MODEL_MAP: dict[str, str] = _cfg.get("model_map", {})
DEFAULT_GEMINI_MODEL: str = _cfg.get("default_gemini_model", "gemini-3-flash-preview")

# Client configuration
client_kwargs = {
    "vertexai": True,
    "project": VERTEX_PROJECT_ID,
    "location": VERTEX_REGION,
}

if VERTEX_API_KEY:
    client_kwargs["api_key"] = VERTEX_API_KEY
    logger.info("Using API Key for authentication")
else:
    logger.info("Using Application Default Credentials (ADC) for authentication")

# google-genai SDK client (handles auth, connection pooling, retries)
gemini_client = genai.Client(**client_kwargs)
