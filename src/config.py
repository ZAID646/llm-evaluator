import os
from dotenv import load_dotenv

load_dotenv()

OPENCODE_ZEN_API_KEY = os.getenv("OPENCODE_ZEN_API_KEY", "")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://opencode.ai/zen/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-v4-flash-free")

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")

LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY", "")
LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT", "llm-evaluator")

TARGET_URL = os.getenv("TARGET_URL", "http://localhost:8000").rstrip("/")

MAX_TOKENS_PER_CALL = int(os.getenv("MAX_TOKENS_PER_CALL", "1024"))
