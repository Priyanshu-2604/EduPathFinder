import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# API Keys
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# Default LLM Configuration
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini")
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-2.5-flash")

# Data directories
DATA_DIR = "data"
PDF_DIR = os.path.join(DATA_DIR, "pdfs")
JSON_DIR = os.path.join(DATA_DIR, "extracted")
CSV_DIR = os.path.join(DATA_DIR, "csv")

# Create directories if they don't exist
os.makedirs(PDF_DIR, exist_ok=True)
os.makedirs(JSON_DIR, exist_ok=True)
os.makedirs(CSV_DIR, exist_ok=True)

# Logging configuration
LOGGING_LEVEL = "INFO"
LOG_FILE = "iit_extractor.log"