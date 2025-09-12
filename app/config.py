import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Ollama Configuration
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral")

# API Keys
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
SERP_API_KEY = os.getenv("SERP_API_KEY")

# API URLs
OPENWEATHER_BASE_URL = "https://api.openweathermap.org/data/2.5/weather"