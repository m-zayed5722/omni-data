# AI Agent with LangChain + FastAPI + Ollama

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-00a393.svg)](https://fastapi.tiangolo.com/)
[![LangChain](https://img.shields.io/badge/LangChain-0.1+-1C3C3C.svg)](https://langchain.com/)
[![Ollama](https://img.shields.io/badge/Ollama-Latest-ff6b35.svg)](https://ollama.ai/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A powerful AI Agent that integrates a local LLM (via Ollama) with external tools including web search, calculator, and weather APIs using LangChain and FastAPI.

## 📋 Table of Contents

- [Features](#-features)
- [Demo](#-demo)
- [Prerequisites](#-prerequisites)
- [Quick Start](#-quick-start)
- [API Endpoints](#-api-endpoints)
- [Environment Variables](#-environment-variables)
- [Docker Deployment](#-docker-deployment)
- [Available Tools](#-available-tools)
- [Example Queries](#-example-queries)
- [Troubleshooting](#-troubleshooting)
- [Development](#-development)
- [License](#-license)

## 🎬 Demo

The AI Agent can handle complex queries by reasoning about which tools to use:

```json
POST /ask
{
  "query": "What is 25 * 4 and what's the weather in London?"
}

Response:
{
  "answer": "25 * 4 equals 100. The current weather in London is 18°C with light rain.",
  "tool_calls": ["calculator", "weather"],
  "error": null
}
```

## Features

- **Local LLM Integration**: Uses Ollama for running local language models (Mistral, Llama2, etc.)
- **External Tools**:
  - Calculator: Basic math operations
  - Web Search: Tavily API or SerpAPI integration
  - Weather: OpenWeatherMap API integration
- **FastAPI Backend**: RESTful API with automatic documentation
- **Docker Support**: Easy deployment with Docker and Docker Compose
- **Error Handling**: Comprehensive error handling for API failures

## Prerequisites

1. **Ollama**: Install and run Ollama locally
   ```bash
   # Install Ollama (visit https://ollama.ai for installation instructions)
   # Pull a model (e.g., Mistral)
   ollama pull mistral
   ```

2. **API Keys** (Optional but recommended):
   - Tavily API Key (for web search)
   - OpenWeatherMap API Key (for weather data)
   - SerpAPI Key (alternative for web search)

## Quick Start

### 1. Clone and Setup

```bash
git clone <your-repo>
cd AI_Project
```

### 2. Environment Configuration

```bash
# Copy environment template
cp .env.example .env

# Edit .env file with your API keys
notepad .env  # or your preferred editor
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Start Ollama (if not already running)

```bash
ollama serve
```

### 5. Run the Application

```bash
# Using Python directly
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Or using the Docker Compose (includes Ollama)
docker-compose up --build
```

### 6. Test the API

Visit `http://localhost:8000/docs` for interactive API documentation.

#### Example API Call:

```bash
curl -X POST "http://localhost:8000/ask" \
     -H "Content-Type: application/json" \
     -d '{"query": "What is the weather in Toronto and what is 5 * 12?"}'
```

Expected Response:
```json
{
  "answer": "Currently 21°C in Toronto with clear skies. Also, 5 * 12 = 60.",
  "tool_calls": [],
  "error": null
}
```

## API Endpoints

- `GET /` - Root endpoint
- `GET /health` - Health check
- `POST /ask` - Main query endpoint
- `GET /tools` - List available tools

## Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `OLLAMA_BASE_URL` | Ollama server URL | No (default: http://localhost:11434) |
| `OLLAMA_MODEL` | Ollama model name | No (default: mistral) |
| `TAVILY_API_KEY` | Tavily API key for web search | No |
| `OPENWEATHER_API_KEY` | OpenWeatherMap API key | No |
| `SERP_API_KEY` | SerpAPI key for web search | No |

## Docker Deployment

### Using Docker Compose (Recommended)

```bash
# Start all services
docker-compose up --build

# Run in background
docker-compose up -d --build
```

### Using Docker Only

```bash
# Build the image
docker build -t ai-agent .

# Run the container
docker run -p 8000:8000 \
  -e OLLAMA_BASE_URL=http://host.docker.internal:11434 \
  -e TAVILY_API_KEY=your_key_here \
  ai-agent
```

## Available Tools

### 1. Calculator Tool
- **Purpose**: Mathematical calculations
- **Example**: "What is 25 * 4 + 100?"

### 2. Web Search Tool
- **Purpose**: Current information from the web
- **APIs**: Tavily (primary) or SerpAPI (fallback)
- **Example**: "What's the latest news about AI?"

### 3. Weather Tool
- **Purpose**: Current weather information
- **API**: OpenWeatherMap
- **Example**: "What's the weather in London?"

## Example Queries

1. **Simple Math**: "Calculate 15 * 8"
2. **Weather**: "What's the weather like in New York?"
3. **Web Search**: "What are the latest developments in AI?"
4. **Combined**: "What's the weather in Tokyo and what is 100 / 4?"

## Troubleshooting

### Common Issues

1. **"Ollama server not accessible"**
   - Ensure Ollama is running: `ollama serve`
   - Check if the model is available: `ollama list`
   - Pull the model if needed: `ollama pull mistral`

2. **"No web search API key configured"**
   - Add your Tavily or SerpAPI key to `.env` file
   - Tools will still work, but web search will be unavailable

3. **"OpenWeatherMap API key not configured"**
   - Add your OpenWeatherMap API key to `.env` file
   - Weather tool will be unavailable without this key

### Logs

Check application logs for detailed error information:
```bash
# When running directly
python -m uvicorn app.main:app --log-level debug

# When using Docker Compose
docker-compose logs ai-agent
```

## Development

### Project Structure
```
AI_Project/
├── app/
│   ├── __init__.py
│   ├── main.py          # FastAPI application
│   ├── agent.py         # AI Agent implementation
│   ├── models.py        # Pydantic models
│   ├── config.py        # Configuration
│   └── tools/
│       ├── __init__.py
│       └── tools.py     # Tool implementations
├── requirements.txt     # Python dependencies
├── Dockerfile          # Docker configuration
├── docker-compose.yml  # Docker Compose setup
├── .env.example        # Environment template
└── README.md          # This file
```

### Adding New Tools

1. Create a new tool class in `app/tools/tools.py`
2. Inherit from `BaseTool`
3. Add to the `get_tools()` function
4. The agent will automatically use the new tool

## License

MIT License