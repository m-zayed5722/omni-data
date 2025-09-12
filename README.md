# 🤖 GenAI Data Visualization Dashboard

An intelligent data visualization platform that combines the power of AI agents with interactive dashboards for seamless data exploration and analysis.

## 🌟 Features

### 🔄 Dual-Mode Interface
- **🗣️ Natural Language Mode**: Type queries like "Create a scatter plot of age vs salary"
- **🎛️ Manual Mode**: Point-and-click interface with advanced customization

### 📊 Advanced Visualizations
- **Histogram** - Distribution analysis with customizable bins
- **Scatter Plot** - Correlation analysis with color and size encoding
- **Box Plot** - Statistical summaries with grouping options
- **Correlation Heatmap** - Relationship matrices for numeric data
- **Bar Chart** - Categorical analysis with multiple aggregations
- **Pie Chart** - Proportional analysis with top-N filtering

### 🧠 AI-Powered Analysis
- **LangChain Integration** - Sophisticated AI agent with tool calling
- **Ollama Support** - Local LLM processing with Mistral model
- **Smart Query Parsing** - Understands natural language patterns
- **Intelligent Fallbacks** - Seamless mode switching for complex queries

### ✨ Enhanced User Experience
- **Real-time Processing** - Instant visualization generation
- **Chart Downloads** - Export visualizations as PNG files
- **Interactive Controls** - Dynamic parameter adjustment
- **Professional UI** - Modern design with visual indicators

## 🏗️ Architecture

```
GenAI Dashboard/
├── 🖥️ Frontend (Streamlit)     # Interactive web interface
├── 🔗 API Layer (FastAPI)      # RESTful API endpoints  
├── 🤖 AI Agent (LangChain)     # Natural language processing
├── 📊 Visualization Engine     # Chart generation tools
├── 🧠 LLM Backend (Ollama)     # Local language model
└── 🐳 Docker Support          # Containerized deployment
```

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- [Ollama](https://ollama.ai) installed locally
- Git

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/m-zayed5722/genai-dashboard.git
   cd genai-dashboard
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up Ollama**
   ```bash
   # Install and start Ollama service
   ollama serve
   
   # Pull the Mistral model
   ollama pull mistral
   ```

4. **Start the application**
   ```bash
   # Start API server (Terminal 1)
   python start_server.py
   
   # Start Streamlit dashboard (Terminal 2) 
   python -m streamlit run frontend/streamlit_app.py --server.port 8501
   ```

5. **Open the dashboard**
   - Dashboard: http://localhost:8501
   - API Documentation: http://localhost:8002/docs

## 💡 Usage Examples

### Natural Language Queries
```
"Show me a histogram of sales"
"Create a scatter plot of price vs rating colored by category"  
"Plot correlation heatmap"
"Box plot of salary by department"
"Bar chart showing average revenue by region"
```

### Manual Mode
1. Upload your CSV file
2. Select visualization type from dropdown
3. Choose columns and parameters
4. Customize colors, grouping, and aggregations
5. Download your visualization

## 📁 Project Structure

```
genai-dashboard/
├── 📱 app/
│   ├── main_viz.py              # FastAPI application
│   ├── agent.py                 # LangChain AI agent
│   ├── models.py                # Pydantic data models
│   └── tools/
│       ├── visualization_tools.py # LangChain visualization tools
│       └── tools.py             # Additional utility tools
├── 🖥️ frontend/
│   └── streamlit_app.py         # Streamlit dashboard
├── ⚙️ backend/
│   └── visualizations.py       # Core visualization engine
├── 📊 uploads/                  # Dataset storage
├── 🐳 docker-compose.yml       # Container orchestration
├── 📋 requirements.txt         # Python dependencies
└── 🚀 start_server.py          # Application launcher
```

## 🛠️ Technology Stack

### Core Technologies
- **FastAPI** - High-performance API framework
- **Streamlit** - Interactive web applications
- **LangChain** - AI agent orchestration
- **Ollama** - Local LLM deployment

### Visualization & Data
- **Plotly** - Interactive visualizations
- **Pandas** - Data manipulation
- **NumPy** - Numerical computing
- **Seaborn** - Statistical plotting
- **Matplotlib** - Base plotting library

### AI & ML
- **Mistral** - Open-source language model
- **Pydantic** - Data validation
- **Python-multipart** - File upload handling

## 📈 Advanced Features

### Smart Query Processing
- **Pattern Recognition**: Automatically detects visualization intent
- **Column Matching**: Case-insensitive column name resolution
- **Query Interpretation**: Visual feedback on parsed queries
- **Error Recovery**: Graceful handling of ambiguous inputs

### Professional Visualizations
- **Interactive Charts**: Zoom, pan, and hover capabilities
- **Customizable Themes**: Professional color schemes
- **High-Quality Exports**: PNG downloads at publication quality
- **Responsive Design**: Works on desktop and mobile

### Data Analysis
- **Automatic Type Detection**: Smart column classification
- **Statistical Summaries**: Built-in data profiling
- **Correlation Analysis**: Relationship discovery
- **Missing Data Handling**: Robust data cleaning

## 🐳 Docker Deployment

```bash
# Build and run with Docker Compose
docker-compose up --build

# Access the application
# Dashboard: http://localhost:8501
# API: http://localhost:8002
```

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Ollama** - For local LLM deployment
- **LangChain** - For AI agent orchestration  
- **Streamlit** - For rapid dashboard development
- **Plotly** - For interactive visualizations

## 📞 Contact

**Mohamed Zayed**
- GitHub: [@m-zayed5722](https://github.com/m-zayed5722)
- Email: mzayed5722@gmail.com

## 🔗 Links

- [Repository](https://github.com/m-zayed5722/genai-dashboard)
- [Issues](https://github.com/m-zayed5722/genai-dashboard/issues)
- [Documentation](https://github.com/m-zayed5722/genai-dashboard/wiki)

---

**Built with ❤️ for the data science community**

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