# Contributing to AI Agent

Thank you for considering contributing to this AI Agent project! 

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/YOUR_USERNAME/ollama-langchain-agent.git`
3. Create a feature branch: `git checkout -b feature/amazing-feature`
4. Make your changes
5. Test your changes: `python test_tools.py` and `python test_client.py`
6. Commit your changes: `git commit -m 'Add amazing feature'`
7. Push to the branch: `git push origin feature/amazing-feature`
8. Open a Pull Request

## Development Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys

# Start Ollama (required)
ollama serve
ollama pull mistral

# Run the application
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## Adding New Tools

To add a new tool:

1. Create a new tool class in `app/tools/tools.py`
2. Inherit from `BaseTool`
3. Implement the `_run` method
4. Add proper type annotations and Pydantic models
5. Add the tool to the `get_tools()` function
6. Test your tool

Example:
```python
class NewTool(BaseTool):
    name: str = "new_tool"
    description: str = "Description of what the tool does"
    args_schema: Type[BaseModel] = NewToolInput
    
    def _run(self, input_param: str) -> str:
        # Your tool logic here
        return "Tool result"
```

## Code Style

- Use Python type hints
- Follow PEP 8 style guidelines
- Add docstrings to functions and classes
- Use meaningful variable names
- Keep functions focused and small

## Testing

Before submitting a PR:

```bash
# Test tools directly
python test_tools.py

# Test full API
python test_client.py

# Check health endpoint
curl http://localhost:8000/health
```

## Questions?

Feel free to open an issue for any questions or discussions!