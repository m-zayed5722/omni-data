# Setup Script for AI Agent
Write-Host "Setting up AI Agent..." -ForegroundColor Green

# Check if Ollama is installed
try {
    $ollamaVersion = ollama --version
    Write-Host "Ollama is installed: $ollamaVersion" -ForegroundColor Green
} catch {
    Write-Host "Ollama is not installed. Please install from: https://ollama.ai" -ForegroundColor Red
    Write-Host "After installation, run: ollama pull mistral" -ForegroundColor Yellow
    exit 1
}

# Check if Ollama is running
try {
    $response = Invoke-WebRequest -Uri "http://localhost:11434/api/version" -TimeoutSec 5
    Write-Host "Ollama is running" -ForegroundColor Green
} catch {
    Write-Host "Ollama is not running. Starting Ollama..." -ForegroundColor Yellow
    Start-Process -FilePath "ollama" -ArgumentList "serve" -WindowStyle Hidden
    Start-Sleep -Seconds 5
}

# Check if mistral model is available
try {
    $models = ollama list
    if ($models -match "mistral") {
        Write-Host "Mistral model is available" -ForegroundColor Green
    } else {
        Write-Host "Pulling Mistral model..." -ForegroundColor Yellow
        ollama pull mistral
    }
} catch {
    Write-Host "Error checking models. Please run: ollama pull mistral" -ForegroundColor Red
}

Write-Host "Setup complete! You can now run the AI Agent." -ForegroundColor Green
Write-Host "Run: python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000" -ForegroundColor Cyan