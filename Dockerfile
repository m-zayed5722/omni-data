FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ ./app/
COPY backend/ ./backend/
COPY frontend/ ./frontend/
COPY .env.example .env

# Create uploads directory
RUN mkdir -p uploads

# Expose ports (8000 for API, 8501 for Streamlit)
EXPOSE 8000 8501

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command runs the API server
CMD ["uvicorn", "app.main_viz:app", "--host", "0.0.0.0", "--port", "8000"]