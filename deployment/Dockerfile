# Auralis Production Dockerfile
# Python 3.12 with PyTorch and audio dependencies

FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    # Build tools
    build-essential \
    git \
    # Audio libraries
    libsndfile1 \
    libopus0 \
    # Cleanup
    && rm -rf /var/lib/apt/lists/*

# Install uv (fast Python package installer)
RUN pip install --no-cache-dir uv

# Copy project files
COPY pyproject.toml ./
COPY README.md LICENSE ./

# Install Python dependencies using uv
# For CPU-only deployment (change if using GPU)
RUN uv pip install --system --no-cache -e .

# Copy application code
COPY auralis/ ./auralis/
COPY server/ ./server/
COPY client/ ./client/

# Create logs directory
RUN mkdir -p logs

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/api/v1/metrics/health', timeout=5)"

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV AURALIS_ENV=production
ENV AURALIS_HOST=0.0.0.0
ENV AURALIS_PORT=8000

# Run server
CMD ["python", "-m", "uvicorn", "server.main:app", "--host", "0.0.0.0", "--port", "8000"]
