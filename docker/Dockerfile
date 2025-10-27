# HeCoReS-DR Dockerfile
# Drug Repositioning with Hetnet Connectivity Search

FROM python:3.10-slim

# Set metadata
LABEL maintainer="Donato Aveiro <dgoncalo.ba@gmail.com>"
LABEL description="Hetnet Connectivity Search for Drug Repositioning"
LABEL version="1.0.0"

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    git \
    curl \
    wget \
    libxml2-dev \
    libxslt-dev \
    libssl-dev \
    libffi-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt

# Download NLTK data
RUN python -m nltk.downloader punkt stopwords wordnet

# Copy project files
COPY setup.py .
COPY README.md .
COPY LICENSE .

# Copy source code
COPY src/ ./src/
COPY data/ ./data/
COPY experiments/ ./experiments/

# Install the package
RUN pip install -e .

# Create directories for data and results
RUN mkdir -p /app/data/raw \
             /app/data/processed \
             /app/experiments/results 

# Set permissions
RUN chmod -R 755 /app

# Expose port for potential web services
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import sys; sys.exit(0)"

# Default command
CMD ["python", "-m", "src.models.training_testing"]
