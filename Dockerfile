# Stage 1: Builder stage
FROM python:3.10-slim AS builder

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# Install essential system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create and set working directory
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Create non-root user for Python installations
RUN useradd -m -U -u 1001 appuser
USER appuser
ENV PATH=/home/appuser/.local/bin:$PATH

# Install Python dependencies to user directory
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --user -r requirements.txt && \
    pip install --no-cache-dir --user python-multipart  # Explicitly install

# Stage 2: Final lightweight image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app/backend:$PYTHONPATH

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user and directories
RUN useradd -m -U -u 1001 appuser && \
    mkdir -p /app/data/images /app/data/annotations && \
    chown -R appuser:appuser /app

# Copy installed packages from builder
COPY --from=builder --chown=appuser:appuser /home/appuser/.local /home/appuser/.local

# Set environment path for user
ENV PATH=/home/appuser/.local/bin:$PATH

# Set working directory
WORKDIR /app

# Copy application code
COPY --chown=appuser:appuser . .

# Create __init__.py files for Python packages
RUN touch backend/__init__.py && \
    touch frontend/__init__.py && \
    touch scripts/__init__.py

# Create empty .env file if not exists
RUN touch .env

# Clean up Python cache
RUN find /home/appuser/.local -type d -name '__pycache__' -exec rm -rf {} + && \
    find /home/appuser/.local -name '*.pyc' -delete

# Switch to non-root user
USER appuser

# Expose ports
EXPOSE 8501 8000

# Default command
CMD ["sh", "-c", "echo 'Please specify service to run: backend or frontend'"]