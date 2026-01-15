# STAGE 1: Builder
# We use the same slim image for building to ensure compatibility
FROM python:3.12-slim as builder

WORKDIR /app

# Install system build dependencies
# (Even if not strictly needed now, this allows compiling libraries like numpy/pandas if added later)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Create a virtual environment to isolate dependencies
RUN python -m venv /opt/venv
# Add the venv to PATH so pip installs into it
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# STAGE 2: Runtime
# This is the final image that will actually run
FROM python:3.12-slim

WORKDIR /app

# Copy the virtual environment from the builder stage
COPY --from=builder /opt/venv /opt/venv

# Add the venv to PATH so python/uvicorn are found automatically
ENV PATH="/opt/venv/bin:$PATH"

# Copy application code
COPY config.py .
COPY app.py .

# Expose the port Elastic Beanstalk expects (default is 8080)
EXPOSE 8080

# Run the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]