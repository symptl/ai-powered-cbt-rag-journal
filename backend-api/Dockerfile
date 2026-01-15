# Use a slim Python image
FROM python:3.11-slim

# Install system deps (if needed for some libs)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy only what we need first for caching
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY . /app

# Create a non-root user (best practice)
RUN useradd --create-home appuser
USER appuser

# Expose port (Elastic Beanstalk TCP proxy usually forwards to 8080)
ENV PORT=8080
EXPOSE 8080

# Optional: Docker healthcheck
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s \
  CMD curl -f http://localhost:${PORT}/health || exit 1

# Start Uvicorn ASGI server
CMD ["sh", "-c", "exec uvicorn main:app --host 0.0.0.0 --port ${PORT} --proxy-headers"]