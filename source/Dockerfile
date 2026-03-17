# Minimal Linux image with Python
FROM python:3.12-slim

# Non-root user for security
RUN useradd -m -u 1000 khundech

WORKDIR /app

# Install dependencies first (layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy bot source
COPY . .

# Data directory for persistent learning storage
RUN mkdir -p /app/data && chown -R khundech:khundech /app

USER root

# -u = unbuffered stdout so logs appear immediately
# bootstrap.py checks missing deps and installs from requirements when needed.
CMD ["python", "-u", "bootstrap.py"]
