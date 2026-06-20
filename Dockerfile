# Syphonix trader — container image for Northflank deployment.
FROM python:3.11-slim

# Avoid .pyc files and unbuffered stdout for clean container logs.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install dependencies first to leverage Docker layer caching.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source.
COPY . .

# Run the scheduler entry point.
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD sh -c "[ -f /proc/1/cmdline ] && grep -q 'main.py' /proc/1/cmdline || exit 1"

CMD ["python", "main.py"]
