# Syphonix trader — container image using MetaTrader 5 under Wine
FROM gmag11/metatrader5_vnc as base

# Install Python 3.11 and pip
RUN apt-get update \
    && apt-get install -y --no-install-recommends python3.11 python3.11-venv python3-pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Avoid .pyc files and unbuffered stdout for clean container logs.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MT5_LOGIN=0 \
    MT5_PASSWORD=your_password \
    MT5_SERVER=your_server \
    MT5_PATH=/opt/mt5 \
    REDIS_URL=redis://localhost:6379/0 \
    LOGFIRE_TOKEN=

# Install Python dependencies.
COPY requirements.txt .
RUN python3.11 -m pip install --upgrade pip \
    && python3.11 -m pip install --no-cache-dir -r requirements.txt

# Copy application source.
COPY . .

# Start MetaTrader5 via the base image entrypoint, then run the bot.
CMD ["/bin/sh", "-c", "python3.11 main.py"]
