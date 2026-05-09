# Dockerfile
# ── Use official slim Python 3.11 (no digest pin — digest changes with every patch release) ──
FROM python:3.11-slim-bookworm

# Prevents .pyc files and enables real-time log output
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install OS deps needed by some Python packages (e.g. numpy, matplotlib)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first so Docker layer-caches pip install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project source
COPY . .

# Create runtime directories that are .gitignored
RUN mkdir -p data/raw data/processed models outputs logs reports

# Expose Flask port
EXPOSE 5000

# FIX: gunicorn must reference webapp.app:app (module path),
# not app:app (which would only work if CWD was webapp/).
# Also added --workers, --timeout, and --log-level for production use.
CMD ["gunicorn", \
     "--bind", "0.0.0.0:5000", \
     "--workers", "2", \
     "--timeout", "120", \
     "--log-level", "info", \
     "webapp.app:app"]
