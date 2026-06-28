# Single Python 3.12 image shared by the Flower server and the bank clients.
# Compose selects the role per service via `command` (python -m server / -m client).
FROM python:3.12-slim

WORKDIR /app

# Install the pinned, CPU-only stack first so the layer caches across rebuilds.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code. Each bank still reads ONLY its own /data volume at runtime —
# nothing here grants cross-node data access.
COPY common ./common
COPY server ./server
COPY client ./client

# Default role; docker-compose overrides this per service.
CMD ["python", "-m", "server"]
