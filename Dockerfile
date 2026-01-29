FROM python:3.10

# Prevent Python from writing pyc files
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies (very minimal)
RUN apt-get update && apt-get install -y \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (better caching)
COPY requirements.txt .

RUN pip install --upgrade pip

# Install torch stack first (cached layer)
RUN pip install torch==2.1.2+cpu torchaudio==2.1.2+cpu \
    --extra-index-url https://download.pytorch.org/whl/cpu

# Then rest
RUN pip install --no-cache-dir -r requirements.txt


# Copy application code
COPY . .

# Expose port
EXPOSE 8000

# Start the API
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
