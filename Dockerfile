FROM python:3.10-slim

# Install system dependencies for audio and libraries
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install
COPY ./requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Set environment variables for Python
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Expose port (Railway will provide this via PORT env)
EXPOSE 8080

# Command to run the application
# Use shell form to allow environment variable expansion for PORT
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8080}
