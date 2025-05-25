FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install sentencepiece

# Set working directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install whisper (separate from requirements.txt for better caching)
RUN pip install --no-cache-dir git+https://github.com/openai/whisper.git 

# Copy all files
COPY . .

# Expose port
EXPOSE 5000

# Run app
CMD ["python3", "app.py"]