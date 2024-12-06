# Use a lightweight base image with Python 3.9
FROM python:3.9-slim

# Install make and system dependencies
RUN apt-get update && apt-get install -y make && rm -rf /var/lib/apt/lists/*

# Set working directory in the container
WORKDIR /app

# Copy the application files into the container
COPY . /app

# Set PYTHONPATH
ENV PYTHONPATH=/app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Default command to run
CMD ["make"]
