FROM python:3.11-slim

WORKDIR /app

# Install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Cache busting step - this ensures fresh builds every time
RUN echo "Build timestamp: $(date)" > /tmp/build_timestamp

# Copy application code
COPY . .

# Expose port
EXPOSE 8003

# Run the application
CMD ["python", "main.py"]