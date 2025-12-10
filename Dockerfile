FROM python:3.9-slim

# Set work directory
WORKDIR /app

# Copy project files
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the source code
COPY src ./src
COPY configs ./configs
COPY models ./models

# Expose API port
EXPOSE 8000

# FastAPI startup command
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
