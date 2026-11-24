FROM python:3.12-slim

WORKDIR /workspace

# Install dependencies first for layer caching (Student Handbook Section 9)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application, package source, and model artifact
COPY src/ src/
COPY models/ models/
COPY app/ app/

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
