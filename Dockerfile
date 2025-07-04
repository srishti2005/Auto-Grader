FROM python:3.10-slim

# Install OS-level dependencies
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    libglib2.0-0 \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /app

# Copy project files
COPY . .

# Upgrade pip & install Python dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# NLTK data (punkt tokenizer)
RUN python -m nltk.downloader punkt

# Expose port
EXPOSE 8000

# Run app with gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "app:app"]
