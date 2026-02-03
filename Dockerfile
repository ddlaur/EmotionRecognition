FROM python:3.9-slim

# Install system libraries for audio (libsndfile is required for librosa)
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 1. Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 2. Copy the Code AND the Model
# Because we ignored RAVDESS_data in .gitignore, this COPY command
# will copy the 'antigravity' folder (with the model inside) 
# but SKIP the 'RAVDESS_data' folder automatically.
COPY . .

# 3. Run the website
EXPOSE 5000
CMD ["python", "-m", "antigravity.app"]