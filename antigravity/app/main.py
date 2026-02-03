"""
FastAPI application for emotion recognition from audio.
Updated for Multi-Modal Feature Stack (52 dims) pipeline.
"""

import os
import tempfile
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydub import AudioSegment

from app.inference import load_resources, predict_emotion

# Create FastAPI app
app = FastAPI(
    title="Emotion Recognition API",
    description="Classify emotion from voice recordings using Multi-Modal CNN",
    version="2.0.0"
)

# Constants (MUST MATCH TRAINING)
MAX_DURATION_MS = 10000  # 10 seconds max
TARGET_SAMPLE_RATE = 22050  # Match training sample rate
ALLOWED_EXTENSIONS = {'.wav', '.mp3', '.m4a', '.ogg', '.flac'}

# Static files directory
STATIC_DIR = Path(__file__).parent / "static"


@app.on_event("startup")
async def startup_event():
    """Load model, scaler, and encoder into memory on startup."""
    print("=" * 50)
    print("Starting Emotion Recognition API...")
    print("=" * 50)
    load_resources()
    print("All resources loaded. Ready to serve predictions.")
    print("=" * 50)


@app.get("/", response_class=FileResponse)
async def serve_index():
    """Serve the main HTML page."""
    return FileResponse(STATIC_DIR / "index.html")


# Mount static files AFTER the index route
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Predict emotion from uploaded audio file.
    
    Pipeline:
    1. Convert to WAV (22050 Hz, mono)
    2. Hard-cut at 10 seconds
    3. Extract 52-dim multi-modal features (MFCC + Chroma + Mel)
    4. Apply scaler normalization
    5. Run voting mechanism across windows
    
    Accepts: .wav, .mp3, .m4a, .ogg, .flac
    
    Returns: {
        "emotion": "happy",
        "confidence": 85.5,
        "breakdown": {"angry": 5.2, "happy": 85.5, ...},
        "windows_analyzed": 3
    }
    """
    # Validate file extension
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
        )
    
    # Create temp directory for processing
    temp_dir = tempfile.mkdtemp()
    temp_input = os.path.join(temp_dir, f"input{file_ext}")
    temp_output = os.path.join(temp_dir, "processed.wav")
    
    try:
        # Save uploaded file
        content = await file.read()
        with open(temp_input, "wb") as f:
            f.write(content)
        
        # Load and process audio with pydub (uses FFmpeg)
        audio = AudioSegment.from_file(temp_input)
        
        # Hard-cut at 10 seconds
        if len(audio) > MAX_DURATION_MS:
            audio = audio[:MAX_DURATION_MS]
        
        # Convert to WAV with target sample rate (22050 Hz, mono)
        audio = audio.set_frame_rate(TARGET_SAMPLE_RATE)
        audio = audio.set_channels(1)
        audio.export(temp_output, format="wav")
        
        # Run prediction with voting mechanism
        result = predict_emotion(temp_output)
        
        return result
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing audio: {str(e)}"
        )
    
    finally:
        # Cleanup temp files
        try:
            if os.path.exists(temp_input):
                os.remove(temp_input)
            if os.path.exists(temp_output):
                os.remove(temp_output)
            if os.path.exists(temp_dir):
                os.rmdir(temp_dir)
        except Exception:
            pass  # Ignore cleanup errors


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "version": "2.0.0",
        "features": "Multi-Modal (MFCC+Chroma+Mel)"
    }