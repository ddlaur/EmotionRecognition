"""
Inference module for emotion recognition model.
Implements Multi-Modal Feature Stack (52 dims) with Voting Mechanism.
"""

import os
import numpy as np
import librosa
import joblib
from tensorflow import keras

# Constants (MUST MATCH TRAINING)
SR = 22050
DURATION = 2.5  # Window size in seconds
OVERLAP = 1.0   # Overlap in seconds

# Paths to artifacts
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, '..', 'models', 'emotion_recognition_model.h5')
SCALER_PATH = os.path.join(BASE_DIR, '..', 'models', 'scaler.pkl')
ENCODER_PATH = os.path.join(BASE_DIR, '..', 'models', 'label_encoder.pkl')

# Global cache (loaded once at startup)
_model = None
_scaler = None
_label_encoder = None


def load_resources():
    """
    Load model, scaler, and label encoder into memory.
    Should be called once at application startup.
    """
    global _model, _scaler, _label_encoder
    
    if _model is None:
        print(f"Loading model from: {MODEL_PATH}")
        _model = keras.models.load_model(MODEL_PATH)
        print("Model loaded.")
    
    if _scaler is None:
        print(f"Loading scaler from: {SCALER_PATH}")
        _scaler = joblib.load(SCALER_PATH)
        print("Scaler loaded.")
    
    if _label_encoder is None:
        print(f"Loading label encoder from: {ENCODER_PATH}")
        _label_encoder = joblib.load(ENCODER_PATH)
        print("Label encoder loaded.")
    
    return _model, _scaler, _label_encoder


def extract_multimodal_features(y, sr):
    """
    Extract 52-dimensional feature stack from audio segment.
    Features: MFCC(20) + Chroma(12) + MelSpectrogram(20)
    
    Args:
        y: Audio waveform (numpy array)
        sr: Sample rate
    
    Returns:
        Feature array of shape (Time, 52)
    """
    try:
        # 1. MFCC (Timbre) - 20 coefficients
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        
        # 2. Chroma (Pitch) - 12 coefficients
        chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=12)
        
        # 3. Mel Spectrogram (Energy) - 20 bands
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=20)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        
        # Stack: (52, Time)
        combined = np.vstack([mfcc, chroma, mel_db])
        
        # Transpose to (Time, 52)
        return combined.T
        
    except Exception as e:
        print(f"Feature extraction error: {e}")
        return None


def predict_emotion(audio_path: str) -> dict:
    """
    Predict emotion from an audio file using voting mechanism.
    
    Steps:
    1. Load audio and downsample to 22050 Hz
    2. Slice into 2.5s windows with overlap
    3. Extract 52-dim features per window
    4. Apply scaler normalization
    5. Run model prediction on all windows
    6. Aggregate predictions via mean probability
    
    Args:
        audio_path: Path to audio file (WAV format recommended)
    
    Returns:
        Dictionary with emotion, confidence, and breakdown
    """
    model, scaler, label_encoder = load_resources()
    
    # 1. Load audio
    y, sr = librosa.load(audio_path, sr=SR)
    
    # 2. Windowing setup
    window_length = int(DURATION * sr)
    step_length = int((DURATION - OVERLAP) * sr)
    
    # Pad if audio is shorter than one window
    if len(y) < window_length:
        y = np.pad(y, (0, window_length - len(y)), mode='constant')
    
    # 3. Extract features from each window
    windows = []
    for i in range(0, len(y) - window_length + 1, step_length):
        segment = y[i : i + window_length]
        features = extract_multimodal_features(segment, sr)
        if features is not None:
            windows.append(features)
    
    if not windows:
        return {
            'emotion': 'unknown',
            'confidence': 0.0,
            'error': 'No valid audio segments found'
        }
    
    # Convert to array: (N_windows, Time, 52)
    features_array = np.array(windows)
    N, Time, Feat = features_array.shape
    
    # 4. Apply scaler normalization
    # Flatten: (N*Time, 52)
    features_flat = features_array.reshape(-1, Feat)
    features_scaled = scaler.transform(features_flat)
    # Reshape back: (N, Time, 52)
    features_final = features_scaled.reshape(N, Time, Feat)
    
    # 5. Predict on all windows
    predictions = model.predict(features_final, verbose=0)
    
    # 6. Aggregate via mean probability
    avg_prediction = np.mean(predictions, axis=0)
    
    # Get result
    predicted_idx = np.argmax(avg_prediction)
    emotion = label_encoder.inverse_transform([predicted_idx])[0]
    confidence = float(avg_prediction[predicted_idx])
    
    # Build breakdown
    breakdown = {}
    for i, label in enumerate(label_encoder.classes_):
        breakdown[label] = round(float(avg_prediction[i]) * 100, 2)
    
    return {
        'emotion': emotion,
        'confidence': round(confidence * 100, 2),
        'breakdown': breakdown,
        'windows_analyzed': N
    }