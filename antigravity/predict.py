import numpy as np
import librosa
import joblib
from tensorflow.keras.models import load_model

# Constants
SR = 22050
DURATION = 2.5

def extract_multimodal_features(y, sr):
    """
    MUST MATCH training extraction exactly.
    """
    try:
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=12)
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=20)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        
        combined = np.vstack([mfcc, chroma, mel_db])
        return combined.T
    except Exception as e:
        print(f"Error: {e}")
        return None

def load_resources():
    """
    Helper to load heavy files once.
    """
    print("Loading resources...")
    model = load_model('emotion_recognition_model.h5')
    scaler = joblib.load('scaler.pkl')
    label_encoder = joblib.load('label_encoder.pkl')
    return model, scaler, label_encoder

def predict_emotion(model, scaler, label_encoder, audio_path):
    """
    Predicts emotion using passed resources.
    Returns: (Emotion_String, Confidence_Score)
    """
    try:
        y, sr = librosa.load(audio_path, sr=SR)
        
        # Windowing Logic
        window_length = int(DURATION * sr)
        step_length = int((DURATION - 1.0) * sr) # 1.5s overlap
        
        # Pad if too short
        if len(y) < window_length:
            y = np.pad(y, (0, window_length - len(y)), mode='constant')
            
        windows = []
        
        # Sliding Window
        for i in range(0, len(y) - window_length + 1, step_length):
            window = y[i : i + window_length]
            features = extract_multimodal_features(window, sr)
            if features is not None:
                windows.append(features)
                
        if not windows:
            return "No valid audio", 0.0

        features_array = np.array(windows) # Shape: (N_Windows, Time, 52)
        
        # 3. Normalize
        N, Time, Feat = features_array.shape
        # Flatten to (N*Time, 52) to match Scaler
        features_flat = features_array.reshape(-1, Feat)
        features_scaled = scaler.transform(features_flat)
        # Reshape back to (N, Time, 52)
        features_final = features_scaled.reshape(N, Time, Feat)
        
        # 4. Predict
        predictions = model.predict(features_final, verbose=0)
        avg_pred = np.mean(predictions, axis=0)
        
        # 5. Result
        max_idx = np.argmax(avg_pred)
        emotion = label_encoder.inverse_transform([max_idx])[0]
        conf = avg_pred[max_idx] * 100
        
        # Return the values instead of printing
        return emotion.upper(), conf

    except Exception as e:
        print(f"Prediction Error: {e}")
        return "Error", 0.0

if __name__ == "__main__":
    # Test logic
    # 1. Load once
    model, scaler, encoder = load_resources()
    
    # 2. Predict multiple times quickly
    em, conf = predict_emotion(model, scaler, encoder, "Fericit_sklearn(neutru).wav")
    print(f"Result: {em} ({conf:.1f}%)")