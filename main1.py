import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import librosa
import librosa.display
from tqdm import tqdm
import warnings
import tensorflow as tf
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from tensorflow import keras
from keras import layers, models, Sequential
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

def load_ravdess_data(data_path):
    print("Scanning files...")
    data = []
    
    # Walk through all subdirectories
    for root, dirs, files in os.walk(data_path):
        for file in files:
            if file.endswith('.wav'):
                try:
                    parts = file.split('-')
                    
                    # Filter: Skip Calm(02), Fearful(06), Disgust(07), Surprised(08)
                    if parts[2] in ["02", "06", "07", "08"]:
                        continue
                        
                    emotion_map = {
                        '01': 'neutral',
                        '03': 'happy',
                        '04': 'sad',
                        '05': 'angry'
                    }
                    
                    emotion = emotion_map.get(parts[2])
                    if not emotion: continue
                        
                    # Extract Actor ID (parts[6])
                    actor_id = int(parts[6].split('.')[0])
                    
                    data.append({
                        'path': os.path.join(root, file),
                        'emotion': emotion,
                        'actor': actor_id
                    })
                    
                except Exception as e:
                    continue
                    
    df = pd.DataFrame(data)
    print(f"Found {len(df)} files.")
    return df

def extract_multimodal_features(y, sr):
    """
    Extracts 52 features: 20 MFCC + 12 Chroma + 20 Mel-Spec
    """
    try:
        # 1. MFCC (Timbre)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        
        # 2. Chroma (Pitch)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=12)
        
        # 3. Mel (Energy)
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=20)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        
        # Stack: (52, Time)
        combined = np.vstack([mfcc, chroma, mel_db])
        return combined.T # Return (Time, 52)
        
    except Exception as e:
        print(f"Feature extraction error: {e}")
        return None

def augment_audio(y, sr):
    """
    Returns a LIST of 3 variations: Noise, Pitch, Speed
    """
    # 1. Noise Injection
    noise_amp = 0.035 * np.random.uniform() * np.amax(y)
    y_noise = y + noise_amp * np.random.normal(size=y.shape)
    
    # 2. Pitch Shift (Randomly +/- 2 semitones)
    step = np.random.uniform(-2, 2)
    y_pitch = librosa.effects.pitch_shift(y=y, sr=sr, n_steps=step)
    
    # 3. Speed Stretch (Randomly 80% to 120% speed)
    rate = np.random.uniform(0.8, 1.2)
    y_speed = librosa.effects.time_stretch(y=y, rate=rate)
    
    return [y_noise, y_pitch, y_speed]

def prepare_features_windowed(df, duration=2.5, overlap=1.0, augment=False):
    """
    Main processing loop. Handles both Normal and Augmented extraction.
    """
    X = []
    y = []
    
    print(f"   Processing {len(df)} files (Augment={augment})...")
    
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        try:
            # Load audio ONCE
            audio, sr = librosa.load(row['path'], sr=22050)
            
            # Prepare versions to process
            audio_versions = [audio]
            
            # If Training, add the 3 augmented versions
            if augment:
                audio_versions.extend(augment_audio(audio, sr))
            
            # Process ALL versions (Original + Augments)
            for version in audio_versions:
                
                # Windowing specs
                window_length = int(duration * sr)
                step_length = int((duration - overlap) * sr)
                
                # Case A: Audio is shorter than window -> Pad it
                if len(version) < window_length:
                    pad_width = window_length - len(version)
                    window = np.pad(version, (0, pad_width), mode='constant')
                    
                    feats = extract_multimodal_features(window, sr)
                    if feats is not None:
                        X.append(feats)
                        y.append(row['emotion'])
                        
                # Case B: Audio is longer -> Slide window
                else:
                    for i in range(0, len(version) - window_length + 1, step_length):
                        window = version[i : i + window_length]
                        
                        feats = extract_multimodal_features(window, sr)
                        if feats is not None:
                            X.append(feats)
                            y.append(row['emotion'])
                            
        except Exception as e:
            print(f"Error on {row['path']}: {e}")
            continue
            
    return np.array(X), np.array(y)

def build_cnn_model(input_shape, num_classes):
    model = Sequential()
    
    # Block 1
    model.add(layers.Conv1D(64, 5, padding='same', input_shape=input_shape))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling1D(2))
    model.add(layers.Dropout(0.3))

    # Block 2
    model.add(layers.Conv1D(128, 5, padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling1D(2))
    model.add(layers.Dropout(0.3))

    # Block 3
    model.add(layers.Conv1D(256, 5, padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling1D(2))
    model.add(layers.Dropout(0.3))

    # Dense Layers
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.4))
    model.add(layers.Dense(num_classes, activation='softmax'))

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=['accuracy']
    )
    
    return model

def main():
    DATA_PATH = "RAVDESS_data/"
    
    # 1. Load Data
    print("\n[1/7] Loading Data...")
    df = load_ravdess_data(DATA_PATH)
    
    # 2. Subject-Independent Split (Split by Actor, not by File)
    print("\n[2/7] Splitting by Actor...")
    unique_actors = df['actor'].unique()
    train_actors, test_actors = train_test_split(unique_actors, test_size=0.2, random_state=42)
    
    train_df = df[df['actor'].isin(train_actors)]
    test_df = df[df['actor'].isin(test_actors)]
    
    print(f"   Training Actors: {train_actors}")
    print(f"   Testing Actors: {test_actors}")
    
    # 3. Feature Extraction (This will take time due to Augmentation)
    print("\n[3/7] Extracting Features (TRAIN - Augmented)...")
    # Train data gets augment=True
    X_train, y_train = prepare_features_windowed(train_df, augment=True)
    
    print("\n[3/7] Extracting Features (TEST - Clean)...")
    # Test data gets augment=False
    X_test, y_test = prepare_features_windowed(test_df, augment=False)
    
    # 4. Encode Labels
    print("\n[4/7] Encoding Labels...")
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test) # Use same encoder
    
    y_train_cat = to_categorical(y_train_encoded)
    y_test_cat = to_categorical(y_test_encoded)
    
    joblib.dump(label_encoder, 'label_encoder.pkl') # Save for prediction
    
    # 5. Scale Features
    print("\n[5/7] Scaling Features...")
    scaler = StandardScaler()
    
    # Reshape (N, Time, 52) -> (N*Time, 52)
    N_train, Time, Feat = X_train.shape
    X_train_flat = X_train.reshape(-1, Feat)
    
    N_test, _, _ = X_test.shape
    X_test_flat = X_test.reshape(-1, Feat)
    
    # Fit on TRAIN only
    scaler.fit(X_train_flat)
    
    # Transform
    X_train_scaled = scaler.transform(X_train_flat)
    X_test_scaled = scaler.transform(X_test_flat)
    
    # Reshape back to 3D
    X_train = X_train_scaled.reshape(N_train, Time, Feat)
    X_test = X_test_scaled.reshape(N_test, Time, Feat)
    
    joblib.dump(scaler, 'scaler.pkl') # Save for prediction
    
    # 6. Build & Train
    print(f"\n[6/7] Training Model (Input Shape: {X_train.shape})...")
    model = build_cnn_model((Time, Feat), len(label_encoder.classes_))
    
    history = model.fit(
        X_train, y_train_cat,
        validation_data=(X_test, y_test_cat),
        epochs=50,
        batch_size=32,
        callbacks=[
            EarlyStopping(patience=8, restore_best_weights=True),
            ReduceLROnPlateau(patience=4, factor=0.5)
        ]
    )
    
    # 7. Evaluate
    print("\n[7/7] Evaluation...")
    model.save('emotion_recognition_model.h5')
    
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    acc = accuracy_score(y_test_encoded, y_pred_classes)
    print(f"   Test Accuracy: {acc*100:.2f}%")
    
    print(classification_report(y_test_encoded, y_pred_classes, target_names=label_encoder.classes_))
    
    return model, label_encoder

if __name__ == "__main__":
    main()
