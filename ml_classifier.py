import torch
import torchaudio
import numpy as np
import librosa
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

MODEL_NAME = "facebook/wav2vec2-base"
TARGET_SR = 16000

processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
wav2vec = Wav2Vec2Model.from_pretrained(MODEL_NAME)
wav2vec.eval()

def _load_audio(path):
    audio, sr = librosa.load(path, sr=TARGET_SR, mono=True)
    return torch.tensor(audio)

def extract_features(path):
    audio = _load_audio(path)

    inputs = processor(audio, sampling_rate=TARGET_SR, return_tensors="pt")
    with torch.no_grad():
        outputs = wav2vec(**inputs)

    emb = outputs.last_hidden_state.squeeze(0).numpy()
    # Temporal smoothness (frame-to-frame change)
    deltas = np.diff(emb, axis=0)
    delta_energy = np.mean(np.linalg.norm(deltas, axis=1))


    # Embedding statistics
    emb_mean = emb.mean(axis=0)
    emb_var = emb.var(axis=0)

    # Signal features
    energy = np.mean(audio.numpy() ** 2)
    zcr = np.mean(np.abs(np.diff(np.sign(audio.numpy())))) / 2

    return {
    "emb_var_mean": float(emb_var.mean()),
    "emb_var_std": float(emb_var.std()),
    "delta_energy": float(delta_energy),
    "energy": float(energy),
    "zcr": float(zcr),
}

# Temporary heuristic classifier (until trained model)
def classify(features):
    score = 0.0

    # Signal 1: Low embedding variance (AI tends to be lower)
    if features["emb_var_mean"] < 0.12:
        score += 0.4

    # Signal 2: Unnaturally high delta energy (AI overcompensation)
    if features["delta_energy"] > 3.0:
        score += 0.3

    # Signal 3: Energy smoothness
    if features["energy"] < 0.02:
        score += 0.2

    # Signal 4: Zero-crossing regularity
    if features["zcr"] > 0.12:
        score += 0.1

    # Final decision
    if score >= 0.6:
        return {
            "label": "AI_GENERATED",
            "confidence": round(min(score, 1.0), 2),
            "explanation": "Multiple deep acoustic indicators consistent with AI-generated speech"
        }

    return {
        "label": "HUMAN",
        "confidence": round(1.0 - score, 2),
        "explanation": "Acoustic patterns consistent with natural human speech"
    }

if __name__ == "__main__":
    feats = extract_features("test.mp3")
    print("Features:", feats)
    result = classify(feats)
    print(result)
