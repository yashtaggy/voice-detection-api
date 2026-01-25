import torch
import torchaudio
import numpy as np
from transformers import Wav2Vec2Processor, Wav2Vec2Model

MODEL_NAME = "facebook/wav2vec2-base"
TARGET_SR = 16000

# Load models once (important for performance)
processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
wav2vec = Wav2Vec2Model.from_pretrained(MODEL_NAME)
wav2vec.eval()

def _load_audio(path):
    waveform, sr = torchaudio.load(path)
    waveform = waveform.mean(dim=0)
    if sr != TARGET_SR:
        waveform = torchaudio.functional.resample(waveform, sr, TARGET_SR)
    return waveform

def _extract_features(audio_path):
    audio = _load_audio(audio_path)

    inputs = processor(
        audio,
        sampling_rate=TARGET_SR,
        return_tensors="pt"
    )

    with torch.no_grad():
        outputs = wav2vec(**inputs)

    emb = outputs.last_hidden_state.squeeze(0).numpy()

    emb_var = emb.var(axis=0)
    emb_var_mean = float(emb_var.mean())

    deltas = np.diff(emb, axis=0)
    delta_energy = float(np.mean(np.linalg.norm(deltas, axis=1)))

    energy = float(np.mean(audio.numpy() ** 2))
    zcr = float(np.mean(np.abs(np.diff(np.sign(audio.numpy())))) / 2)

    return {
        "emb_var_mean": emb_var_mean,
        "delta_energy": delta_energy,
        "energy": energy,
        "zcr": zcr
    }

def classify_audio(audio_path: str) -> dict:
    features = _extract_features(audio_path)

    score = 0.0

    if features["emb_var_mean"] < 0.12:
        score += 0.4
    if features["delta_energy"] > 3.0:
        score += 0.3
    if features["energy"] < 0.02:
        score += 0.2
    if features["zcr"] > 0.12:
        score += 0.1

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
