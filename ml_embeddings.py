import torch
import torchaudio
import numpy as np
import librosa
from transformers import Wav2Vec2Processor, Wav2Vec2Model

MODEL_NAME = "facebook/wav2vec2-base"
TARGET_SR = 16000

processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
model = Wav2Vec2Model.from_pretrained(MODEL_NAME)
model.eval()

def _load_audio(path):
    audio, sr = librosa.load(path, sr=TARGET_SR, mono=True)
    return torch.tensor(audio)

    if sr != TARGET_SR:
        waveform = torchaudio.functional.resample(waveform, sr, TARGET_SR)

    return waveform

if __name__ == "__main__":
    audio = _load_audio("test.mp3")

    inputs = processor(
        audio,
        sampling_rate=TARGET_SR,
        return_tensors="pt"
    )

    with torch.no_grad():
        outputs = model(**inputs)

    embeddings = outputs.last_hidden_state.mean(dim=1)

    print("Embedding shape:", embeddings.shape)
