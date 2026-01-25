import torch
import torchaudio
import numpy as np
from transformers import Wav2Vec2Processor, Wav2Vec2Model

MODEL_NAME = "facebook/wav2vec2-base"
TARGET_SR = 16000

processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
model = Wav2Vec2Model.from_pretrained(MODEL_NAME)
model.eval()

def load_audio(path):
    waveform, sr = torchaudio.load(path)
    waveform = waveform.mean(dim=0)

    if sr != TARGET_SR:
        waveform = torchaudio.functional.resample(waveform, sr, TARGET_SR)

    return waveform

if __name__ == "__main__":
    audio = load_audio("test.mp3")

    inputs = processor(
        audio,
        sampling_rate=TARGET_SR,
        return_tensors="pt"
    )

    with torch.no_grad():
        outputs = model(**inputs)

    embeddings = outputs.last_hidden_state.mean(dim=1)

    print("Embedding shape:", embeddings.shape)
