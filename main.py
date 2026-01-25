from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from fastapi import Request
from fastapi.responses import JSONResponse
from ml_engine import classify_audio
from dotenv import load_dotenv
import base64
import tempfile
import os

load_dotenv()

app = FastAPI()

API_KEY = os.getenv("API_KEY")
if not API_KEY:
    raise RuntimeError("API_KEY not set in environment")

ALLOWED_LANGUAGES = {"Tamil", "English", "Hindi", "Malayalam", "Telugu"}
ALLOWED_AUDIO_FORMAT = "mp3"

@app.get("/health")
def health_check():
    return {"status": "ok"}

def verify_api_key(x_api_key: str = Header(None)):
    if x_api_key != API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key"
        )

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "status": "error",
            "message": exc.detail
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={
            "status": "error",
            "message": "Internal server error"
        }
    )
    
class VoiceDetectionRequest(BaseModel):
    language: str
    audioFormat: str
    audioBase64: str

@app.get("/protected")
def protected_endpoint(x_api_key: str = Header(None)):
    verify_api_key(x_api_key)
    return {"message": "API key is valid"}

@app.post("/api/voice-detection")
def voice_detection(
    request: VoiceDetectionRequest,
    x_api_key: str = Header(None)
):
    verify_api_key(x_api_key)

    if request.language not in ALLOWED_LANGUAGES:
        raise HTTPException(status_code=400, detail="Unsupported language")

    if request.audioFormat.lower() != ALLOWED_AUDIO_FORMAT:
        raise HTTPException(
            status_code=400,
            detail="Unsupported audio format. Only mp3 is allowed"
        )

    try:
        audio_bytes = base64.b64decode(request.audioBase64, validate=True)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid Base64 audio data")

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio:
            temp_audio.write(audio_bytes)
            temp_audio_path = temp_audio.name
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to process audio file")
    
    try:
        result = classify_audio(temp_audio_path)
    finally:
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)

    return {
        "status": "success",
        "language": request.language,
        "classification": result["label"],
        "confidenceScore": result["confidence"],
        "explanation": result["explanation"]
    }

# Only for local dev; Render ignores this
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))