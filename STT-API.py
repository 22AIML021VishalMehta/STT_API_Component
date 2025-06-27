from fastapi import FastAPI, UploadFile, File, HTTPException
from pathlib import Path
from tempfile import NamedTemporaryFile
import shutil
import os
from dotenv import load_dotenv
from sarvamai import SarvamAI

load_dotenv()
SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")
client = SarvamAI(api_subscription_key=SARVAM_API_KEY)

app = FastAPI()

def transcribe_audio_file(audio_path: str, model: str = "saarika:v2.5"):
    audio_file_path = Path(audio_path)
    if not audio_file_path.is_file():
        raise FileNotFoundError(f"{audio_path} not found")

    with audio_file_path.open("rb") as af:
        response = client.speech_to_text.transcribe(
            file=af,
            model=model
        )
    return response


@app.post("/transcribe/")
async def transcribe_audio_endpoint(file: UploadFile = File(...)):
    try:
        with NamedTemporaryFile(delete=False, suffix=".mp3") as temp_file:
            shutil.copyfileobj(file.file, temp_file)
            temp_path = temp_file.name

        result = transcribe_audio_file(temp_path)
        os.remove(temp_path)
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
