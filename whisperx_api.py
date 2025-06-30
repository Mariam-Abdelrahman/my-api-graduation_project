from fastapi import FastAPI, File, UploadFile
from starlette.responses import FileResponse
import whisperx
import tempfile
import torch
import os
#import json
import subprocess
from starlette.middleware import Middleware
from starlette.middleware.gzip import GZipMiddleware
from pymongo import MongoClient
from pydantic import BaseModel
import datetime
from langdetect import detect
from dotenv import load_dotenv  # استيراد مكتبة python-dotenv

# تحميل متغيرات البيئة من ملف .env
load_dotenv()

# إعدادات الاتصال بـ MongoDB
DB_URL = os.getenv("DB_URL")  # قراءة DB_URL من .env
client = MongoClient(DB_URL)
db = client["StellaLearnDB"]
collection = db["transcriptions"]

# تعريف Middleware
middleware = [Middleware(GZipMiddleware, minimum_size=1000)]
app = FastAPI(middleware=middleware)

# إعدادات الـ Device
device = "cuda" if torch.cuda.is_available() else "cpu"
model = whisperx.load_model("whisper-ct2", device)

# دالة تحويل MP4 لـ WAV
def convert_mp4_to_wav(mp4_path: str, wav_path: str):
    cmd = [
        'ffmpeg',
        '-i', mp4_path,
        '-acodec', 'pcm_s16le',
        '-ar', '8000',
        '-ac', '1',
        wav_path,
        '-y'
    ]
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

try:
    client.admin.command('ping')
    print("✅ Connected to MongoDB successfully!")
except Exception as e:
    print(f"❌ Connection failed: {e}")
    
# نموذج Pydantic للتحقق من البيانات (اختياري)
class Transcription(BaseModel):
    video_id: str
    filename: str
    transcript: str
    language: str
    created_at: str

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_mp4:
        tmp_mp4.write(await file.read())
        tmp_mp4_path = tmp_mp4.name

    # Convert MP4 to WAV
    wav_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
    convert_mp4_to_wav(tmp_mp4_path, wav_path)

    audio = whisperx.load_audio(wav_path)
    result = model.transcribe(audio, batch_size=4, language=None)

    if "segments" in result and result["segments"]:
        segments = result["segments"]  # قائمة بـ Segments فيها النص والوقت
    else:
        segments = [{"text": result.get("text", "No transcription available"), "start": 0.0, "end": 0.0}]
    
    language = result.get("language", "unknown")
    if language == "unknown":
        # محاولة كشف اللغة يدويًا من أول 100 كلمة
        text = " ".join([seg["text"] for seg in segments[:10] if seg.get("text")])
        if text:
            try:
                language = detect(text[:1000])  # أول 1000 حرف
            except:
                language = "unknown"
    # txt_path = f"{original_filename}.txt"
    # json_path = f"{original_filename}.json"

    # with open(txt_path, "w", encoding="utf-8") as txt_file:
    #     txt = " ".join([seg["text"] for seg in result["segments"]])
    #     txt_file.write(txt)

    # with open(json_path, "w", encoding="utf-8") as json_file:
    #     json.dump(result, json_file, ensure_ascii=False, indent=4)

    # حفظ النص في قاعدة البيانات
    transcription_data = {
        "video_id": os.path.basename(tmp_mp4_path),  # أو ID فريد
        "filename": file.filename,
        "transcript": segments,
        "language": language,
        "created_at": datetime.datetime.now().isoformat()
    }
    collection.insert_one(transcription_data)  # حفظ في MongoDB

    # Cleanup
    os.unlink(tmp_mp4_path)
    os.unlink(wav_path)

    return {
        "segments": segments,  # رجوع النص مع الـ Timestamps
        "language": language,
        "filename": file.filename,
        "db_status": "Transcription saved to database"
    }

