import base64
import os
from flask import Flask, jsonify, send_file  # 需要安裝 Flask: pip install flask
from pydub import AudioSegment  # 需要安裝 pydub: pip install pydub
import io

app = Flask(__name__)

def wav_to_base64(file_path, compress=True):
    """
    將 .wav 文件轉換為 base64 編碼的字符串，以便通過 API 發送。
    如果 compress=True 轉換為 MP3 以減小大小。

    :param file_path: .wav 文件的路徑
    :param compress: 是否壓縮為 MP3
    :return: base64 編碼的字符串 和 內容類型
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件 {file_path} 不存在")

    if compress:
        # 載入 WAV 並轉換為 MP3
        audio = AudioSegment.from_wav(file_path)
        # 設置比特率為 128k 以減小大小
        audio.export("/tmp/temp_audio.mp3", format="mp3", bitrate="128k")
        file_to_encode = "/tmp/temp_audio.mp3"
        content_type = "audio/mpeg"
    else:
        file_to_encode = file_path
        content_type = "audio/wav"

    with open(file_to_encode, 'rb') as f:
        audio_data = f.read()

    base64_string = base64.b64encode(audio_data).decode('utf-8')
    return base64_string, content_type

@app.route('/get_audio/<filename>')
def get_audio(filename):
    file_path = f"{filename}.wav"
    try:
        base64_audio, content_type = wav_to_base64(file_path, compress=True)
        return jsonify({'audio_base64': base64_audio, 'content_type': content_type})
    except FileNotFoundError:
        return jsonify({'error': '文件不存在'}), 404

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import secrets

UPLOAD_DIR = Path("./uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

MAX_MB = 20
MAX_BYTES = MAX_MB * 1024 * 1024


@app.post("/upload/pdf")
async def upload_pdf(file: UploadFile = File(...)):
    # 1) 基本檢查：副檔名 & content-type（兩者都檢查比較穩）
    filename = file.filename or "upload.pdf"
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only .pdf files are allowed")
    if file.content_type not in ("application/pdf", "application/x-pdf"):
        raise HTTPException(status_code=400, detail=f"Invalid content-type: {file.content_type}")

    # 2) 讀取（同時做大小限制）
    data = await file.read()
    if len(data) > MAX_BYTES:
        raise HTTPException(status_code=413, detail=f"File too large (>{MAX_MB}MB)")

    # 3) 簡單檢查 PDF magic header
    if not data.startswith(b"%PDF"):
        raise HTTPException(status_code=400, detail="File does not look like a valid PDF")

    # 4) 存檔（避免檔名衝突）
    safe_name = f"{secrets.token_hex(8)}_{Path(filename).name}"
    save_path = UPLOAD_DIR / safe_name
    save_path.write_bytes(data)

    return {
        "ok": True,
        "original_filename": filename,
        "saved_filename": safe_name,
        "path": str(save_path),
        "bytes": len(data),
    }

@app.route('/get_audio_file/<filename>')
def get_audio_file(filename):
    """
    直接發送壓縮後的音頻文件，而不是 base64。
    這對於大文件更高效，因為避免了編碼/解碼開銷。
    """
    file_path = f"{filename}.wav"
    if not os.path.exists(file_path):
        return jsonify({'error': '文件不存在'}), 404

    # 壓縮為 MP3
    audio = AudioSegment.from_wav(file_path)
    audio.export("/tmp/temp_audio.mp3", format="mp3", bitrate="128k")

    return send_file("/tmp/temp_audio.mp3", mimetype="audio/mpeg", as_attachment=False)

if __name__ == "__main__":
    # 示例用法（無 API）
    file_path = 'output.wav'  # 替換為您的 .wav 文件路徑
    try:
        base64_audio, content_type = wav_to_base64(file_path, compress=True)
        print(f"壓縮後的 Base64 編碼的音頻數據 (前100字符), 類型: {content_type}")
        print(base64_audio[:100] + "...")
        print(f"原始大小: {os.path.getsize(file_path)} bytes")
        print(f"壓縮後大小估計: {len(base64_audio) * 3 // 4} bytes")  # 粗略估計
    except FileNotFoundError as e:
        print(e)

    # 運行 Flask API（取消註釋以運行）
    # app.run(debug=True)


import os
import secrets
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from openai import OpenAI

app = FastAPI(title="Instruction + Voice-to-Text API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # production 請改成你的前端 domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# OpenAI client (reads OPENAI_API_KEY from env by default)
client = OpenAI()

# Audio constraints (per OpenAI docs: 25MB)
MAX_AUDIO_BYTES = 25 * 1024 * 1024
ALLOWED_AUDIO_EXT = {".mp3", ".mp4", ".mpeg", ".mpga", ".m4a", ".wav", ".webm"}

UPLOAD_DIR = Path("./uploads_audio")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


class InstructionIn(BaseModel):
    instruction: str
    session_id: Optional[str] = None


@app.post("/instruction")
async def post_instruction(payload: InstructionIn):
    text = payload.instruction.strip()
    if not text:
        raise HTTPException(status_code=400, detail="instruction is empty")

    return {
        "ok": True,
        "type": "text_instruction",
        "session_id": payload.session_id,
        "instruction": text,
    }


@app.post("/instruction/voice")
async def post_voice_instruction(
    file: UploadFile = File(...),
    session_id: Optional[str] = None,
):
    # 1) 基本檢查：副檔名 & 大小
    filename = file.filename or "audio"
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_AUDIO_EXT:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported audio type: {ext}. Allowed: {sorted(ALLOWED_AUDIO_EXT)}",
        )

    data = await file.read()
    if len(data) > MAX_AUDIO_BYTES:
        raise HTTPException(status_code=413, detail="Audio file too large (>25MB)")

    # 2) (可選) 先存本地，方便 debug/追蹤
    safe_name = f"{secrets.token_hex(8)}_{Path(filename).name}"
    save_path = UPLOAD_DIR / safe_name
    save_path.write_bytes(data)

    # 3) 丟給 Whisper 轉文字 (speech -> text)
    # OpenAI transcription endpoint usage: client.audio.transcriptions.create(model=..., file=...)
    # whisper-1 仍可用於 transcriptions。 [oai_citation:1‡OpenAI Platform](https://platform.openai.com/docs/guides/speech-to-text)
    
    with open(save_path, "rb") as f:
        transcription = client.audio.transcriptions.create(
            model="whisper-1",
            file=f,
        )

    text = (getattr(transcription, "text", None) or transcription.get("text", "")).strip()
    if not text:
        raise HTTPException(status_code=500, detail="Empty transcription result")

    # 4) 把轉出來的文字當成 instruction 使用
    return {
        "ok": True,
        "type": "voice_instruction",
        "session_id": session_id,
        "audio_saved_as": safe_name,
        "instruction": text,
    }