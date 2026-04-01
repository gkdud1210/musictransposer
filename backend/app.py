"""
ScoreShift Backend — AI 악보 조 변환기
FastAPI + OpenAI Vision + music21
"""

import os
import io
import base64
import json
import logging
from typing import Optional, List

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn

# ── 로깅 설정 ──────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ── 앱 생성 ───────────────────────────────────────────────────
app = FastAPI(
    title="ScoreShift API",
    description="AI 악보 분석 및 조 변환 서비스",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # 프로덕션에서는 실제 도메인으로 변경
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── 환경 변수 로드 ─────────────────────────────────────────────
from dotenv import load_dotenv
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# ── music21 임포트 (설치된 경우) ───────────────────────────────
try:
    from music21 import stream, note, key, pitch, interval, chord
    MUSIC21_AVAILABLE = True
    logger.info("✅ music21 사용 가능")
except ImportError:
    MUSIC21_AVAILABLE = False
    logger.warning("⚠️  music21 미설치 — 로컬 알고리즘으로 대체합니다")

# ── OpenAI 임포트 (설치된 경우) ────────────────────────────────
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = bool(OPENAI_API_KEY)
    if OPENAI_AVAILABLE:
        openai_client = OpenAI(api_key=OPENAI_API_KEY)
        logger.info("✅ OpenAI API 사용 가능")
    else:
        logger.warning("⚠️  OPENAI_API_KEY 없음 — 데모 데이터로 대체합니다")
except ImportError:
    OPENAI_AVAILABLE = False
    logger.warning("⚠️  openai 미설치")

# ═══════════════════════════════════════════════════════════════
#   Pydantic 모델
# ═══════════════════════════════════════════════════════════════

class NoteData(BaseModel):
    pitch: str = "C"
    octave: int = 4
    duration: str = "q"
    measure: int = 1
    beat: float = 1.0
    type: str = "note"        # "note" | "rest"
    accidental: Optional[str] = None
    original_pitch: Optional[str] = None
    original_octave: Optional[int] = None

class TransposeRequest(BaseModel):
    notes: List[NoteData]
    original_key: str = "C"
    target_key: str = "G"
    semitones: int = 7
    time_signature: str = "4/4"
    tempo: Optional[int] = 120
    maintain_octave: bool = True
    use_flats: bool = False

# ═══════════════════════════════════════════════════════════════
#   유틸리티
# ═══════════════════════════════════════════════════════════════

ALL_NOTES   = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
FLAT_NOTES  = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B']

FLAT_MAP = {'Db': 'C#', 'Eb': 'D#', 'Gb': 'F#', 'Ab': 'G#', 'Bb': 'A#', 'Cb': 'B'}

def normalize_pitch(p: str) -> str:
    return FLAT_MAP.get(p, p)

def transpose_note_local(note_data: NoteData, semitones: int, use_flats: bool = False) -> NoteData:
    """로컬 알고리즘으로 음표 1개 조 변환"""
    if note_data.type == "rest":
        return note_data.copy()

    note_arr = FLAT_NOTES if use_flats else ALL_NOTES
    norm_pitch = normalize_pitch(note_data.pitch)
    original_idx = ALL_NOTES.index(norm_pitch) if norm_pitch in ALL_NOTES else 0

    total_semitones = note_data.octave * 12 + original_idx + semitones
    new_octave = total_semitones // 12
    new_idx    = total_semitones % 12

    # 옥타브 범위 제한 (0–8)
    new_octave = max(0, min(8, new_octave))
    new_pitch  = note_arr[new_idx]

    return NoteData(
        pitch=new_pitch,
        octave=new_octave,
        duration=note_data.duration,
        measure=note_data.measure,
        beat=note_data.beat,
        type=note_data.type,
        original_pitch=note_data.pitch,
        original_octave=note_data.octave,
    )

def transpose_with_music21(notes: list, semitones: int, use_flats: bool = False) -> list:
    """music21을 사용한 정확한 조 변환"""
    s = stream.Stream()
    for n in notes:
        if n["type"] == "rest":
            r = note.Rest(quarterLength=dur_to_ql(n["duration"]))
            s.append(r)
        else:
            p = pitch.Pitch(f"{n['pitch']}{n['octave']}")
            n_obj = note.Note(p, quarterLength=dur_to_ql(n["duration"]))
            s.append(n_obj)

    transposed = s.transpose(semitones)

    result = []
    orig_data = notes
    for i, element in enumerate(transposed.flat.notesAndRests):
        orig = orig_data[i] if i < len(orig_data) else {}
        if isinstance(element, note.Note):
            p = element.pitch
            pitch_str = p.step
            acc = None
            if p.accidental:
                if p.accidental.name == 'sharp': acc = '#'; pitch_str += '#'
                elif p.accidental.name == 'flat': acc = 'b'; pitch_str += 'b'
            result.append({
                "pitch": pitch_str,
                "octave": p.octave,
                "duration": ql_to_dur(element.quarterLength),
                "measure": orig.get("measure", 1),
                "beat": orig.get("beat", 1),
                "type": "note",
                "accidental": acc,
                "original_pitch": orig.get("pitch"),
                "original_octave": orig.get("octave"),
            })
        elif isinstance(element, note.Rest):
            result.append({
                "pitch": "B",
                "octave": 4,
                "duration": ql_to_dur(element.quarterLength),
                "measure": orig.get("measure", 1),
                "beat": orig.get("beat", 1),
                "type": "rest",
            })
    return result

def dur_to_ql(dur: str) -> float:
    """duration 문자열 → quarterLength"""
    m = {'w': 4, 'whole': 4, '1': 4, 'h': 2, 'half': 2, '2': 2,
         'q': 1, 'quarter': 1, '4': 1, '8': 0.5, 'eighth': 0.5,
         '16': 0.25, 'sixteenth': 0.25}
    return m.get(str(dur), 1.0)

def ql_to_dur(ql: float) -> str:
    m = {4: 'w', 2: 'h', 1: 'q', 0.5: '8', 0.25: '16'}
    return m.get(ql, 'q')

# ═══════════════════════════════════════════════════════════════
#   AI 악보 분석 (GPT-4 Vision)
# ═══════════════════════════════════════════════════════════════

ANALYSIS_PROMPT = """당신은 전문 음악 이론가이자 악보 분석 AI입니다.
첨부된 악보 이미지를 분석하여 아래 JSON 형식으로 정확하게 응답하세요.

응답 형식 (JSON만 반환, 다른 텍스트 없음):
{
  "key": "C",
  "scale": "Major",
  "time_signature": "4/4",
  "tempo": 120,
  "clef": "treble",
  "notes": [
    {
      "pitch": "C",
      "octave": 4,
      "duration": "q",
      "measure": 1,
      "beat": 1.0,
      "type": "note",
      "accidental": null
    }
  ],
  "raw_analysis": "악보에 대한 자세한 설명",
  "confidence": 0.9
}

규칙:
- key: 조성의 루트 음 (C, G, D, A, E, B, F#, F, Bb, Eb, Ab, Db 중 하나)
- scale: "Major" 또는 "Minor"
- duration: "w"(온음), "h"(2분), "q"(4분), "8"(8분), "16"(16분)
- octave: 국제 표준 옥타브 번호 (보통 3-6)
- accidental: null, "#", "b", "n" 중 하나
- type: "note" 또는 "rest"
- confidence: 분석 확신도 (0.0-1.0)"""

async def analyze_with_openai(image_data: bytes, mime_type: str) -> dict:
    """OpenAI GPT-4 Vision으로 악보 분석"""
    if not OPENAI_AVAILABLE:
        return get_demo_data()

    b64 = base64.b64encode(image_data).decode('utf-8')
    data_url = f"data:{mime_type};base64,{b64}"

    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text",  "text": ANALYSIS_PROMPT},
                    {"type": "image_url", "image_url": {"url": data_url, "detail": "high"}},
                ]
            }],
            max_tokens=4000,
            temperature=0.1,
        )
        raw_text = response.choices[0].message.content.strip()
        logger.info(f"OpenAI 응답 수신 ({len(raw_text)} chars)")

        # JSON 파싱
        if "```json" in raw_text:
            raw_text = raw_text.split("```json")[1].split("```")[0].strip()
        elif "```" in raw_text:
            raw_text = raw_text.split("```")[1].split("```")[0].strip()

        parsed = json.loads(raw_text)
        parsed["success"] = True
        parsed["key_display"] = f"{parsed.get('key', 'C')} {parsed.get('scale', 'Major')}"
        return parsed

    except json.JSONDecodeError as e:
        logger.error(f"JSON 파싱 오류: {e}")
        return {**get_demo_data(), "raw_analysis": f"JSON 파싱 오류: {raw_text[:500]}"}
    except Exception as e:
        logger.error(f"OpenAI API 오류: {e}")
        raise

def get_demo_data() -> dict:
    """데모용 기본 데이터"""
    return {
        "success": True,
        "key": "C",
        "scale": "Major",
        "key_display": "C Major",
        "time_signature": "4/4",
        "tempo": 120,
        "clef": "treble",
        "notes": [
            {"pitch": "C", "octave": 4, "duration": "q", "measure": 1, "beat": 1.0, "type": "note"},
            {"pitch": "E", "octave": 4, "duration": "q", "measure": 1, "beat": 2.0, "type": "note"},
            {"pitch": "G", "octave": 4, "duration": "q", "measure": 1, "beat": 3.0, "type": "note"},
            {"pitch": "C", "octave": 5, "duration": "q", "measure": 1, "beat": 4.0, "type": "note"},
            {"pitch": "D", "octave": 5, "duration": "h", "measure": 2, "beat": 1.0, "type": "note"},
            {"pitch": "B", "octave": 4, "duration": "q", "measure": 2, "beat": 3.0, "type": "note"},
            {"pitch": "G", "octave": 4, "duration": "q", "measure": 2, "beat": 4.0, "type": "note"},
            {"pitch": "A", "octave": 4, "duration": "q", "measure": 3, "beat": 1.0, "type": "note"},
            {"pitch": "F", "octave": 4, "duration": "q", "measure": 3, "beat": 2.0, "type": "note"},
            {"pitch": "G", "octave": 4, "duration": "h", "measure": 3, "beat": 3.0, "type": "note"},
            {"pitch": "E", "octave": 4, "duration": "q", "measure": 4, "beat": 1.0, "type": "note"},
            {"pitch": "D", "octave": 4, "duration": "q", "measure": 4, "beat": 2.0, "type": "note"},
            {"pitch": "C", "octave": 4, "duration": "h", "measure": 4, "beat": 3.0, "type": "note"},
        ],
        "raw_analysis": "데모 모드: OpenAI API 키가 없어 샘플 데이터를 사용합니다.\n.env 파일에 OPENAI_API_KEY를 설정하면 실제 AI 분석이 가능합니다.",
        "confidence": 1.0,
    }

# ═══════════════════════════════════════════════════════════════
#   API 엔드포인트
# ═══════════════════════════════════════════════════════════════

@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "openai": OPENAI_AVAILABLE,
        "music21": MUSIC21_AVAILABLE,
        "version": "1.0.0",
    }

@app.post("/analyze")
async def analyze_score(file: UploadFile = File(...)):
    """악보 이미지 AI 분석"""
    logger.info(f"📤 분석 요청: {file.filename} ({file.content_type})")

    # 파일 유효성 검사
    allowed_types = ["image/jpeg", "image/png", "image/webp", "image/gif", "application/pdf"]
    if file.content_type not in allowed_types:
        raise HTTPException(400, f"지원하지 않는 파일 형식: {file.content_type}")

    contents = await file.read()
    if len(contents) > 10 * 1024 * 1024:
        raise HTTPException(400, "파일 크기가 10MB를 초과합니다")

    try:
        result = await analyze_with_openai(contents, file.content_type)
        logger.info(f"✅ 분석 완료: {result.get('key')} {result.get('scale')}, {len(result.get('notes', []))} 음표")
        return JSONResponse(result)
    except Exception as e:
        logger.error(f"❌ 분석 오류: {e}")
        raise HTTPException(500, f"분석 중 오류 발생: {str(e)}")

@app.post("/transpose")
async def transpose_score(req: TransposeRequest):
    """음표 조 변환"""
    logger.info(f"🎵 변환 요청: {req.original_key} → {req.target_key} ({req.semitones:+d} 반음)")

    try:
        notes_list = [n.dict() for n in req.notes]

        if MUSIC21_AVAILABLE:
            transposed_notes = transpose_with_music21(notes_list, req.semitones, req.use_flats)
        else:
            transposed_notes = []
            for n in req.notes:
                transposed = transpose_note_local(n, req.semitones, req.use_flats)
                transposed_notes.append(transposed.dict())

        logger.info(f"✅ 변환 완료: {len(transposed_notes)} 음표")
        return JSONResponse({
            "success": True,
            "original_key": req.original_key,
            "target_key": req.target_key,
            "semitones": req.semitones,
            "notes": transposed_notes,
            "time_signature": req.time_signature,
            "tempo": req.tempo,
            "engine": "music21" if MUSIC21_AVAILABLE else "local",
        })
    except Exception as e:
        logger.error(f"❌ 변환 오류: {e}")
        raise HTTPException(500, f"변환 중 오류 발생: {str(e)}")

@app.get("/keys")
async def get_keys():
    """지원 조성 목록"""
    return {
        "major": ['C', 'G', 'D', 'A', 'E', 'B', 'F#', 'F', 'Bb', 'Eb', 'Ab', 'Db'],
        "minor": ['A', 'E', 'B', 'F#', 'C#', 'G#', 'D', 'G', 'C', 'F', 'Bb', 'Eb'],
    }

# ═══════════════════════════════════════════════════════════════
#   실행
# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    logger.info("🚀 ScoreShift 백엔드 서버 시작...")
    logger.info(f"   OpenAI API: {'✅ 사용 가능' if OPENAI_AVAILABLE else '❌ API 키 없음'}")
    logger.info(f"   music21:    {'✅ 사용 가능' if MUSIC21_AVAILABLE else '❌ 미설치'}")
    logger.info("   http://localhost:8000 에서 실행 중")
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
