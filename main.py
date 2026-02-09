import os
import uuid
import json
import re
import cv2
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import easyocr
import fitz # PyMuPDF
import numpy as np
from pydantic import BaseModel, RootModel


# Gemini
from google import genai


# ================================
# CONFIG
# ================================

API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise RuntimeError("Set GEMINI_API_KEY in .env")

client = genai.Client(api_key=API_KEY)

# Stable + fast model
MODEL_NAME = "models/gemini-2.5-flash"

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


# ================================
# INIT FASTAPI
# ================================

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load EasyOCR ONCE
reader = easyocr.Reader(['en','hi'], gpu=False)

# ===== OCR WARMUP (ADD HERE) =====
@app.on_event("startup")
def warmup():
    reader.readtext(np.zeros((50, 50), dtype=np.uint8))


# ================================
# OCR FUNCTION
# ================================
def extract_text(image_path: str) -> str:
    ext = image_path.lower()

    # ================= PDF =================
    if ext.endswith(".pdf"):

        text = ""
        doc = fitz.open(image_path)

        # Try native text extraction first
        for page in doc:
            text += page.get_text()

        doc.close()

        if text.strip():
            return text

        # ===== OCR FALLBACK =====
        images_text = ""
        doc = fitz.open(image_path)

        for page in doc:

            pix = page.get_pixmap(dpi=400)

            img = np.frombuffer(pix.samples, dtype=np.uint8)
            img = img.reshape(pix.h, pix.w, pix.n)

            # Preprocess
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.medianBlur(img, 3)

            img = cv2.adaptiveThreshold(
                img,
                255,
                cv2.ADAPTIVE_THRESH_MEAN_C,
                cv2.THRESH_BINARY,
                31,
                15
            )

            results = reader.readtext(
                img,
                detail=0,
                paragraph=True
            )

            images_text += "\n".join(results)


        doc.close()
        return images_text

# ================= IMAGE =================
    else:
        img = cv2.imread(image_path)

        if img is None:
            raise RuntimeError("Failed to read image")

    # ===== PREPROCESS (ADD HERE) =====
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.medianBlur(img, 3)

        img = cv2.adaptiveThreshold(
            img,
            255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY,
            31,
            15
        )

        # OCR
        results = reader.readtext(
            img,
            detail=0,
            paragraph=True
        )

        return "\n".join(results)


# ================================
# GEMINI SAFE CALL
# ================================
def call_gemini(prompt: str):
    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
            timeout=20
        )   


        raw = ""

        # SAFER EXTRACTION
        if hasattr(response, "text") and response.text:
            raw = response.text
        elif response.candidates:
            raw = response.candidates[0].content.parts[0].text
        else:
            raw = ""

        raw = raw.strip()

        # Remove markdown fences
        raw = re.sub(r"```json", "", raw)
        raw = re.sub(r"```", "", raw)

        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            return {
                "form_type": "Unknown",
                "summary": raw[:300],
                "fields": []
            }

        data = json.loads(match.group(0))

        if not isinstance(data, dict):
            data = {}

        data.setdefault("form_type", "Unknown Form")
        data.setdefault("summary", "")
        data.setdefault("fields", [])

        return data

    except Exception as e:
        return {
            "form_type": "Unknown",
            "summary": "Gemini parsing failed",
            "fields": [],
            "error": str(e)
        }

# ================================
# MAIN ENDPOINT
# ================================

@app.post("/analyze-form")
async def analyze_form(file: UploadFile = File(...), mode: str = "explain"):
    try:
        # Extension
        ext = os.path.splitext(file.filename)[1].lower()

        # Validate type FIRST
        if ext not in [".jpg", ".jpeg", ".png", ".pdf"]:
            raise HTTPException(status_code=400, detail="Unsupported file type")

        filename = f"{uuid.uuid4()}{ext}"
        path = os.path.join(UPLOAD_DIR, filename)

        # Size check
        MAX_SIZE = 8 * 1024 * 1024
        contents = await file.read()

        if len(contents) > MAX_SIZE:
            raise HTTPException(413, "File too large")

        # Save file
        with open(path, "wb") as f:
            f.write(contents)

        # OCR
        try:
            extracted_text = extract_text(path)
        except Exception:
            extracted_text = ""


        if len(extracted_text.strip()) < 5:
            return {
                "form_type": "Unknown",
                "summary": "Form unreadable",
                "fields": []
            }


        # Clean noise before truncation
        extracted_text = re.sub(r"\s+", " ", extracted_text)
        extracted_text = extracted_text[:12000]


        # Prompt
        prompt = f"""
You are an expert government form parser.

Extract ALL fillable fields from this form text.

TEXT:
----------------
{extracted_text}
----------------

Return STRICT JSON ONLY:
{{
 "form_type": "...",
 "summary": "...",
 "confidence": 0-100,
 "fields": [
   {{
     "name": "...",
     "meaning": "...",
     "how_to_fill": "..."
   }}
 ]
}}
"""


        analysis = call_gemini(prompt)
        analysis.setdefault("confidence", 80)
        return analysis


    finally:
        if 'path' in locals() and os.path.exists(path):
            os.remove(path)
# ================================ 
# =========================================================
# EXTRA ENDPOINTS FOR FRONTEND FEATURES
# =========================================================

# -------- TRANSLATION --------
class TranslateRequest(BaseModel):
    text: list[str]

@app.post("/translate")
def translate(req: TranslateRequest):

    joined = "\n".join(req.text)

    prompt = f"""
Translate the following text into Hindi.

Return STRICT JSON ONLY:
{{
 "translated":[ "...", "..."]
}}

TEXT:
{joined}
"""

    result = call_gemini(prompt)

    if "translated" in result:
        return result

    lines = result.get("summary", "").split("\n")
    return {"translated": lines}


# -------- FILLED SAMPLE --------
class SampleRequest(RootModel):
    root: dict

@app.post("/generate-sample")
def generate_sample(req: SampleRequest):

    answers = json.dumps(req.__root__, indent=2)

    prompt = f"""
Create a realistic example filled government form
using the user answers below.

Return plain formatted text.

ANSWERS:
{answers}
"""

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
        timeout=20
    )

    return {"sample": response.text or "AI could not generate sample"}


# -------- LETTER GENERATOR --------
class LetterRequest(RootModel):
    root: dict

@app.post("/letter")
def generate_letter(req: LetterRequest):

    answers = json.dumps(req.root, indent=2)

    prompt = f"""
Write a professional government correction letter
based on these form details.

Keep it formal and concise.

DETAILS:
{answers}
"""

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
        timeout=20
    )

    return {"letter": response.text or "AI could not generate letter"}
@app.get("/")
def health():
    return {"status":"running"}
