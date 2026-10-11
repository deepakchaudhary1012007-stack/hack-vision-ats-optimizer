import io
import json
import os
import requests
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pypdf import PdfReader

# Google Auth Token
AUTH_TOKEN = os.environ.get("GEMINI_API_KEY", "AQ.Ab8RN6Lw3lumSQ4sEwfuTMJ7QXcgi8dCAgrgwK-alEQGcd6HVA")

app = FastAPI(title="AI ATS Resume Optimizer")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

EVALUATION_HISTORY = []

def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        text = ""
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
        return text.strip()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"PDF read error: {str(e)}")

@app.get("/", response_class=HTMLResponse)
def serve_home():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.post("/api/analyze")
async def analyze_resume(
    resume_file: UploadFile = File(...),
    job_description: str = Form("Software Engineer / Full Stack Developer"),
):
    contents = await resume_file.read()
    if resume_file.filename.endswith(".pdf"):
        resume_text = extract_text_from_pdf(contents)
    else:
        resume_text = contents.decode("utf-8", errors="ignore")

    if not resume_text or len(resume_text.strip()) < 20:
        raise HTTPException(status_code=400, detail="Resume empty or text not readable.")

    prompt = f"""You are an ATS Resume Analyzer. Analyze the following resume against the target job.
Return ONLY valid raw JSON with this exact schema:
{{
  "ats_score": 85,
  "verdict": "Good Fit",
  "matched_keywords": ["Python", "FastAPI"],
  "missing_keywords": ["Docker"],
  "section_scores": {{"skills": 80, "experience": 85, "education": 90}},
  "strengths": ["Clear project structure"],
  "weaknesses": ["Add metrics and impact numbers"],
  "tailored_suggestions": ["Include quantifiable results"]
}}

TARGET JOB:
{job_description}

RESUME:
{resume_text}
"""

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {AUTH_TOKEN}"
    }

    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"response_mime_type": "application/json"}
    }

    try:
        res = requests.post(url, headers=headers, json=payload, timeout=60)
        
        # Agar Bearer se bhi reject hua to query param try karega
        if res.status_code != 200:
            fallback_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={AUTH_TOKEN}"
            res = requests.post(fallback_url, json=payload, timeout=60)

        if res.status_code != 200:
            raise HTTPException(status_code=res.status_code, detail=res.text)

        data = res.json()
        raw_text = data["candidates"][0]["content"]["parts"][0]["text"]
        result = json.loads(raw_text)
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI Error: {str(e)}")