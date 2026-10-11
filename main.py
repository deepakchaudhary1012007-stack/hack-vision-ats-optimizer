import io
import json
import os
import requests
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pypdf import PdfReader

# Groq API Key
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "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")

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

    system_prompt = (
        "You are an ATS Resume Analyzer. Analyze the resume against the target job. "
        "Return ONLY a valid JSON object matching this schema without any markdown formatting:\n"
        "{\n"
        '  "ats_score": 85,\n'
        '  "verdict": "Good Fit",\n'
        '  "matched_keywords": ["Python", "FastAPI"],\n'
        '  "missing_keywords": ["Docker"],\n'
        '  "section_scores": {"skills": 80, "experience": 85, "education": 90},\n'
        '  "strengths": ["Clear project structure"],\n'
        '  "weaknesses": ["Add metrics and impact numbers"],\n'
        '  "tailored_suggestions": ["Include quantifiable results"]\n'
        "}"
    )

    user_prompt = f"TARGET JOB:\n{job_description}\n\nRESUME:\n{resume_text}"

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
       "model": "llama3-8b-8192",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "response_format": {"type": "json_object"}
    }

    try:
        res = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload, timeout=60)
        if res.status_code != 200:
            raise HTTPException(status_code=res.status_code, detail=res.text)

        data = res.json()
        raw_text = data["choices"][0]["message"]["content"]
        result = json.loads(raw_text)
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI Error: {str(e)}")