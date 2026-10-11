import io
import json
import os
import re
import requests
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pypdf import PdfReader

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "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")

app = FastAPI(title="AI ATS Resume Optimizer")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

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
        return ""

@app.get("/", response_class=HTMLResponse)
def serve_home():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()

# SAFETY FUNCTION: Ye assure karega ki AI chahe jaisa data de, frontend hamesha list hi receive karega
def sanitize_list(data, key):
    val = data.get(key)
    if val is None:
        return ["Not specified"]
    if isinstance(val, str):
        return [val]
    if isinstance(val, list):
        return val if len(val) > 0 else ["None found"]
    return ["Not specified"]

@app.post("/api/analyze")
async def analyze_resume(
    resume_file: UploadFile = File(...),
    job_description: str = Form("Software Engineer / Data Science"),
):
    try:
        contents = await resume_file.read()
        if resume_file.filename.endswith(".pdf"):
            resume_text = extract_text_from_pdf(contents)
        else:
            resume_text = contents.decode("utf-8", errors="ignore")

        system_prompt = (
            "You are an expert ATS Resume Analyzer. Compare the resume to the job description. "
            "You MUST respond ONLY with a VALID JSON object exactly matching this schema. No extra text:\n"
            "{\n"
            '  "ats_score": 85,\n'
            '  "verdict": "Good Fit",\n'
            '  "matched_keywords": ["Skill1"],\n'
            '  "missing_keywords": ["Skill2"],\n'
            '  "section_scores": {"skills": 80, "experience": 85, "education": 90},\n'
            '  "strengths": ["Point 1"],\n'
            '  "weaknesses": ["Point 1"],\n'
            '  "tailored_suggestions": ["Point 1"]\n'
            "}"
        )

        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": "openai/gpt-oss-120b",
            "messages": [
                {"role": "system", "content": system_prompt},
                # Extra lambe text se crash rokne ke liye limit lagayi hai
                {"role": "user", "content": f"TARGET JOB:\n{job_description}\n\nRESUME:\n{resume_text[:4000]}"}
            ],
            "temperature": 0.2
        }

        res = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload, timeout=60)
        
        if res.status_code != 200:
            raise Exception(f"API Error: {res.text}")

        # AI ke result se markdown/kachra saaf karna
        raw_text = res.json()["choices"][0]["message"]["content"].strip()
        raw_text = re.sub(r'```json\s*', '', raw_text)
        raw_text = re.sub(r'```\s*', '', raw_text)
        
        start_idx = raw_text.find("{")
        end_idx = raw_text.rfind("}")
        if start_idx != -1 and end_idx != -1:
            raw_text = raw_text[start_idx:end_idx+1]
            
        ai_data = json.loads(raw_text)

        # Hamesha safe aur clean data frontend ko bhejna
        return {
            "ats_score": int(ai_data.get("ats_score") or 75),
            "verdict": str(ai_data.get("verdict") or "Evaluation Complete"),
            "matched_keywords": sanitize_list(ai_data, "matched_keywords"),
            "missing_keywords": sanitize_list(ai_data, "missing_keywords"),
            "section_scores": ai_data.get("section_scores") or {"skills": 75, "experience": 75, "education": 75},
            "strengths": sanitize_list(ai_data, "strengths"),
            "weaknesses": sanitize_list(ai_data, "weaknesses"),
            "tailored_suggestions": sanitize_list(ai_data, "tailored_suggestions")
        }

    except Exception as e:
        # Agar kuch bhi fatega, toh code crash nahi hoga, ye screen par error dikhayega safely
        return {
            "ats_score": 0,
            "verdict": "Error processing response",
            "matched_keywords": ["None"],
            "missing_keywords": ["None"],
            "section_scores": {"skills": 0, "experience": 0, "education": 0},
            "strengths": ["None"],
            "weaknesses": [f"Backend Error: {str(e)}"],
            "tailored_suggestions": ["Please try again"]
        }