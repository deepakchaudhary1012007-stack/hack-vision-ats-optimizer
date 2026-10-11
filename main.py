import io
import json
import os
import re
import requests
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pypdf import PdfReader

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

app = FastAPI(title="AI ATS Resume Optimizer")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Admin tab ke liye simple storage
stats = {"total_scans": 0, "total_score": 0}

def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        return "\n".join(page.extract_text() for page in reader.pages if page.extract_text()).strip()
    except Exception:
        return ""

def sanitize_list(data, key):
    val = data.get(key)
    if isinstance(val, list) and len(val) > 0:
        return val
    if isinstance(val, str):
        return [val]
    return ["Not specified"]

@app.get("/", response_class=HTMLResponse)
def serve_home():
    try:
        with open("index.html", "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return "<h1>index.html not found</h1>"

# Naya Admin route jo index.html maang raha tha
@app.get("/api/admin/stats")
def get_stats():
    avg = round(stats["total_score"] / stats["total_scans"]) if stats["total_scans"] > 0 else 0
    return {"total_scans": stats["total_scans"], "average_score": avg}

@app.post("/api/analyze")
async def analyze_resume(request: Request):
    try:
        if not GROQ_API_KEY:
            raise ValueError("API Key is missing in Render Environment Variables.")

        form_data = await request.form()
        resume_bytes = b""
        filename = ""
        job_description = "Software Engineer"
        
        for key, value in form_data.items():
            if hasattr(value, "filename") and value.filename:
                resume_bytes = await value.read()
                filename = value.filename
            elif isinstance(value, str) and len(value) > 2:
                job_description = value

        if not resume_bytes:
            raise ValueError("Resume file upload failed or missing.")

        if filename.lower().endswith(".pdf"):
            resume_text = extract_text_from_pdf(resume_bytes)
        else:
            resume_text = resume_bytes.decode("utf-8", errors="ignore")

        # EXACT Match to index.html variables (actionable_fixes, projects, formatting)
        system_prompt = (
            "You are an ATS Resume Analyzer. Compare the resume to the job description. "
            "You MUST respond ONLY with a VALID JSON object exactly matching this schema. No markdown:\n"
            "{\n"
            '  "ats_score": 85,\n'
            '  "verdict": "Good Fit",\n'
            '  "matched_keywords": ["Skill1"],\n'
            '  "missing_keywords": ["Skill2"],\n'
            '  "section_scores": {"skills": 80, "experience": 85, "projects": 90, "formatting": 85},\n'
            '  "actionable_fixes": ["Point 1", "Point 2"]\n'
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
                {"role": "user", "content": f"JOB:\n{job_description}\n\nRESUME:\n{resume_text[:4000]}"}
            ],
            "temperature": 0.2
        }

        res = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload, timeout=60)
        
        if res.status_code != 200:
            raise ValueError(f"Groq API Error: {res.text}")

        raw_text = res.json()["choices"][0]["message"]["content"].strip()
        raw_text = re.sub(r'```json\s*', '', raw_text)
        raw_text = re.sub(r'```\s*', '', raw_text)
        
        start_idx = raw_text.find("{")
        end_idx = raw_text.rfind("}")
        if start_idx != -1 and end_idx != -1:
            ai_data = json.loads(raw_text[start_idx:end_idx+1])
        else:
            raise ValueError("AI did not return valid JSON format.")

        final_score = int(ai_data.get("ats_score", 70))
        
        # Admin stats update ho rahe hain
        stats["total_scans"] += 1
        stats["total_score"] += final_score

        return {
            "ats_score": final_score,
            "verdict": str(ai_data.get("verdict", "Evaluated successfully")),
            "matched_keywords": sanitize_list(ai_data, "matched_keywords"),
            "missing_keywords": sanitize_list(ai_data, "missing_keywords"),
            "section_scores": ai_data.get("section_scores", {"skills": 70, "experience": 70, "projects": 70, "formatting": 70}),
            "actionable_fixes": sanitize_list(ai_data, "actionable_fixes")
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))