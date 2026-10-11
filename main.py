import io
import json
import os
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

        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": "openai/gpt-oss-120b",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"TARGET JOB:\n{job_description}\n\nRESUME:\n{resume_text}"}
            ]
        }

        res = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload, timeout=60)
        
        if res.status_code != 200:
            raise Exception(f"API Error {res.status_code}: {res.text}")

        data = res.json()
        raw_text = data["choices"][0]["message"]["content"]
        
        # Extract JSON safely
        start_idx = raw_text.find("{")
        end_idx = raw_text.rfind("}")
        if start_idx == -1 or end_idx == -1:
            raise Exception("AI did not return valid JSON")
            
        result = json.loads(raw_text[start_idx:end_idx+1])
        
        # Guarantee arrays exist for frontend mapping
        return {
            "ats_score": result.get("ats_score", 70),
            "verdict": result.get("verdict", "Evaluated"),
            "matched_keywords": result.get("matched_keywords", ["Found basic skills"]),
            "missing_keywords": result.get("missing_keywords", ["Review JD"]),
            "section_scores": result.get("section_scores", {"skills": 70, "experience": 70, "education": 70}),
            "strengths": result.get("strengths", ["Resume parsed"]),
            "weaknesses": result.get("weaknesses", ["Update format"]),
            "tailored_suggestions": result.get("tailored_suggestions", ["Tailor resume better"])
        }

    except Exception as e:
        # MAGIC SHIELD: Return formatted dummy data with the EXACT error message so frontend NEVER crashes
        return {
            "ats_score": 0,
            "verdict": "Check Weaknesses Section for Error",
            "matched_keywords": ["None"],
            "missing_keywords": ["None"],
            "section_scores": {"skills": 0, "experience": 0, "education": 0},
            "strengths": ["None"],
            "weaknesses": [f"BACKEND ERROR: {str(e)}"],
            "tailored_suggestions": ["Please fix the backend issue shown above"]
        }