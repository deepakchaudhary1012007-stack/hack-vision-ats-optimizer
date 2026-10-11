import io
import json
import os
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from google import genai
from google.genai import types
from pypdf import PdfReader

API_KEY = os.environ.get("GEMINI_API_KEY", "AQ.Ab8RN6Lw3lumSQ4sEwfuTMJ7QXcgi8dCAgrgwK-alEQGcd6HVA")
client = genai.Client(api_key=API_KEY)

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
    raise HTTPException(
        status_code=400, detail="Resume empty or text not readable."
    )

  prompt = f"TARGET JOB:\n{job_description}\n\nRESUME:\n{resume_text}"
  try:
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema={
                "type": "OBJECT",
                "properties": {
                    "ats_score": {"type": "NUMBER"},
                    "verdict": {"type": "STRING"},
                    "matched_keywords": {
                        "type": "ARRAY",
                        "items": {"type": "STRING"},
                    },
                    "missing_keywords": {
                        "type": "ARRAY",
                        "items": {"type": "STRING"},
                    },
                    "section_scores": {
                        "type": "OBJECT",
                        "properties": {
                            "skills": {"type": "NUMBER"},
                            "experience": {"type": "NUMBER"},
                            "projects": {"type": "NUMBER"},
                            "formatting": {"type": "NUMBER"},
                        },
                        "required": [
                            "skills",
                            "experience",
                            "projects",
                            "formatting",
                        ],
                    },
                    "actionable_fixes": {
                        "type": "ARRAY",
                        "items": {"type": "STRING"},
                    },
                },
                "required": [
                    "ats_score",
                    "verdict",
                    "matched_keywords",
                    "missing_keywords",
                    "section_scores",
                    "actionable_fixes",
                ],
            },
        ),
    )

    result_data = json.loads(response.text)
    EVALUATION_HISTORY.append({
        "filename": resume_file.filename,
        "score": result_data.get("ats_score", 0),
        "target_job": job_description[:30],
        "verdict": result_data.get("verdict", "N/A"),
    })
    return result_data
  except Exception as e:
    raise HTTPException(status_code=500, detail=f"AI Error: {str(e)}")


@app.get("/api/admin/stats")
def get_admin_stats():
  total = len(EVALUATION_HISTORY)
  if total == 0:
    return {"total_scans": 0, "average_score": 0, "history": []}
  avg_score = sum([item["score"] for item in EVALUATION_HISTORY]) / total
  return {
      "total_scans": total,
      "average_score": round(avg_score, 1),
      "history": EVALUATION_HISTORY[-10:],
  }


if __name__ == "__main__":
  import uvicorn

  uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)