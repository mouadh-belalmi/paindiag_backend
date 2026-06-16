"""
PainDiag+ FastAPI Backend
Run: uvicorn main:app --host 0.0.0.0 --port 8000 --reload
"""

import hashlib
import json
import logging
import os
import random
import re
import sqlite3
import string
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Optional

import google.genai as genai
from dotenv import load_dotenv
import secrets as _secrets

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel, Field

# ── Environment ───────────────────────────────────────────────────────────────

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "paindiag.db")
LOG_PATH = os.path.join(BASE_DIR, "logs", "app.log")

os.makedirs(os.path.join(BASE_DIR, "logs"), exist_ok=True)

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH, encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("paindiag")

# ── Gemini ────────────────────────────────────────────────────────────────────

GEMINI_KEY = os.getenv("GEMINI_API_KEY", "")
gemini_client = genai.Client(api_key=GEMINI_KEY) if GEMINI_KEY else None

# ── DB helper ─────────────────────────────────────────────────────────────────

def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def db_scalar(sql: str, params: tuple = ()) -> Any:
    with get_conn() as conn:
        row = conn.execute(sql, params).fetchone()
        return row[0] if row else None


# ── NRS → Chancellerie ────────────────────────────────────────────────────────

def nrs_to_chancellerie(nrs: float) -> str:
    if nrs <= 2:
        return "1/7 - tres leger"
    if nrs <= 3:
        return "2/7 - leger"
    if nrs <= 5:
        return "3/7 - modere"
    if nrs <= 6:
        return "4/7 - moyen"
    if nrs <= 7:
        return "5/7 - assez important"
    if nrs <= 9:
        return "6/7 - important"
    return "7/7 - tres important"


# ── Record number ──────────────────────────────────────────────────────────────

def generate_record_number() -> str:
    date_part = datetime.now().strftime("%Y%m%d")
    suffix = "".join(random.choices(string.digits, k=4))
    candidate = f"REC-{date_part}-{suffix}"
    # Ensure uniqueness against the DB
    while db_scalar("SELECT 1 FROM pain_records WHERE record_number = ?", (candidate,)):
        suffix = "".join(random.choices(string.digits, k=4))
        candidate = f"REC-{date_part}-{suffix}"
    return candidate


# ── SHA-256 hash ──────────────────────────────────────────────────────────────

def compute_hash(patient_name: str, record_number: str, pain_description: str, created_at: str) -> str:
    raw = f"{patient_name}{record_number}{pain_description}{created_at}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


# ── SQLite symptom/disease matching ──────────────────────────────────────────

def sqlite_symptom_match(text: str) -> list[str]:
    """Return symptom name_en values whose names appear (loosely) in text."""
    text_lower = text.lower().replace("_", " ")
    with get_conn() as conn:
        rows = conn.execute("SELECT name_en FROM symptoms").fetchall()
    matched = []
    for row in rows:
        sym = row["name_en"].replace("_", " ").lower()
        if sym in text_lower:
            matched.append(row["name_en"])
    return matched


def sqlite_disease_match(symptom_names: list[str], limit: int = 5) -> list[dict]:
    """
    Given symptom name_en list, score diseases by how many symptoms overlap,
    normalised by total symptoms of that disease.
    """
    if not symptom_names:
        return []

    placeholders = ",".join("?" * len(symptom_names))
    sql = f"""
        SELECT
            d.name_en,
            d.description_en,
            d.precautions,
            COUNT(DISTINCT ds.symptom_id)                     AS matched,
            COUNT(DISTINCT ds2.symptom_id)                    AS total
        FROM diseases d
        JOIN disease_symptom ds  ON ds.disease_id = d.id
        JOIN symptoms s          ON s.id = ds.symptom_id
                                 AND s.name_en IN ({placeholders})
        JOIN disease_symptom ds2 ON ds2.disease_id = d.id
        GROUP BY d.id
        ORDER BY matched DESC, total ASC
        LIMIT {limit}
    """
    with get_conn() as conn:
        rows = conn.execute(sql, symptom_names).fetchall()

    results = []
    for row in rows:
        total = max(row["total"], 1)
        confidence = round(row["matched"] / total * 100)
        results.append({
            "name": row["name_en"],
            "confidence": confidence,
            "name_ar": "",
            "description": row["description_en"] or "",
            "precautions": row["precautions"] or "",
        })
    return results


# ── Gemini diagnosis ──────────────────────────────────────────────────────────

GEMINI_PROMPT = """\
Tu es un assistant médical IA pour PainDiag+, un système de documentation \
légale de la douleur utilisé en Algérie par les tribunaux et les compagnies \
d'assurance (CNAS/CASNOS).

Profil du patient :
- Âge : {age} ans
- Sexe : {gender_fr}
- Localisation de la douleur : {body_part_fr}
- Intensité de la douleur (ENS) : {nrs_score}/10
- Description de la douleur : {description}
- Antécédents médicaux : {medical_history}

Tâche :
1. DIAGNOSTIC DIFFÉRENTIEL : Les 3 affections les plus probables avec un \
pourcentage de confiance. Tenir compte de l'âge, du sexe, de la localisation \
et des antécédents médicaux. Pour les femmes : considérer les différences de \
présentation hormonale et cardiaque. Pour douleur à la tête + ENS élevé + \
nausées : toujours évoquer AVC/AIT.

2. SYMPTÔMES EXTRAITS : Principaux symptômes médicaux mentionnés ou implicites.

3. TRIAGE :
   EMERGENCY (ENS 8-10 ou drapeaux rouges) → urgences immédiates
   URGENT (ENS 5-7) → médecin dans les 24h
   MODERATE (ENS 3-4) → médecin dans la semaine
   MILD (ENS 1-2) → auto-soin

4. Recommandation bilingue (arabe + français).

Retourner UNIQUEMENT ce JSON exact, sans aucun autre texte :
{{
  "diagnoses": [
    {{"name": "diagnostic le plus probable en français", "confidence": <entier_1_100_selon_probabilité_clinique>, "name_ar": "التشخيص بالعربية"}},
    {{"name": "deuxième diagnostic probable en français", "confidence": <entier_1_100_selon_probabilité_clinique>, "name_ar": "التشخيص بالعربية"}},
    {{"name": "troisième diagnostic à exclure en français", "confidence": <entier_1_100_selon_probabilité_clinique>, "name_ar": "التشخيص بالعربية"}}
  ],
  "extracted_symptoms": ["symptôme1", "symptôme2"],
  "triage_level": "urgent",
  "recommendation_ar": "...",
  "recommendation_fr": "...",
  "red_flags": true
}}

RÈGLES ABSOLUES :
- Ne jamais inventer de symptômes non mentionnés ou non implicites
- Toujours tenir compte de l'âge et du sexe dans les 3 diagnostics différentiels
- Utiliser "confidence" (pas "probability") — le nom du champ doit correspondre exactement
- triage_level doit être exactement l'un de : emergency, urgent, moderate, mild
- Tous les noms de diagnostics (champ "name") doivent être en français
- Les symptômes extraits (extracted_symptoms) doivent être en français
- recommendation_ar en arabe, recommendation_fr en français
- La confidence de chaque diagnostic EST CALCULÉE PAR TOI selon la probabilité \
clinique réelle — ce n'est PAS un exemple fixe. Le premier diagnostic peut avoir \
92%, le deuxième 45%, le troisième 20% — selon ce que tu juges cliniquement.
- Les trois confidences doivent être DIFFÉRENTES et refléter la réalité clinique.
- Ne jamais répéter les valeurs 85, 60, 40 — ce sont des exemples à NE PAS copier.
"""

# Translation tables for gender and body part
_GENDER_FR = {
    "male": "homme",
    "female": "femme",
    "unknown": "non précisé",
}

_BODY_PART_FR = {
    "head": "tête",
    "neck": "cou",
    "chest": "poitrine",
    "abdomen": "abdomen",
    "back": "dos",
    "lower_back": "bas du dos",
    "shoulder": "épaule",
    "arm": "bras",
    "elbow": "coude",
    "wrist": "poignet",
    "hand": "main",
    "hip": "hanche",
    "knee": "genou",
    "ankle": "cheville",
    "foot": "pied",
    "leg": "jambe",
    "thigh": "cuisse",
    "groin": "aine",
    "pelvis": "pelvis",
    "general": "général",
    "spine": "colonne vertébrale",
    "ribs": "côtes",
    "throat": "gorge",
    "face": "visage",
    "jaw": "mâchoire",
    "ear": "oreille",
    "eye": "œil",
    "nose": "nez",
}

FALLBACK_RESULT = {
    "extracted_symptoms": [],
    "diagnoses": [],
    "triage_level": "normal",
    "recommendation_ar": "يرجى مراجعة طبيب مختص",
    "recommendation_fr": "Veuillez consulter un medecin specialise",
    "red_flags": False,
}


def call_gemini(pain_description: str, body_region: str, pain_scale_nrs: float,
                medical_history: str, uploaded_docs_summary: str,
                gender: str = "unknown", date_of_birth: str = "",
                patient_age: int = 0) -> dict:
    if not gemini_client:
        raise RuntimeError("Gemini not configured")

    gender_fr = _GENDER_FR.get(gender, gender)
    body_part_fr = _BODY_PART_FR.get(body_region, body_region.replace("_", " "))
    prompt = GEMINI_PROMPT.format(
        age=patient_age,
        gender_fr=gender_fr,
        body_part_fr=body_part_fr,
        nrs_score=pain_scale_nrs,
        description=pain_description,
        medical_history=medical_history or "aucun",
    )
    response = gemini_client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
    )
    raw = response.text.strip()

    # Strip markdown code fences if present
    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
    raw = re.sub(r"\s*```$", "", raw)

    return json.loads(raw)


# Map Gemini triage levels to Flutter AppConstants values
_TRIAGE_MAP = {
    "emergency": "emergency",
    "urgent": "consult_24h",
    "moderate": "monitor",
    "mild": "safe",
    "normal": "safe",
}


def get_ai_diagnosis(pain_description: str, body_region: str, pain_scale_nrs: float,
                     medical_history: str, uploaded_docs_summary: str,
                     gender: str = "unknown", date_of_birth: str = "",
                     patient_age: int = 0) -> dict:
    """Try Gemini; fall back to SQLite matching on any failure."""
    try:
        result = call_gemini(pain_description, body_region, pain_scale_nrs,
                             medical_history, uploaded_docs_summary,
                             gender=gender, date_of_birth=date_of_birth,
                             patient_age=patient_age)
        # Map triage level to Flutter-expected constant
        if "triage_level" in result:
            result["triage_level"] = _TRIAGE_MAP.get(
                result["triage_level"], result["triage_level"]
            )
        # Enrich with SQLite confidence scores if Gemini returned symptoms
        symptoms = result.get("extracted_symptoms") or []
        if symptoms and not result.get("diagnoses"):
            result["diagnoses"] = sqlite_disease_match(symptoms)
        return result
    except Exception as exc:
        log.warning("Gemini fallback triggered: %s", exc)

    # Pure SQLite fallback
    symptoms = sqlite_symptom_match(pain_description)
    diseases = sqlite_disease_match(symptoms)
    return {
        **FALLBACK_RESULT,
        "extracted_symptoms": symptoms,
        "diagnoses": diseases,
    }


# ── Pydantic models ───────────────────────────────────────────────────────────

class CreateRecordRequest(BaseModel):
    patient_name: str
    patient_age: int
    pain_description: str
    body_region: str
    pain_scale_nrs: float = Field(default=5.0, ge=0, le=10)
    gender: str = "unknown"
    date_of_birth: Optional[str] = None
    medical_history: Optional[str] = None
    uploaded_docs_summary: Optional[str] = None


class RecordResponse(BaseModel):
    success: bool = True
    record_number: str
    patient_name: str
    patient_age: int
    gender: str = 'unknown'
    body_region: str = ""
    date_of_birth: Optional[str] = None
    pain_description: str = ""
    medical_history: str = ""
    pain_scale_nrs: float
    pain_scale_chancellerie: str
    extracted_symptoms: list
    diagnoses: list
    triage_level: str
    recommendation_ar: str
    recommendation_fr: str
    red_flags: bool
    record_hash: str
    created_at: str
    hash_verified: Optional[bool] = None
    expert_view: Optional[bool] = None


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Create experts table if it doesn't exist
    with get_conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS experts (
                id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                full_name           TEXT NOT NULL,
                accreditation_number TEXT UNIQUE NOT NULL,
                expert_type         TEXT NOT NULL,
                email               TEXT UNIQUE NOT NULL,
                password_hash       TEXT NOT NULL,
                status              TEXT DEFAULT 'pending',
                created_at          TEXT,
                approved_at         TEXT
            )
        """)

    # Migrate existing DB: add new columns if they don't exist yet
    conn = get_conn()
    for col in ['gender', 'date_of_birth', 'body_region']:
        try:
            conn.execute(f"ALTER TABLE pain_records ADD COLUMN {col} TEXT DEFAULT ''")
            conn.commit()
            print(f"Added column: {col}")
        except Exception as e:
            print(f"Column {col} already exists: {e}")
    conn.close()

    count = db_scalar("SELECT COUNT(*) FROM pain_records") or 0
    db_mb = os.path.getsize(DB_PATH) / (1024 * 1024) if os.path.exists(DB_PATH) else 0
    key_status = "SET" if gemini_client else "NOT SET (SQLite fallback active)"
    print("\nPainDiag+ Backend running on http://0.0.0.0:8000")
    print(f"  DB records  : {count}")
    print(f"  DB size     : {db_mb:.2f} MB")
    print(f"  Gemini key  : {key_status}\n")
    log.info("PainDiag+ startup — %d records, Gemini key %s", count, key_status)
    yield


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(title="PainDiag+ API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.post("/api/records/create", response_model=RecordResponse)
async def create_record(req: CreateRecordRequest):
    print(f"BACKEND RECEIVED: pain_scale_nrs={req.pain_scale_nrs}, body_region={req.body_region}, patient_name={req.patient_name}, patient_age={req.patient_age}")
    print(f"SAVING RECORD: gender={req.gender}, body_region={req.body_region}")
    try:
        record_number = generate_record_number()
        created_at = datetime.now().isoformat(timespec="seconds")
        pain_scale_chancellerie = nrs_to_chancellerie(req.pain_scale_nrs)
        record_hash = compute_hash(req.patient_name, record_number, req.pain_description, created_at)

        ai = get_ai_diagnosis(
            req.pain_description,
            req.body_region,
            req.pain_scale_nrs,
            req.medical_history or "",
            req.uploaded_docs_summary or "",
            gender=req.gender,
            date_of_birth=req.date_of_birth or "",
            patient_age=req.patient_age,
        )

        extracted_symptoms = ai.get("extracted_symptoms", [])
        diagnoses = ai.get("diagnoses", [])
        triage_level = ai.get("triage_level", "normal")
        recommendation_ar = ai.get("recommendation_ar", FALLBACK_RESULT["recommendation_ar"])
        recommendation_fr = ai.get("recommendation_fr", FALLBACK_RESULT["recommendation_fr"])
        red_flags = bool(ai.get("red_flags", False))

        with get_conn() as conn:
            conn.execute(
                """INSERT INTO pain_records (
                    patient_name, patient_age, record_number, pain_description,
                    body_region, pain_scale_nrs, pain_scale_chancellerie,
                    extracted_symptoms, diagnoses, triage_level,
                    recommendation_ar, record_hash, medical_history, created_at,
                    gender, date_of_birth
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    req.patient_name,
                    req.patient_age,
                    record_number,
                    req.pain_description,
                    req.body_region,
                    req.pain_scale_nrs,
                    pain_scale_chancellerie,
                    json.dumps(extracted_symptoms, ensure_ascii=False),
                    json.dumps(diagnoses, ensure_ascii=False),
                    triage_level,
                    recommendation_ar,
                    record_hash,
                    req.medical_history,
                    created_at,
                    req.gender,
                    req.date_of_birth,
                ),
            )

        log.info("Record created: %s for %s", record_number, req.patient_name)

        return RecordResponse(
            record_number=record_number,
            patient_name=req.patient_name,
            patient_age=req.patient_age,
            gender=req.gender,
            body_region=req.body_region,
            date_of_birth=req.date_of_birth,
            pain_description=req.pain_description,
            medical_history=req.medical_history or "",
            pain_scale_nrs=req.pain_scale_nrs,
            pain_scale_chancellerie=pain_scale_chancellerie,
            extracted_symptoms=extracted_symptoms,
            diagnoses=diagnoses,
            triage_level=triage_level,
            recommendation_ar=recommendation_ar,
            recommendation_fr=recommendation_fr,
            red_flags=red_flags,
            record_hash=record_hash,
            created_at=created_at,
        )

    except Exception as exc:
        log.error("create_record failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail={"error": str(exc)})


@app.get("/api/records/search")
async def search_records_by_name(name: str = ""):
    if not name.strip():
        return {"records": [], "count": 0}
    pattern = f"%{name.strip()}%"
    with get_conn() as conn:
        rows = conn.execute(
            """SELECT record_number, patient_name, patient_age, gender, body_region,
                      pain_scale_nrs, pain_scale_chancellerie, created_at
               FROM pain_records
               WHERE patient_name LIKE ? COLLATE NOCASE
               ORDER BY created_at DESC
               LIMIT 10""",
            (pattern,),
        ).fetchall()
    return {
        "records": [
            {
                "record_number": r["record_number"],
                "patient_name": r["patient_name"],
                "patient_age": r["patient_age"] or 0,
                "gender": r["gender"] or "unknown",
                "body_region": r["body_region"] or "",
                "pain_scale_nrs": r["pain_scale_nrs"] or 0.0,
                "pain_scale_chancellerie": r["pain_scale_chancellerie"] or "",
                "created_at": r["created_at"] or "",
            }
            for r in rows
        ],
        "count": len(rows),
    }


@app.get("/api/records/{record_number}", response_model=RecordResponse)
async def get_record(record_number: str):
    try:
        with get_conn() as conn:
            row = conn.execute(
                "SELECT * FROM pain_records WHERE record_number = ?", (record_number,)
            ).fetchone()

        if not row:
            raise HTTPException(
                status_code=404,
                detail={
                    "error": "السجل غير موجود",
                    "error_fr": "Dossier introuvable",
                },
            )

        # Verify hash
        expected_hash = compute_hash(
            row["patient_name"],
            row["record_number"],
            row["pain_description"],
            row["created_at"],
        )
        hash_verified = expected_hash == row["record_hash"]

        # Parse JSON fields stored as strings
        def safe_json(val):
            if not val:
                return []
            try:
                return json.loads(val)
            except Exception:
                return []

        # Retrieve recommendation_fr from AI field if stored (may be absent for old records)
        diagnoses = safe_json(row["diagnoses"])
        extracted_symptoms = safe_json(row["extracted_symptoms"])

        return RecordResponse(
            record_number=row["record_number"],
            patient_name=row["patient_name"],
            patient_age=row["patient_age"] or 0,
            gender=row["gender"] or "unknown",
            body_region=row["body_region"] or "",
            date_of_birth=row["date_of_birth"],
            pain_description=row["pain_description"] or "",
            medical_history=row["medical_history"] or "",
            pain_scale_nrs=row["pain_scale_nrs"] or 0.0,
            pain_scale_chancellerie=row["pain_scale_chancellerie"] or "",
            extracted_symptoms=extracted_symptoms,
            diagnoses=diagnoses,
            triage_level=row["triage_level"] or "normal",
            recommendation_ar=row["recommendation_ar"] or "",
            recommendation_fr="",
            red_flags=False,
            record_hash=row["record_hash"] or "",
            created_at=row["created_at"] or "",
            hash_verified=hash_verified,
            expert_view=True,
        )

    except HTTPException:
        raise
    except Exception as exc:
        log.error("get_record failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail={"error": str(exc)})


@app.get("/api/symptoms/search")
async def search_symptoms(q: str = ""):
    try:
        if not q:
            return {"results": []}
        pattern = f"%{q.lower()}%"
        with get_conn() as conn:
            rows = conn.execute(
                """SELECT name_en, severity_weight
                   FROM symptoms
                   WHERE LOWER(name_en) LIKE ?
                   ORDER BY severity_weight DESC NULLS LAST
                   LIMIT 10""",
                (pattern,),
            ).fetchall()
        return {"results": [{"name": r["name_en"], "severity_weight": r["severity_weight"]} for r in rows]}
    except Exception as exc:
        log.error("search_symptoms failed: %s", exc)
        raise HTTPException(status_code=500, detail={"error": str(exc)})


@app.get("/api/diseases/{symptom_list}")
async def diseases_by_symptoms(symptom_list: str):
    try:
        symptoms = [s.strip() for s in symptom_list.split(",") if s.strip()]
        if not symptoms:
            return {"results": []}
        results = sqlite_disease_match(symptoms, limit=5)
        return {"results": results}
    except Exception as exc:
        log.error("diseases_by_symptoms failed: %s", exc)
        raise HTTPException(status_code=500, detail={"error": str(exc)})


@app.get("/api/kb/search")
async def kb_search(q: str = "", lang: str = "ar"):
    try:
        if not q:
            return {"results": []}
        with get_conn() as conn:
            rows = conn.execute(
                """SELECT source, question, answer, language, category
                   FROM qa_kb
                   WHERE qa_kb MATCH ? AND language = ?
                   LIMIT 5""",
                (q, lang),
            ).fetchall()
        return {
            "results": [
                {
                    "source": r["source"],
                    "question": r["question"],
                    "answer": r["answer"],
                    "category": r["category"],
                }
                for r in rows
            ]
        }
    except Exception as exc:
        log.error("kb_search failed: %s", exc)
        raise HTTPException(status_code=500, detail={"error": str(exc)})


# ── Expert Pydantic models ────────────────────────────────────────────────────

class ExpertRegisterRequest(BaseModel):
    full_name: str
    accreditation_number: str
    expert_type: str
    email: str
    password: str


class ExpertLoginRequest(BaseModel):
    accreditation_number: str
    password: str


# ── Admin basic-auth ──────────────────────────────────────────────────────────

_http_security = HTTPBasic()

_ADMIN_USER = "admin"
_ADMIN_PASS = "paindiag2026"


def _verify_admin(credentials: HTTPBasicCredentials = Depends(_http_security)):
    ok_user = _secrets.compare_digest(credentials.username.encode(), _ADMIN_USER.encode())
    ok_pass = _secrets.compare_digest(credentials.password.encode(), _ADMIN_PASS.encode())
    if not (ok_user and ok_pass):
        raise HTTPException(
            status_code=401,
            detail="Unauthorized",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials


# ── Expert type labels ────────────────────────────────────────────────────────

_EXPERT_LABELS = {
    "medical_expert": "خبير طبي",
    "insurance": "شركة تأمين",
}

_VALID_EXPERT_TYPES = {"medical_expert", "insurance"}

# ── Admin CSS (kept as a variable to avoid f-string brace escaping) ───────────

_ADMIN_CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: 'Segoe UI', Arial, sans-serif; background: #f0f4f8; direction: rtl; padding: 24px; }
.header { background: #1B4F72; color: #fff; padding: 20px 28px; border-radius: 12px; margin-bottom: 24px; text-align: center; }
.header h1 { font-size: 22px; }
.header p { opacity: .8; font-size: 12px; margin-top: 4px; }
.section { background: #fff; border-radius: 12px; padding: 20px; margin-bottom: 24px; box-shadow: 0 2px 8px rgba(0,0,0,.08); }
h2 { color: #1B4F72; font-size: 15px; margin-bottom: 14px; padding-bottom: 8px; border-bottom: 2px solid #1B4F72; }
table { width: 100%; border-collapse: collapse; }
th { background: #1B4F72; color: #fff; padding: 10px 12px; text-align: right; font-size: 13px; }
td { padding: 10px 12px; border-bottom: 1px solid #eee; font-size: 13px; vertical-align: middle; }
tr:last-child td { border-bottom: none; }
tr:hover { background: #f9fbfc; }
.btn-approve { background: #27ae60; color: #fff; border: none; padding: 5px 14px; border-radius: 6px; cursor: pointer; font-size: 12px; margin-left: 4px; }
.btn-approve:hover { background: #229954; }
.btn-reject { background: #e74c3c; color: #fff; border: none; padding: 5px 14px; border-radius: 6px; cursor: pointer; font-size: 12px; }
.btn-reject:hover { background: #c0392b; }
.note { text-align: center; color: #aaa; font-size: 11px; margin-top: 8px; }
"""


# ── Expert endpoints ──────────────────────────────────────────────────────────

@app.post("/api/experts/register")
async def expert_register(req: ExpertRegisterRequest):
    if req.expert_type not in _VALID_EXPERT_TYPES:
        return JSONResponse(status_code=400, content={"error": "نوع الجهة غير صحيح"})
    password_hash = hashlib.sha256(req.password.encode("utf-8")).hexdigest()
    created_at = datetime.now().isoformat(timespec="seconds")
    try:
        with get_conn() as conn:
            conn.execute(
                """INSERT INTO experts
                   (full_name, accreditation_number, expert_type, email, password_hash, status, created_at)
                   VALUES (?, ?, ?, ?, ?, 'pending', ?)""",
                (req.full_name, req.accreditation_number, req.expert_type, req.email,
                 password_hash, created_at),
            )
    except Exception as exc:
        if "UNIQUE" in str(exc).upper():
            return JSONResponse(status_code=400, content={"error": "رقم الاعتماد مسجل مسبقاً"})
        raise HTTPException(status_code=500, detail={"error": str(exc)})

    return {"success": True, "message": "تم إرسال طلبك، في انتظار موافقة الإدارة"}


@app.post("/api/experts/login")
async def expert_login(req: ExpertLoginRequest):
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM experts WHERE accreditation_number = ?",
            (req.accreditation_number,),
        ).fetchone()

    if row is None:
        return JSONResponse(status_code=401, content={"error": "رقم الاعتماد أو كلمة المرور غير صحيحة"})

    pw_hash = hashlib.sha256(req.password.encode("utf-8")).hexdigest()
    if pw_hash != row["password_hash"]:
        return JSONResponse(status_code=401, content={"error": "رقم الاعتماد أو كلمة المرور غير صحيحة"})

    if row["status"] == "pending":
        return JSONResponse(status_code=403, content={"error": "طلبك قيد المراجعة من الإدارة", "status": "pending"})

    if row["status"] == "rejected":
        return JSONResponse(status_code=403, content={"error": "تم رفض طلبك", "status": "rejected"})

    return {
        "success": True,
        "expert_id": row["id"],
        "full_name": row["full_name"],
        "expert_type": row["expert_type"],
    }


# ── Admin dashboard ───────────────────────────────────────────────────────────

@app.get("/admin", response_class=HTMLResponse)
async def admin_dashboard(credentials: HTTPBasicCredentials = Depends(_verify_admin)):
    with get_conn() as conn:
        pending = conn.execute(
            "SELECT * FROM experts WHERE status='pending' ORDER BY created_at DESC"
        ).fetchall()
        approved = conn.execute(
            "SELECT * FROM experts WHERE status='approved' ORDER BY approved_at DESC"
        ).fetchall()

    pending_rows = ""
    for e in pending:
        label = _EXPERT_LABELS.get(e["expert_type"], e["expert_type"])
        pending_rows += (
            f"<tr>"
            f"<td>{e['full_name']}</td>"
            f"<td><code>{e['accreditation_number']}</code></td>"
            f"<td>{label}</td>"
            f"<td>{e['email']}</td>"
            f"<td>{(e['created_at'] or '')[:10]}</td>"
            f"<td>"
            f"<form method='post' action='/admin/experts/{e['id']}/approve' style='display:inline'>"
            f"<button type='submit' class='btn-approve'>موافقة ✓</button></form>"
            f"<form method='post' action='/admin/experts/{e['id']}/reject' style='display:inline'>"
            f"<button type='submit' class='btn-reject'>رفض ✗</button></form>"
            f"</td></tr>"
        )
    if not pending_rows:
        pending_rows = "<tr><td colspan='6' style='text-align:center;color:#aaa;padding:20px'>لا توجد طلبات معلقة</td></tr>"

    approved_rows = ""
    for e in approved:
        label = _EXPERT_LABELS.get(e["expert_type"], e["expert_type"])
        approved_rows += (
            f"<tr>"
            f"<td>{e['full_name']}</td>"
            f"<td><code>{e['accreditation_number']}</code></td>"
            f"<td>{label}</td>"
            f"<td>{(e['approved_at'] or '')[:10]}</td>"
            f"</tr>"
        )
    if not approved_rows:
        approved_rows = "<tr><td colspan='4' style='text-align:center;color:#aaa;padding:20px'>لا يوجد خبراء موافق عليهم بعد</td></tr>"

    html = f"""<!DOCTYPE html>
<html dir="rtl" lang="ar">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>PainDiag+ — لوحة تحكم الإدارة</title>
  <meta http-equiv="refresh" content="30">
  <style>{_ADMIN_CSS}</style>
</head>
<body>
  <div class="header">
    <h1> +PainDiag — لوحة تحكم الإدارة</h1>
    <p>تتجدد الصفحة تلقائياً كل 30 ثانية</p>
  </div>
  <div class="section">
    <h2>⏳ طلبات التسجيل المعلقة</h2>
    <table>
      <thead><tr>
        <th>الاسم الكامل</th><th>رقم الاعتماد</th><th>النوع</th>
        <th>البريد الإلكتروني</th><th>تاريخ الطلب</th><th>الإجراء</th>
      </tr></thead>
      <tbody>{pending_rows}</tbody>
    </table>
  </div>
  <div class="section">
    <h2>✅ الخبراء الموافق عليهم</h2>
    <table>
      <thead><tr>
        <th>الاسم الكامل</th><th>رقم الاعتماد</th><th>النوع</th><th>تاريخ الموافقة</th>
      </tr></thead>
      <tbody>{approved_rows}</tbody>
    </table>
  </div>
  <p class="note">🔄 يتجدد تلقائياً كل 30 ثانية</p>
</body>
</html>"""
    return HTMLResponse(content=html)


@app.post("/admin/experts/{expert_id}/approve")
async def approve_expert(expert_id: int, credentials: HTTPBasicCredentials = Depends(_verify_admin)):
    approved_at = datetime.now().isoformat(timespec="seconds")
    with get_conn() as conn:
        conn.execute(
            "UPDATE experts SET status='approved', approved_at=? WHERE id=?",
            (approved_at, expert_id),
        )
    return RedirectResponse(url="/admin", status_code=302)


@app.post("/admin/experts/{expert_id}/reject")
async def reject_expert(expert_id: int, credentials: HTTPBasicCredentials = Depends(_verify_admin)):
    with get_conn() as conn:
        conn.execute("UPDATE experts SET status='rejected' WHERE id=?", (expert_id,))
    return RedirectResponse(url="/admin", status_code=302)


@app.get("/health")
async def health():
    try:
        records_count = db_scalar("SELECT COUNT(*) FROM pain_records") or 0
        db_size_mb = round(os.path.getsize(DB_PATH) / (1024 * 1024), 2) if os.path.exists(DB_PATH) else 0
        return {"status": "ok", "db_size_mb": db_size_mb, "records_count": records_count}
    except Exception as exc:
        log.error("health check failed: %s", exc)
        return {"status": "error", "detail": str(exc)}
