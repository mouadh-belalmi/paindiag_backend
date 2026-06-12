"""
init_db.py — Initialize PainDiag+ SQLite database and load all datasets.
Run with: python init_db.py
"""

import sqlite3
import json
import os
import sys

try:
    import pandas as pd
except ImportError:
    sys.exit("pandas is required: pip install pandas openpyxl")

# ── Paths ────────────────────────────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "paindiag.db")
DS = os.path.join(BASE_DIR, "datasets")

# ── Schema ───────────────────────────────────────────────────────────────────

SCHEMA = """
CREATE TABLE IF NOT EXISTS body_regions (
    id      INTEGER PRIMARY KEY,
    name_en TEXT,
    name_fr TEXT,
    name_ar TEXT
);

CREATE TABLE IF NOT EXISTS diseases (
    id             INTEGER PRIMARY KEY,
    name_en        TEXT UNIQUE,
    name_fr        TEXT,
    description_en TEXT,
    precautions    TEXT
);

CREATE TABLE IF NOT EXISTS symptoms (
    id             INTEGER PRIMARY KEY,
    name_en        TEXT UNIQUE,
    severity_weight INTEGER
);

CREATE TABLE IF NOT EXISTS disease_symptom (
    disease_id INTEGER,
    symptom_id INTEGER,
    FOREIGN KEY(disease_id) REFERENCES diseases(id),
    FOREIGN KEY(symptom_id) REFERENCES symptoms(id),
    PRIMARY KEY(disease_id, symptom_id)
);

CREATE TABLE IF NOT EXISTS ddx_conditions (
    id       INTEGER PRIMARY KEY,
    code     TEXT,
    name_fr  TEXT,
    name_en  TEXT,
    severity TEXT
);

CREATE TABLE IF NOT EXISTS ddx_evidences (
    id          INTEGER PRIMARY KEY,
    code        TEXT,
    question_fr TEXT,
    question_en TEXT,
    type        TEXT
);

CREATE TABLE IF NOT EXISTS pain_records (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_name         TEXT,
    patient_age          INTEGER,
    record_number        TEXT UNIQUE,
    pain_description     TEXT,
    body_region          TEXT,
    pain_scale_nrs       REAL,
    pain_scale_chancellerie TEXT,
    extracted_symptoms   TEXT,
    diagnoses            TEXT,
    triage_level         TEXT,
    recommendation_ar    TEXT,
    record_hash          TEXT,
    medical_history      TEXT,
    created_at           TEXT
);

CREATE VIRTUAL TABLE IF NOT EXISTS qa_kb USING fts5(
    source, question, answer, language, category
);

CREATE TABLE IF NOT EXISTS mornet_scale (
    id                    INTEGER PRIMARY KEY,
    level                 TEXT,
    label_fr              TEXT,
    label_ar              TEXT,
    nrs_min               REAL,
    nrs_max               REAL,
    compensation_eur_min  INTEGER,
    compensation_eur_max  INTEGER
);
"""

# ── Helpers ──────────────────────────────────────────────────────────────────

def clean(val):
    """Strip whitespace from a string value, return None for blanks."""
    if pd.isna(val):
        return None
    s = str(val).strip()
    return s if s else None


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print(f"Creating database at {DB_PATH}")
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(SCHEMA)
    conn.commit()

    # ── Body regions ─────────────────────────────────────────────────────────
    regions = [
        (1,  "head",           "tête",             "الرأس"),
        (2,  "neck",           "cou",              "الرقبة"),
        (3,  "chest",          "poitrine",         "الصدر"),
        (4,  "abdomen",        "abdomen",          "البطن"),
        (5,  "left_shoulder",  "épaule gauche",    "الكتف الأيسر"),
        (6,  "right_shoulder", "épaule droite",    "الكتف الأيمن"),
        (7,  "upper_back",     "dos supérieur",    "أعلى الظهر"),
        (8,  "lower_back",     "dos inférieur",    "أسفل الظهر"),
        (9,  "left_arm",       "bras gauche",      "الذراع الأيسر"),
        (10, "right_arm",      "bras droit",       "الذراع الأيمن"),
        (11, "left_leg",       "jambe gauche",     "الساق اليسرى"),
        (12, "right_leg",      "jambe droite",     "الساق اليمنى"),
    ]
    conn.executemany(
        "INSERT OR IGNORE INTO body_regions (id, name_en, name_fr, name_ar) VALUES (?,?,?,?)",
        regions,
    )
    conn.commit()
    print("  Body regions inserted.")

    # ── Mornet scale ─────────────────────────────────────────────────────────
    mornet = [
        (1, "1/7", "très léger",      "خفيف جداً",    0, 2,  0,     2000),
        (2, "2/7", "léger",           "خفيف",         2, 3,  2000,  4000),
        (3, "3/7", "modéré",          "معتدل",        3, 5,  4000,  8000),
        (4, "4/7", "moyen",           "متوسط",        5, 6,  8000,  20000),
        (5, "5/7", "assez important", "مهم نسبياً",   6, 7,  20000, 35000),
        (6, "6/7", "important",       "مهم",          7, 9,  35000, 50000),
        (7, "7/7", "très important",  "مهم جداً",     9, 10, 50000, 80000),
    ]
    conn.executemany(
        "INSERT OR IGNORE INTO mornet_scale VALUES (?,?,?,?,?,?,?,?)",
        mornet,
    )
    conn.commit()
    print("  Mornet scale inserted.")

    # ── A) dataset.csv — diseases, symptoms, relationships ───────────────────
    diseases_loaded = 0
    symptoms_loaded = 0
    try:
        df = pd.read_csv(os.path.join(DS, "dataset.csv"), encoding="utf-8")
        symptom_cols = [c for c in df.columns if c.startswith("Symptom_")]

        # Collect unique diseases and symptoms
        disease_map = {}   # name -> id
        symptom_map = {}   # name -> id

        for _, row in df.iterrows():
            disease_name = clean(row["Disease"])
            if not disease_name:
                continue

            if disease_name not in disease_map:
                cur = conn.execute(
                    "INSERT OR IGNORE INTO diseases (name_en) VALUES (?)",
                    (disease_name,),
                )
                conn.execute(
                    "SELECT id FROM diseases WHERE name_en = ?", (disease_name,)
                )
                disease_id = conn.execute(
                    "SELECT id FROM diseases WHERE name_en = ?", (disease_name,)
                ).fetchone()[0]
                disease_map[disease_name] = disease_id

            d_id = disease_map[disease_name]

            for col in symptom_cols:
                sym = clean(row.get(col))
                if not sym:
                    continue

                if sym not in symptom_map:
                    conn.execute(
                        "INSERT OR IGNORE INTO symptoms (name_en) VALUES (?)", (sym,)
                    )
                    symptom_id = conn.execute(
                        "SELECT id FROM symptoms WHERE name_en = ?", (sym,)
                    ).fetchone()[0]
                    symptom_map[sym] = symptom_id

                s_id = symptom_map[sym]
                conn.execute(
                    "INSERT OR IGNORE INTO disease_symptom (disease_id, symptom_id) VALUES (?,?)",
                    (d_id, s_id),
                )

        conn.commit()
        diseases_loaded = len(disease_map)
        symptoms_loaded = len(symptom_map)
        print(f"  dataset.csv -> {diseases_loaded} diseases, {symptoms_loaded} symptoms loaded.")
    except FileNotFoundError:
        print("  WARNING: datasets/dataset.csv not found — skipped.")
    except Exception as e:
        print(f"  WARNING: Failed to load dataset.csv — {e}")

    # ── B) Symptom-severity.csv — update severity_weight ─────────────────────
    try:
        df = pd.read_csv(os.path.join(DS, "Symptom-severity.csv"), encoding="utf-8")
        updated = 0
        for _, row in df.iterrows():
            sym = clean(row.get("Symptom"))
            weight = row.get("weight")
            if sym and pd.notna(weight):
                cur = conn.execute(
                    "UPDATE symptoms SET severity_weight = ? WHERE name_en = ?",
                    (int(weight), sym),
                )
                updated += cur.rowcount
        conn.commit()
        print(f"  Symptom-severity.csv -> {updated} symptoms updated with weight.")
    except FileNotFoundError:
        print("  WARNING: datasets/Symptom-severity.csv not found — skipped.")
    except Exception as e:
        print(f"  WARNING: Failed to load Symptom-severity.csv — {e}")

    # ── C) symptom_Description.csv — update description_en ───────────────────
    try:
        df = pd.read_csv(os.path.join(DS, "symptom_Description.csv"), encoding="utf-8")
        updated = 0
        for _, row in df.iterrows():
            disease = clean(row.get("Disease"))
            desc = clean(row.get("Description"))
            if disease and desc:
                cur = conn.execute(
                    "UPDATE diseases SET description_en = ? WHERE name_en = ?",
                    (desc, disease),
                )
                updated += cur.rowcount
        conn.commit()
        print(f"  symptom_Description.csv -> {updated} diseases updated with description.")
    except FileNotFoundError:
        print("  WARNING: datasets/symptom_Description.csv not found — skipped.")
    except Exception as e:
        print(f"  WARNING: Failed to load symptom_Description.csv — {e}")

    # ── D) symptom_precaution.csv — update precautions ───────────────────────
    try:
        df = pd.read_csv(os.path.join(DS, "symptom_precaution.csv"), encoding="utf-8")
        precaution_cols = [c for c in df.columns if c.startswith("Precaution_")]
        updated = 0
        for _, row in df.iterrows():
            disease = clean(row.get("Disease"))
            if not disease:
                continue
            parts = [clean(row.get(c)) for c in precaution_cols]
            prec_str = ", ".join(p for p in parts if p)
            if prec_str:
                cur = conn.execute(
                    "UPDATE diseases SET precautions = ? WHERE name_en = ?",
                    (prec_str, disease),
                )
                updated += cur.rowcount
        conn.commit()
        print(f"  symptom_precaution.csv -> {updated} diseases updated with precautions.")
    except FileNotFoundError:
        print("  WARNING: datasets/symptom_precaution.csv not found — skipped.")
    except Exception as e:
        print(f"  WARNING: Failed to load symptom_precaution.csv — {e}")

    # ── E) release_conditions.json — ddx_conditions ──────────────────────────
    ddx_conditions_loaded = 0
    try:
        with open(os.path.join(DS, "release_conditions.json"), encoding="utf-8") as f:
            conditions = json.load(f)

        rows = []
        for _key, cond in conditions.items():
            rows.append((
                cond.get("icd10-id") or cond.get("icd10_code"),
                cond.get("cond-name-fr") or cond.get("condition_name"),
                cond.get("cond-name-eng") or cond.get("condition_name"),
                str(cond.get("severity", "")),
            ))

        conn.executemany(
            "INSERT INTO ddx_conditions (code, name_fr, name_en, severity) VALUES (?,?,?,?)",
            rows,
        )
        conn.commit()
        ddx_conditions_loaded = len(rows)
        print(f"  release_conditions.json -> {ddx_conditions_loaded} conditions loaded.")
    except FileNotFoundError:
        print("  WARNING: datasets/release_conditions.json not found — skipped.")
    except Exception as e:
        print(f"  WARNING: Failed to load release_conditions.json — {e}")

    # ── F) release_evidences.json — ddx_evidences ────────────────────────────
    ddx_evidences_loaded = 0
    try:
        with open(os.path.join(DS, "release_evidences.json"), encoding="utf-8") as f:
            evidences = json.load(f)

        rows = []
        for key, ev in evidences.items():
            rows.append((
                ev.get("code_question") or key,
                ev.get("question_fr", ""),
                ev.get("question_en", ""),
                ev.get("data_type") or ev.get("type", ""),
            ))

        conn.executemany(
            "INSERT INTO ddx_evidences (code, question_fr, question_en, type) VALUES (?,?,?,?)",
            rows,
        )
        conn.commit()
        ddx_evidences_loaded = len(rows)
        print(f"  release_evidences.json -> {ddx_evidences_loaded} evidences loaded.")
    except FileNotFoundError:
        print("  WARNING: datasets/release_evidences.json not found — skipped.")
    except Exception as e:
        print(f"  WARNING: Failed to load release_evidences.json — {e}")

    # ── G) AHD.xlsx — qa_kb (FTS5) ───────────────────────────────────────────
    ahd_loaded = 0
    try:
        df = pd.read_excel(os.path.join(DS, "AHD.xlsx"), nrows=50000)

        # Normalise column names to lowercase for robust matching
        df.columns = [c.strip().lower() for c in df.columns]

        # Detect question / answer / category columns flexibly
        col_map = {}
        for target, candidates in {
            "question": ["question", "q", "سؤال"],
            "answer":   ["answer",   "a", "ans", "جواب", "إجابة"],
            "category": ["category", "cat", "تصنيف", "فئة"],
        }.items():
            for candidate in candidates:
                if candidate in df.columns:
                    col_map[target] = candidate
                    break

        if "question" not in col_map or "answer" not in col_map:
            print(f"  WARNING: AHD.xlsx — could not identify question/answer columns "
                  f"(found: {list(df.columns)}) — skipped.")
        else:
            q_col = col_map["question"]
            a_col = col_map["answer"]
            c_col = col_map.get("category", None)

            rows = []
            for _, row in df.iterrows():
                q = clean(row[q_col])
                a = clean(row[a_col])
                if not q or not a:
                    continue
                cat = clean(row[c_col]) if c_col else None
                rows.append(("AHD", q, a, "ar", cat or ""))

            conn.executemany(
                "INSERT INTO qa_kb (source, question, answer, language, category) VALUES (?,?,?,?,?)",
                rows,
            )
            conn.commit()
            ahd_loaded = len(rows)
            print(f"  AHD.xlsx -> {ahd_loaded} rows loaded into qa_kb.")
    except FileNotFoundError:
        print("  WARNING: datasets/AHD.xlsx not found — skipped.")
    except Exception as e:
        print(f"  WARNING: Failed to load AHD.xlsx — {e}")

    # ── Summary ───────────────────────────────────────────────────────────────
    db_size_mb = os.path.getsize(DB_PATH) / (1024 * 1024)

    print("\n" + "=" * 50)
    print("PainDiag+ database initialisation complete")
    print("=" * 50)
    print(f"  Diseases loaded          : {diseases_loaded}")
    print(f"  Symptoms loaded          : {symptoms_loaded}")
    print(f"  DDXPlus conditions       : {ddx_conditions_loaded}")
    print(f"  DDXPlus evidences        : {ddx_evidences_loaded}")
    print(f"  AHD rows loaded          : {ahd_loaded}")
    print(f"  Database size            : {db_size_mb:.2f} MB")
    print(f"  Database path            : {DB_PATH}")
    print("=" * 50)

    conn.close()


if __name__ == "__main__":
    main()
