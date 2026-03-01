import pandas as pd
import os
import pydicom
import requests
import zipfile

from pathlib import Path
from typing import Any
from langchain_core.messages import AIMessage
from read_excel import read_excel
from etape_1 import download_study


# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
ORTHANC = "http://10.0.1.215:8042"
AUTH = ("unboxed", "unboxed2026")
WORK_DIR = "studies"

# Chargement initial pour récupérer les noms de colonnes
df_init = read_excel("data/protected-clinical-data.xlsx", "Trois petits cochons")
df_init.columns = df_init.columns.str.strip()

COL_SERIES = next(c for c in df_init.columns if "Série" in c)
COL_PATIENT_ID = next(c for c in df_init.columns if "PatientID" in c)
COL_ACCESSION = next(c for c in df_init.columns if "AccessionNumber" in c)
COL_REPORT = next(c for c in df_init.columns if "Clinical" in c)


# ─────────────────────────────────────────────
# GET STUDY DATE FROM ORTHANC
# ─────────────────────────────────────────────
def get_study_date_from_orthanc(accession_number: str):
    response = requests.post(
        f"{ORTHANC}/tools/find",
        auth=AUTH,
        json={"Level": "Study", "Query": {"AccessionNumber": accession_number}}
    )
    response.raise_for_status()
    studies = response.json()

    if not studies:
        return None

    study_id = studies[0]
    study_info_response = requests.get(f"{ORTHANC}/studies/{study_id}", auth=AUTH)
    study_info_response.raise_for_status()
    study_info = study_info_response.json()

    raw_date = study_info.get("MainDicomTags", {}).get("StudyDate")
    if not raw_date:
        return None

    study_date = pd.to_datetime(raw_date, format="%Y%m%d")
    return study_date, study_id  # retourne aussi le study_id


# ─────────────────────────────────────────────
# FETCH HISTORY
# ─────────────────────────────────────────────
def fetch_history(state: dict) -> dict:
    excel_path = state.get("excel_path", "patients_history.xlsx")
    password = state.get("excel_password", None)
    current_study_id = str(state.get("current_study_id", "")).strip()
    existing_messages = state.get("messages", [])
    
    zip_path = download_study(current_study_id) 
    extract_dir = str(Path(WORK_DIR) / f"extracted_{current_study_id[:8]}")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
        
    target_dir = extract_dir
    mots_cles = ["torax", "thorax", "pulmn", "lung", "1.25mm", "1.25 mm"]
    for root, _, files in os.walk(extract_dir):
        if any(f.endswith('.dcm') for f in files) and any(m in root.lower() for m in mots_cles):
            if "sr " not in root.lower() and "seg " not in root.lower():
                target_dir = root
                break
    
    dcm_sample = [f for f in os.listdir(target_dir) if f.endswith('.dcm')][0]
    ds = pydicom.dcmread(os.path.join(target_dir, dcm_sample))

    # ─────────────────────────────
    # Lecture Excel
    # ─────────────────────────────
    df = read_excel(excel_path, password=password)
    df.columns = df.columns.str.strip()
 
    patient_id = getattr(ds, 'PatientID', 'INCONNU')
    raw_date = getattr(ds, 'StudyDate', None)
    if raw_date is None:
        return {**state, "error": "Aucune date trouvée dans le DICOM courant", "has_history": False}

    current_study_date = pd.to_datetime(raw_date, format="%Y%m%d")
    print(f"[DEBUG] PatientID courant : {patient_id} | Date examen courant : {current_study_date.date()}")

    # ─────────────────────────────
    # Filtrage patient dans Excel
    # ─────────────────────────────
    mask = df[COL_PATIENT_ID].fillna("").astype(str).str.strip().str.upper() == patient_id.upper()
    patient_df = df[mask].copy()

    # ─────────────────────────────
    # Récupération dates historiques et study_id
    # ─────────────────────────────
    records_with_dates = []
    history_study_ids = []

    for _, row in patient_df.iterrows():
        accession = str(row[COL_ACCESSION]).strip()
        if accession == current_study_id:
            continue

        result = get_study_date_from_orthanc(accession)
        if result is None:
            continue

        study_date, study_id = result
        if study_date < current_study_date:
            records_with_dates.append((study_date, row))
            history_study_ids.append(study_id)

    # TRI DÉCROISSANT
    records_with_dates.sort(key=lambda x: x[0], reverse=True)
    print(f"[DEBUG] Examens trouvés pour patient dans l'historique : {len(records_with_dates)}")

    # ─────────────────────────────
    # Construction historique
    # ─────────────────────────────
    history_records = []
    for study_date, row in records_with_dates:
        record_str = (
            f"Date={study_date.date()} | "
            f"AccessionNumber={row[COL_ACCESSION]} | "
            f"Série={row[COL_SERIES]} | "
            f"Report={row[COL_REPORT]}"
        )
        history_records.append(record_str)

    has_history = len(history_records) > 0
    summary = "\n".join(history_records) if has_history else "[fetch_history] Aucun antécédent antérieur trouvé."

    return {
        **state,
        "has_history": has_history,
        "patient_history": "; ".join(history_records),
        "history_study_ids": ",".join(history_study_ids),
        "error": None,
        "messages": existing_messages + [AIMessage(content=summary)]
    }