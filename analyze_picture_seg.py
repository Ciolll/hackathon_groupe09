import os
import zipfile
import pydicom
from pathlib import Path

from etape_1 import (
    download_study,
    get_nodule_coordinates_sitk,
    get_geometric_localization,
    extract_physio_constants,
    WORK_DIR
)
from dcm_seg_nodules import extract_seg

def analyze_study_pipeline(study_id: str) -> str:
    """
    Télécharge, extrait et analyse une ancienne étude complète.
    Retourne un résumé formaté (en anglais) des lésions trouvées.
    """   
    try:
        zip_path = download_study(study_id)
    except Exception as e:
        return f"Failure downloading prior study {study_id[:8]}: {str(e)}"
    
    extract_dir = str(Path(WORK_DIR) / f"extracted_{study_id[:8]}")
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
    except zipfile.BadZipFile:
        print(f"  ❌ Error : zip of previous study n° {study_id[:8]} is corrupted.")
        if os.path.exists(zip_path):
            os.remove(zip_path)
        return f"Error: Corrupted DICOM ZIP file on PACS for prior study {study_id[:8]}."
    except Exception as e:
        return f"Extraction error for prior study {study_id[:8]}: {str(e)}"

    target_dir = extract_dir
    mots_cles = ["torax", "thorax", "pulmn", "lung", "1.25mm", "1.25 mm"]
    
    for root, _, files in os.walk(extract_dir):
        dicom_files = [f for f in files if f.endswith('.dcm') and not f.startswith('.')]
        if dicom_files and any(m in root.lower() for m in mots_cles):
            if "sr " not in root.lower() and "seg " not in root.lower():
                target_dir = root
                break

    try:
        dcm_sample = [f for f in os.listdir(target_dir) if f.endswith('.dcm')][0]
        ds = pydicom.dcmread(os.path.join(target_dir, dcm_sample))
        physio = extract_physio_constants(ds)
        
        raw_date = getattr(ds, 'StudyDate', None)
        study_date = f"{str(raw_date)[6:8]}/{str(raw_date)[4:6]}/{str(raw_date)[0:4]}" if raw_date else "Unknown date"
    except Exception:
        physio = {"summary": "Unknown patient context"}
        study_date = "Unknown date"

    try:
        res_path, res_text = extract_seg(target_dir, WORK_DIR)
        seg_file = str(res_path)
        
        # Calcul des coordonnées
        coords = get_nodule_coordinates_sitk(seg_file)
        
        # Localisation Géo-Anatomique (On passe bien les 2 arguments !)
        loc_geo = get_geometric_localization(coords, target_dir)
        clean_res_text = res_text.replace("Date  : None", "").strip()
        
        # Sécurité : Si aucun nodule n'est trouvé
        if coords:
            texte_coords = (
                f"🎯 Coordinates (x,y,z) : {coords['x']}, {coords['y']}, {coords['z']} mm\n"
                f"📦 Volume : {coords['volume']} mm3"
            )
        else:
            texte_coords = "🎯 Coordinates and volume: Unavailable (0 nodules detected by AI or calculation error)."

        # Formatage final du rapport historique
        anomalie = (
            f"--- PRIOR STUDY CONTEXT (Date: {study_date}) ---\n{physio['summary']}\n\n"
            f"--- PRIOR LESION ANALYSIS ---\n{clean_res_text}\n"
            f"📍 Localisation : {loc_geo}\n"
            f"{texte_coords}"
        )
        return anomalie
        
    except Exception as e:
        return f"Failure during AI analysis of prior study {study_id[:8]}: {str(e)}"