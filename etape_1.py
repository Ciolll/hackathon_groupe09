import os
import zipfile
import requests
import pydicom
import numpy as np
import json
import matplotlib.pyplot as plt
from pathlib import Path
from typing import TypedDict, Optional
from langgraph.graph import StateGraph, END
from langchain_mistralai.chat_models import ChatMistralAI
from langchain_core.messages import SystemMessage, HumanMessage

import MedicalAgentState

# Import de ton outil de segmentation
from dcm_seg_nodules import extract_seg

ORTHANC = "http://10.0.1.215:8042" 
AUTH = ("unboxed", "unboxed2026") 
WORK_DIR = "studies"

# ==========================================
# LES OUTILS 
# ==========================================
def download_study(current_study_id: str, out_dir: str = WORK_DIR) -> str:
    """Télécharge une étude complète (.zip) depuis Orthanc."""
    dest = Path(out_dir) / f"study_{current_study_id[:8]}.zip"
    print(f"  ⬇️  Download of study n° {current_study_id[:12]}…")
    with requests.get(f"{ORTHANC}/studies/{current_study_id}/archive",
                      auth=AUTH, stream=True, timeout=120) as r:
        r.raise_for_status()
        total = 0
        with open(dest, "wb") as f:
            for chunk in r.iter_content(8192):
                f.write(chunk)
                total += len(chunk)
    print(f"  ✅ Saved : {dest}  ({total/1e6:.1f} Mo)")
    return str(dest)

def show_dicom(path: str):
    """Affiche une coupe DICOM avec ses métadonnées principales."""
    ds  = pydicom.dcmread(path)
    img = ds.pixel_array.astype(np.float32)
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)

    fig, ax = plt.subplots(1, 1, figsize=(6, 6), facecolor="#111")
    ax.imshow(img, cmap="gray", interpolation="bilinear")
    ax.set_title(
        f"Patient: {getattr(ds,'PatientID','?')}  |  "
        f"Modality: {getattr(ds,'Modality','?')}  |  "
        f"Slice: {getattr(ds,'InstanceNumber','?')}",
        color="white", fontsize=10, pad=10
    )
    ax.axis("off")
    plt.tight_layout()
    plt.show(block=False) # block=False permet au script de continuer sans bloquer
    plt.pause(3) # Affiche l'image pendant 3 secondes
    plt.close()

    print(f"  Dimensions    : {img.shape}")
    print(f"  Pixel spacing : {getattr(ds,'PixelSpacing','N/A')}")
    print(f"  Study date    : {getattr(ds,'StudyDate','N/A')}")
    return ds
    
import SimpleITK as sitk
def get_nodule_coordinates_sitk(seg_path: str):
    """Calcule le centre de masse (x,y,z) et le volume du nodule."""
    try:
        seg_img = sitk.ReadImage(seg_path)
        label_stats = sitk.LabelShapeStatisticsImageFilter()
        label_stats.Execute(seg_img)
        all_labels = [l for l in label_stats.GetLabels() if l != 0]
        if not all_labels: return None
        
        target = all_labels[0]
        centroid = label_stats.GetCentroid(target)
        return {
            "x": round(centroid[0], 2), "y": round(centroid[1], 2), "z": round(centroid[2], 2),
            "volume": round(label_stats.GetPhysicalSize(target), 2)
        }
    except: return None

def get_geometric_localization(coords, study_dir):
    """Détermine le lobe pulmonaire par calcul géométrique pur."""
    if not coords: return "Localisation spatiale indéterminée"
    
    # On définit arbitrairement des zones basées sur un thorax standard
    # X < 0 : Droite | X > 0 : Gauche (Référentiel LPS)
    cote = "Droit" if coords['x'] < 0 else "Gauche"
    
    # Pour Z, on pourrait affiner avec les limites du scanner (get_lung_limits)
    # Ici en version simplifiée :
    if coords['z'] > 20: lobe = "Superior"
    elif coords['z'] < -60: lobe = "Inferior"
    else: lobe = "Middle"
    
    return f"Lobe {lobe} {cote}"

def extract_physio_constants(ds):
    """Extrait le poids, l'âge et le sexe des métadonnées DICOM."""
    weight = getattr(ds, 'PatientWeight', None)
    age = getattr(ds, 'PatientAge', 'N/A')
    sex = getattr(ds, 'PatientSex', 'U')
    size = getattr(ds, 'PatientSize', None)
    
    weight_str = str(weight) if weight is not None else "N/A"
    age_str = str(age) if age is not None else "N/A"
    sex_str = str(sex) if sex is not None else "U"
        
    return {
        "weight": weight_str, "age": age_str, "sex": sex_str, 
        "summary": f"Age: {age_str}, Gender: {sex_str}, Weight: {weight_str}kg"
    }
# ==========================================
# DÉFINITION DES NOEUDS LANGGRAPH
# ==========================================
def node_process_current(state: MedicalAgentState):
    current_study_id = state["current_study_id"]
    
    # A. Acquisition & Décompression
    zip_path = download_study(current_study_id) # ta fonction existante
    extract_dir = str(Path(WORK_DIR) / f"extracted_{current_study_id[:8]}")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
        
    # B. Recherche de la série thoracique 1.25mm
    target_dir = extract_dir
    mots_cles = ["torax", "thorax", "pulmn", "lung", "1.25mm", "1.25 mm"]
    for root, _, files in os.walk(extract_dir):
        if any(f.endswith('.dcm') for f in files) and any(m in root.lower() for m in mots_cles):
            if "sr " not in root.lower() and "seg " not in root.lower():
                target_dir = root
                break
    
    # C. Extraction des métadonnées Patient & Constantes
    dcm_sample = [f for f in os.listdir(target_dir) if f.endswith('.dcm')][0]
    ds = pydicom.dcmread(os.path.join(target_dir, dcm_sample))
    
    physio = extract_physio_constants(ds)
    patient_id = getattr(ds, 'PatientID', 'INCONNU')

    raw_date = getattr(ds, 'StudyDate', None)
    if raw_date:
        # Formatage optionnel pour lisibilité : 20231025 -> 25/10/2023
        study_date = f"{raw_date[6:8]}/{raw_date[4:6]}/{raw_date[0:4]}"
    else:
        study_date = "Unknown date"

        
    # D. Segmentation IA & Analyse Spatiale
    print(f"🔄 Analysis of the current study {patient_id}...")
    try:
        res_path, res_text = extract_seg(target_dir, WORK_DIR)
        seg_file = str(res_path)
        
        # Calcul des coordonnées SITK
        coords = get_nodule_coordinates_sitk(seg_file)
        
        # Localisation Géo-Anatomique
        loc_geo = get_geometric_localization(coords, target_dir)
        clean_res_text = res_text.replace("Date  : None", "").strip()
        # Compilation du résultat ultra-précis pour le futur rapport
        anomalie = (
            f"--- PATIENT CONTEXT ---\n{physio['summary']}\n\n"
            f"--- Lesion Analysis ---\n{clean_res_text}\n"
            f"📍 Localisation : {loc_geo}\n"
            f"🎯 Coordonidates (x,y,z) : {coords['x']}, {coords['y']}, {coords['z']} mm\n"
            f"📦 Volume : {coords['volume']} mm3"
        )
    except Exception as e:
        anomalie = f"Failure during the analysis: {str(e)}"
        seg_file = None

    return {
        "study_zip_path": zip_path,
        "study_dir_path": extract_dir,
        "patient_id": patient_id,
        "study_date": study_date,
        "patient_age": physio['age'],
        "patient_sex": physio['sex'],
        "patient_weight": physio['weight'],
        "anomaly_detected": anomalie,
        "seg_file_path": seg_file
    }