import os
import requests
from pathlib import Path
from datetime import datetime
from MedicalAgentState import MedicalAgentState

# --- Configuration Orthanc ---
ORTHANC = "http://10.0.1.215:8042"
AUTH    = ("unboxed", "unboxed2026")

def upload_dicom(path: str) -> str | None:
    """Envoie un fichier .dcm vers Orthanc. Retourne l'instance ID."""
    try:
        with open(path, "rb") as f:
            r = requests.post(f"{ORTHANC}/instances", auth=AUTH,
                              data=f.read(), headers={"Content-Type": "application/dicom"})
        if r.status_code == 200:
            iid = r.json().get("ID")
            print(f"    ✅ Upload OK  →  instance ID : {iid}")
            return iid
        print(f"    ❌ Erreur {r.status_code} : {r.text}")
    except Exception as e:
        print(f"    ❌ Erreur de connexion au PACS : {e}")
    return None

def upload_dicom_folder(folder: str) -> None:
    """Upload tous les .dcm d'un dossier (récursif) vers Orthanc."""
    files = list(Path(folder).rglob("*.dcm"))
    print(f"  📂 {len(files)} fichier(s) .dcm trouvé(s) dans '{folder}'")
    print("  " + "─" * 48)
    ok = err = 0
    for f in files:
        try:
            with open(f, "rb") as fh:
                r = requests.post(f"{ORTHANC}/instances", auth=AUTH,
                                  data=fh.read(), headers={"Content-Type": "application/dicom"})
            if r.status_code == 200:
                print(f"    ✅  {f.name}")
                ok += 1
            else:
                print(f"    ❌  {f.name}  (HTTP {r.status_code})")
                err += 1
        except Exception as e:
            print(f"    ❌  {f.name} (Erreur réseau: {e})")
            err += 1
            
    print("  " + "─" * 48)
    print(f"  📊 Résultat de l'export PACS : {ok} OK · {err} erreur(s)")


def save_report_1(state :MedicalAgentState):
    return

def save_report(state: MedicalAgentState) -> dict:
    print("\n▶️ NŒUD 7 : SAUVEGARDE ET EXPORT PACS (SAVE_REPORT)")
    
    # ==========================================
    # 1. SAUVEGARDE LOCALE DU COMPTE-RENDU (TEXTE)
    # ==========================================
    patient_id = state.get("patient_id", "INCONNU")
    study_id = state.get("current_study_id", "INCONNU")
    rapport_final = state.get("draft_report", "Aucun contenu généré.")
    
    dossier_destination = "rapports_finaux"
    os.makedirs(dossier_destination, exist_ok=True)
    
    date_jour = datetime.now().strftime("%Y%m%d_%H%M%S")
    nom_fichier = f"CR_{patient_id}_{study_id[:8]}_{date_jour}.md" 
    chemin_complet = os.path.join(dossier_destination, nom_fichier)
    
    signature = f"\n\n---\n*Rapport validé électroniquement le {datetime.now().strftime('%d/%m/%Y à %H:%M')}*\n*Généré par Agent Radiology Hub*"
    contenu_a_sauvegarder = rapport_final + signature
    
    try:
        with open(chemin_complet, "w", encoding="utf-8") as file:
            file.write(contenu_a_sauvegarder)
        print(f"  📝 Compte-rendu texte enregistré : {chemin_complet}")
    except Exception as e:
        print(f"  ❌ Erreur lors de la sauvegarde texte : {e}")


    # ==========================================
    # 2. UPLOAD DES RÉSULTATS IA VERS ORTHANC
    # ==========================================
    # On vérifie si l'IA a stocké un chemin vers les masques DICOM SEG
    dossier_dicom_ia = state.get("seg_file_path") 
    
    if dossier_dicom_ia and os.path.exists(dossier_dicom_ia):
        print("  🔄 Début du transfert des images IA vers Orthanc...")
        
        if os.path.isdir(dossier_dicom_ia):
            upload_dicom_folder(dossier_dicom_ia)
        elif os.path.isfile(dossier_dicom_ia) and str(dossier_dicom_ia).endswith('.dcm'):
            upload_dicom(dossier_dicom_ia)
            
    else:
        print("  ℹ️ Aucun fichier DICOM (.dcm) d'IA trouvé dans 'seg_file_path' pour l'export PACS.")

    return {"draft_report": contenu_a_sauvegarder}