import requests
import zipfile
import os
from pathlib import Path

from analyze_picture_seg import analyze_study_pipeline

import MedicalAgentState
from typing import List, Tuple, Optional


ORTHANC = "http://10.0.1.215:8042"
AUTH = ("unboxed", "unboxed2026")
def node_process_orthanc(state: MedicalAgentState):
    print("\n▶️ NODE 3 : ANALYSIS OF THE PRIOR STUDY (ORTHANC)")
    
    history_str = state.get("history_study_ids", "")
    print (history_str)
    
    if not history_str or history_str.strip() == "":
        print("  ℹ️ No prior exams found.")
        return {"past_metrics": "No prior metrics."}
    
    all_sids = [sid.strip() for sid in history_str.split(",") if sid.strip()]
    
    if not all_sids:
        return {"past_metrics": "No valide prior exams found."}
        
    etudes_a_traiter = all_sids[:3]
    
    print(f"  📂 {len(all_sids)} prior exams listed.")
    print(f"  ⚡ Limitation t the {len(etudes_a_traiter)} most recents exams. Launching Computer Vision Analysis...")
    
    historique_complet_textes = []
    
    # 4. Boucle d'analyse sur les anciennes images
    for index, sid in enumerate(etudes_a_traiter):
        # L'index + 1 permet d'indiquer "Antécédent 1", "Antécédent 2", etc.
        print(f"  🔄 Analysis : Prior study N-{index + 1} (ID: {sid[:8]})...")
        
        # Lancement de l'IA sur cette ancienne étude
        try:
            resultat_ancien = analyze_study_pipeline(sid)
        except Exception as e:
            print(f"  ❌ Failure during study n° {sid[:8]} : {e}")
            resultat_ancien = f"failure duirng prior exams analysis : {e}"
        
        # Formatage du bloc (on ne met plus la date puisqu'on ne l'a pas ici, mais l'ordre suffit)
        bloc_etude = (
            f"--- PRIOR EXAM N-{index + 1} | STUDY ID : {sid[:8]} ---\n"
            f"{resultat_ancien}\n"
        )
        historique_complet_textes.append(bloc_etude)
        
    # 5. Fusion de tous les blocs
    texte_final = "\n\n".join(historique_complet_textes)
    
    print("  ✅ Node 3  : Prior studies were analyzed.")
    print(texte_final)
    
    # Retourne le texte complet pour le nœud de comparaison
    return {"past_metrics": texte_final}