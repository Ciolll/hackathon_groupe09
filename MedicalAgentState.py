from typing import TypedDict, List, Optional
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
import pydicom

class MedicalAgentState(TypedDict):
    patient_id: Optional[str]
    patient_sex: Optional[str]    
    patient_age: Optional[str]     
    patient_weight: Optional[str]
    
    current_study_id: str
    study_date : Optional[str]
    
    past_metrics: Optional[str] 
    
    study_zip_path: Optional[str]
    study_dir_path: Optional[str]
    anomaly_detected: Optional[str]
    seg_file_path: Optional[str]
    
    excel_path: str
    excel_password: Optional[str]
    patient_history: Optional[str]
    
    has_history: bool
    history_study_ids: Optional[str]

    clinical_comparison :Optional[str]
    draft_report: Optional[str]
    critic_feedback: Optional[str]
    revision_count: int
    is_human_approved: bool

    lifestyle_data: dict          # Ex: {"tabac": "Actif", "genetique": "Oui", "stress": 8}
    twin_projections: dict        # Ex: {"3_mois": 12000, "6_mois": 15000}
    twin_analysis_report: str     # Le rapport final généré par Mistral