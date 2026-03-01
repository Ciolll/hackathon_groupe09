import streamlit as st
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
import traceback
import sys

class StreamlitPrintCapture:
    def __init__(self, st_container):
        self.st_container = st_container
        self.logs = ""

    def write(self, text):
        # On ajoute le texte au fur et à mesure et on met à jour la boîte
        if text != "":
            self.logs += text
            self.st_container.code(self.logs, language="bash")

    def flush(self):
        pass
        
# ==========================================
# 1. COPIE DE TON AGENT LANGGRAPH ICI
# ==========================================
# (Remplace ces fonctions par TES vraies fonctions du notebook)
from MedicalAgentState import MedicalAgentState
from etape_1 import node_process_current
from fetch_history import fetch_history
from access_orthanc import node_process_orthanc
from comparing import node_clinical_comparison
from RAG import prepare_documents
from agent_rapport import RAGAgent, draft_report
from agent_review import review_report
from savereport import save_report


def route_after_review(state: MedicalAgentState) -> str:
    """Si le relecteur trouve des erreurs ET qu'on a fait moins de 3 essais, on recommence."""
    feedback = state.get("critic_feedback", "")
    retries = state.get("revision_count", 0)
    
    if feedback != "OK" and retries < 3:
        print(f"  ⚠️ Erreur trouvée par le critique : {feedback}. Retour à la rédaction.")
        return "draft_report"
    
    # Si c'est OK ou qu'on a atteint la limite, on avance vers la validation finale
    return "save_report"

def route_after_history(state: MedicalAgentState) -> str:
    """
    Vérifie dans la mémoire si l'Excel a trouvé des antécédents.
    Si oui -> On télécharge les vieilles images sur Orthanc.
    Si non -> On passe directement à la rédaction du rapport.
    """
    historique_excel = state.get("patient_history", "")
    
    # Si la liste est vide (ou n'existe pas)
    if not historique_excel:
        print("  ⏭️ ROUTAGE : Aucun antécédent dans l'Excel. Saut d'Orthanc et de la comparaison.")
        return "draft_report"
    else:
        print("  🔀 ROUTAGE : Antécédents trouvés dans l'Excel. Envoi vers Orthanc (PACS).")
        return "access_orthanc"

workflow = StateGraph(MedicalAgentState)

# Ajout des nœuds
workflow.add_node("process_current", node_process_current)
workflow.add_node("fetch_history", fetch_history)
workflow.add_node("access_orthanc", node_process_orthanc)
workflow.add_node("comparison", node_clinical_comparison)
workflow.add_node("draft_report", draft_report)
workflow.add_node("review_report", review_report)
workflow.add_node("save_report",save_report)

# Définition du flux principal (les arêtes)
workflow.add_edge(START, "process_current")
workflow.add_edge("process_current", "fetch_history")


workflow.add_conditional_edges(
    "fetch_history",
    route_after_history,
    {
        "draft_report" :"draft_report",
        "access_orthanc" : "access_orthanc"
    }
)

workflow.add_edge("access_orthanc","comparison")
workflow.add_edge("comparison","draft_report")

workflow.add_edge("draft_report", "review_report")

workflow.add_conditional_edges(
    "review_report",
    route_after_review,
    {
        "draft_report": "draft_report",   # Retour en arrière
        "save_report": "save_report"    # On avance
    }
)
workflow.add_edge("save_report", END)

# ==========================================
# 2. L'INTERFACE STREAMLIT INTERACTIVE
# ==========================================
st.set_page_config(page_title="Agent Radiology", layout="wide", page_icon="🏥")
st.title("🏥 AGENT RADIOLOGY : Multi-Agent Hub")

if "memory" not in st.session_state:
    st.session_state.memory = MemorySaver()

if "config" not in st.session_state:
    st.session_state.config = None

medical_agent = workflow.compile(
    checkpointer=st.session_state.memory,
    interrupt_before=["save_report"] # 👈 C'est ici qu'on met le graphe en pause !
)

# Initialisation de la mémoire Streamlit (pour ne pas perdre les données quand on clique)
if "agent_results" not in st.session_state:
    st.session_state.agent_results = None

# BARRE LATÉRALE : Le médecin entre l'ID de l'étude
with st.sidebar:
    st.header("⚙️ Configuration")
    etude_id = st.text_input("ID de l'étude (Orthanc)", value="1fc1a205-400007b2...")
    
# LE BOUTON QUI DÉCLENCHE TON IA
    if st.button("🚀 Lancer l'IA (LangGraph)", type="primary"):
        
        st.markdown("### 🕵️‍♂️ Cerveau de l'Agent en direct")
        log_container = st.empty() # La boîte vide qui va se remplir de tes prints
        
        # 1. On détourne la console vers notre boîte Streamlit
        old_stdout = sys.stdout
        sys.stdout = StreamlitPrintCapture(log_container)
        
        try:
            st.session_state.config={"configurable": {"thread_id": f"session_{etude_id}"}}
            
            inputs = {
                "current_study_id": etude_id, 
                "excel_path": "data/protected-clinical-data.xlsx", 
                "history_texts": "",
                "excel_password":"Trois petits cochons",
                "revision_count": 0,
                "is_human_approved": True 
            } 
            
            # L'agent tourne, et TOUS tes prints s'afficheront en direct à l'écran !
            final_state = medical_agent.invoke(inputs, config=config)
            st.session_state.agent_results = final_state
            
        except Exception as e:
            st.error("❌ ERREUR FATALE DANS L'AGENT LANGGRAPH :")
            st.code(traceback.format_exc(), language="python")
            
        finally:
            # 2. TRÈS IMPORTANT : On rend la console à Python une fois fini (ou si ça plante)
            sys.stdout = old_stdout
            # On sauvegarde le vrai résultat dans la mémoire de Streamlit
            # st.session_state.agent_results = final_state

# AFFICHAGE DYNAMIQUE DES RÉSULTATS
if st.session_state.agent_results:
    res = st.session_state.agent_results
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.header("🔍 Input Analysis")
        st.metric("Patient ID", res.get("patient_id", "N/A"))
        st.write(f"**Constantes :** {res.get('patient_age')} | Sexe: {res.get('patient_sex')}")
        st.info(f"**Findings Spatiales :** {res.get('anomaly_detected')}")
        
    with col2:
        st.header("📝 DRAFT PRELIMINARY REPORT")
        # On charge le vrai brouillon généré par ton graphe
        draft = res.get("draft_report", "")
        
        edited_report = st.text_area("Review & Edit:", value=draft, height=300)
        
        if st.button("✅ VALIDATE & SAVE"):
            with st.spinner("Mise à jour et sauvegarde finale dans le PACS..."):
                
                # 1. On injecte les modifications du médecin dans le State
                medical_agent.update_state(
                    st.session_state.config,
                    {
                        "draft_report": edited_report, 
                        "is_human_approved": True
                    }
                )
                
                # 2. On relance le graphe en lui passant "None". 
                # Il comprend qu'il doit reprendre là où il s'était arrêté (save_report) !
                medical_agent.invoke(None, config=st.session_state.config)
                
            st.balloons()
            st.success("Rapport validé et sauvegardé par l'agent final !")
else:
    st.info("👈 Entrez un ID d'étude à gauche et cliquez sur 'Lancer l'IA' pour commencer l'analyse.")