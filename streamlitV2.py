import streamlit as st
from langgraph.graph import StateGraph, START, END
import traceback
from MedicalAgentState import MedicalAgentState
from etape_1 import node_process_current
from fetch_history import fetch_history
from access_orthanc import node_process_orthanc
from comparing import node_clinical_comparison
from RAG import prepare_documents
from agent_rapport import RAGAgent, draft_report
from agent_review import review_report
from savereport import save_report_1
from jumeau_numerique import node_digital_twin

# ==========================================
# GRAPH LANGGRAPH
# ==========================================
def route_after_review(state: MedicalAgentState) -> str:
    feedback = state.get("critic_feedback", "")
    retries = state.get("revision_count", 0)
    if feedback != "OK" and retries < 3:
        return "draft_report"
    return "save_report"

def route_after_history(state: MedicalAgentState) -> str:
    historique_excel = state.get("patient_history", "")
    if not historique_excel:
        return "draft_report"
    return "access_orthanc"

workflow = StateGraph(MedicalAgentState)
workflow.add_node("process_current", node_process_current)
workflow.add_node("fetch_history", fetch_history)
workflow.add_node("access_orthanc", node_process_orthanc)
workflow.add_node("comparison", node_clinical_comparison)
workflow.add_node("draft_report", draft_report)
workflow.add_node("review_report", review_report)
workflow.add_node("save_report", save_report_1)
workflow.add_node("digital_twin", node_digital_twin)  # ← NOUVEAU

workflow.add_edge(START, "process_current")
workflow.add_edge("process_current", "fetch_history")
workflow.add_conditional_edges("fetch_history", route_after_history, {
    "draft_report": "draft_report",
    "access_orthanc": "access_orthanc"
})
workflow.add_edge("access_orthanc", "comparison")
workflow.add_edge("comparison", "draft_report")
workflow.add_edge("draft_report", "review_report")
workflow.add_conditional_edges("review_report", route_after_review, {
    "draft_report": "draft_report",
    "save_report": "save_report"
})
workflow.add_edge("save_report", "digital_twin")  # ← NOUVEAU
workflow.add_edge("digital_twin", END)             # ← NOUVEAU

medical_agent = workflow.compile()

# ==========================================
# INTERFACE STREAMLIT
# ==========================================
st.set_page_config(page_title="Agent Radiology", layout="wide", page_icon="🏥")
st.title("🏥 AGENT RADIOLOGY : Multi-Agent Hub")

if "agent_results" not in st.session_state:
    st.session_state.agent_results = None

# ── SIDEBAR ──────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuration")
    etude_id = st.text_input("ID de l'étude (Orthanc)", value="1fc1a205-400007b2...")

    st.divider()
    st.header("🧬 Profil Patient")
    tabac        = st.selectbox("Tabagisme", ["Non", "Ancien", "Actif"])
    genetique    = st.selectbox("Antécédents génétiques cancer ?", ["Non", "Oui"])
    stress       = st.slider("Niveau de stress (1-10)", 1, 10, 5)
    sport        = st.selectbox("Activité physique", ["Sédentaire", "Modérée", "Régulière"])
    alimentation = st.selectbox("Alimentation", ["Déséquilibrée", "Correcte", "Saine"])

    st.divider()
    if st.button("🚀 Lancer l'IA (LangGraph)", type="primary"):
        with st.spinner("L'agent analyse le CT Scan et l'historique..."):
            try:
                config = {"configurable": {"thread_id": f"session_{etude_id}"}}
                inputs = {
                    "current_study_id": etude_id,
                    "excel_path": "clinical/protected-clinical-data.xlsx",
                    "history_texts": "",
                    "revision_count": 0,
                    "is_human_approved": True,
                    "lifestyle_data": {
                        "tabac": tabac,
                        "genetique": genetique,
                        "stress": stress,
                        "sport": sport,
                        "alimentation": alimentation
                    }
                }
                final_state = medical_agent.invoke(inputs, config=config)
                st.session_state.agent_results = final_state

            except Exception as e:
                st.error("❌ ERREUR FATALE DANS L'AGENT LANGGRAPH :")
                st.code(traceback.format_exc(), language="python")

# ── RÉSULTATS ─────────────────────────────
if st.session_state.agent_results:
    res = st.session_state.agent_results

    # — Rapport Radiologique —
    col1, col2 = st.columns([1, 2])

    with col1:
        st.header("🔍 Input Analysis")
        st.metric("Patient ID", res.get("patient_id", "N/A"))
        st.write(f"**Constantes :** {res.get('patient_age')} | Sexe: {res.get('patient_sex')}")
        st.info(f"**Findings Spatiales :** {res.get('anomaly_detected')}")

    with col2:
        st.header("📝 DRAFT PRELIMINARY REPORT")
        draft = res.get("draft_report", "")
        edited_report = st.text_area("Review & Edit:", value=draft, height=300)
        if st.button("✅ VALIDATE & SAVE"):
            st.balloons()
            st.success("Rapport validé !")

    # — Jumeau Numérique —
    st.divider()
    st.header("🧬 Jumeau Numérique")

    twin_col1, twin_col2 = st.columns([1, 2])

    with twin_col1:
        st.subheader("📈 Projections")
        projections = res.get("twin_projections", {})
        if projections:
            st.metric("Volume actuel", f"{projections.get('V0')} mm³")
            st.metric("Dans 3 mois",   f"{projections.get('3_mois')} mm³")
            st.metric("Dans 6 mois",   f"{projections.get('6_mois')} mm³")

    with twin_col2:
        st.subheader("🤖 Analyse IA")
        twin_report = res.get("twin_analysis_report", "")
        if twin_report:
            if "PARTIE 2" in twin_report:
                parts = twin_report.split("PARTIE 2")
                with st.expander("👨‍⚕️ Pour le staff médical (RCP)", expanded=True):
                    st.markdown(parts[0].replace("PARTIE 1 :", "").strip())
                with st.expander("🙋 Pour le patient", expanded=True):
                    st.markdown("PARTIE 2" + parts[1])
            else:
                st.markdown(twin_report)

else:
    st.info("👈 Entrez un ID d'étude à gauche et cliquez sur 'Lancer l'IA' pour commencer l'analyse.")











###########












import streamlit as st
from langgraph.graph import StateGraph, START, END
import traceback
from MedicalAgentState import MedicalAgentState
from etape_1 import node_process_current
from fetch_history import fetch_history
from access_orthanc import node_process_orthanc
from comparing import node_clinical_comparison
from RAG import prepare_documents
from agent_rapport import RAGAgent, draft_report
from agent_review import review_report
from savereport import save_report_1
from jumeau_numerique import node_digital_twin

# ==========================================
# GRAPH LANGGRAPH
# ==========================================
def route_after_review(state: MedicalAgentState) -> str:
    feedback = state.get("critic_feedback", "")
    retries = state.get("revision_count", 0)
    if feedback != "OK" and retries < 3:
        return "draft_report"
    return "save_report"

def route_after_history(state: MedicalAgentState) -> str:
    historique_excel = state.get("patient_history", "")
    if not historique_excel:
        return "draft_report"
    return "access_orthanc"

workflow = StateGraph(MedicalAgentState)
workflow.add_node("process_current", node_process_current)
workflow.add_node("fetch_history", fetch_history)
workflow.add_node("access_orthanc", node_process_orthanc)
workflow.add_node("comparison", node_clinical_comparison)
workflow.add_node("draft_report", draft_report)
workflow.add_node("review_report", review_report)
workflow.add_node("save_report", save_report_1)
workflow.add_node("digital_twin", node_digital_twin)

workflow.add_edge(START, "process_current")
workflow.add_edge("process_current", "fetch_history")
workflow.add_conditional_edges("fetch_history", route_after_history, {
    "draft_report": "draft_report",
    "access_orthanc": "access_orthanc"
})
workflow.add_edge("access_orthanc", "comparison")
workflow.add_edge("comparison", "draft_report")
workflow.add_edge("draft_report", "review_report")
workflow.add_conditional_edges("review_report", route_after_review, {
    "draft_report": "draft_report",
    "save_report": "save_report"
})
workflow.add_edge("save_report", "digital_twin")
workflow.add_edge("digital_twin", END)

medical_agent = workflow.compile()

# ==========================================
# INTERFACE STREAMLIT
# ==========================================
st.set_page_config(page_title="Agent Radiology", layout="wide", page_icon="🏥")
st.title("🏥 AGENT RADIOLOGY : Multi-Agent Hub")

if "agent_results" not in st.session_state:
    st.session_state.agent_results = None

# ── SIDEBAR ──────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuration")
    etude_id = st.text_input("ID de l'étude (Orthanc)", value="1fc1a205-400007b2...")

    st.divider()
    st.header("🧬 Profil Patient")
    tabac        = st.selectbox("Tabagisme", ["Non", "Ancien", "Actif"])
    genetique    = st.selectbox("Antécédents génétiques cancer ?", ["Non", "Oui"])
    stress       = st.slider("Niveau de stress (1-10)", 1, 10, 5)
    sport        = st.selectbox("Activité physique", ["Sédentaire", "Modérée", "Régulière"])
    alimentation = st.selectbox("Alimentation", ["Déséquilibrée", "Correcte", "Saine"])

    st.divider()
    if st.button("🚀 Lancer l'IA (LangGraph)", type="primary"):
        with st.spinner("L'agent analyse le CT Scan et l'historique..."):
            try:
                config = {"configurable": {"thread_id": f"session_{etude_id}"}}
                inputs = {
                    "current_study_id": etude_id,
                    "excel_path": "/home/jovyan/work/FINAL/clinical/protected-clinical-data.xlsx",  # ← CORRIGÉ
                    "history_texts": "",
                    "revision_count": 0,
                    "is_human_approved": True,
                    "lifestyle_data": {
                        "tabac": tabac,
                        "genetique": genetique,
                        "stress": stress,
                        "sport": sport,
                        "alimentation": alimentation
                    }
                }
                final_state = medical_agent.invoke(inputs, config=config)
                st.session_state.agent_results = final_state

            except Exception as e:
                st.error("❌ ERREUR FATALE DANS L'AGENT LANGGRAPH :")
                st.code(traceback.format_exc(), language="python")

# ── RÉSULTATS ─────────────────────────────
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
        draft = res.get("draft_report", "")
        edited_report = st.text_area("Review & Edit:", value=draft, height=300)
        if st.button("✅ VALIDATE & SAVE"):
            st.balloons()
            st.success("Rapport validé !")

    # — Jumeau Numérique —
    st.divider()
    st.header("🧬 Jumeau Numérique")

    twin_col1, twin_col2 = st.columns([1, 2])

    with twin_col1:
        st.subheader("📈 Projections")
        projections = res.get("twin_projections", {})
        if projections:
            st.metric("Volume actuel", f"{projections.get('V0')} mm³")
            st.metric("Dans 3 mois",   f"{projections.get('3_mois')} mm³")
            st.metric("Dans 6 mois",   f"{projections.get('6_mois')} mm³")

    with twin_col2:
        st.subheader("🤖 Analyse IA")
        twin_report = res.get("twin_analysis_report", "")
        if twin_report:
            if "PARTIE 2" in twin_report:
                parts = twin_report.split("PARTIE 2")
                with st.expander("👨‍⚕️ Pour le staff médical (RCP)", expanded=True):
                    st.markdown(parts[0].replace("PARTIE 1 :", "").strip())
                with st.expander("🙋 Pour le patient", expanded=True):
                    st.markdown("PARTIE 2" + parts[1])
            else:
                st.markdown(twin_report)

else:
    st.info("👈 Entrez un ID d'étude à gauche et cliquez sur 'Lancer l'IA' pour commencer l'analyse.")




#########


import streamlit as st
from langgraph.graph import StateGraph, START, END
import traceback
from MedicalAgentState import MedicalAgentState
from etape_1 import node_process_current
from fetch_history import fetch_history
from access_orthanc import node_process_orthanc
from comparing import node_clinical_comparison
from RAG import prepare_documents
from agent_rapport import RAGAgent, draft_report
from agent_review import review_report
from savereport import save_report_1
from jumeau_numerique import node_digital_twin

# ==========================================
# GRAPH LANGGRAPH
# ==========================================
def route_after_review(state: MedicalAgentState) -> str:
    feedback = state.get("critic_feedback", "")
    retries = state.get("revision_count", 0)
    if feedback != "OK" and retries < 3:
        return "draft_report"
    return "save_report"

def route_after_history(state: MedicalAgentState) -> str:
    historique_excel = state.get("patient_history", "")
    if not historique_excel:
        return "draft_report"
    return "access_orthanc"

workflow = StateGraph(MedicalAgentState)
workflow.add_node("process_current", node_process_current)
workflow.add_node("fetch_history", fetch_history)
workflow.add_node("access_orthanc", node_process_orthanc)
workflow.add_node("comparison", node_clinical_comparison)
workflow.add_node("draft_report", draft_report)
workflow.add_node("review_report", review_report)
workflow.add_node("save_report", save_report_1)
workflow.add_node("digital_twin", node_digital_twin)

workflow.add_edge(START, "process_current")
workflow.add_edge("process_current", "fetch_history")
workflow.add_conditional_edges("fetch_history", route_after_history, {
    "draft_report": "draft_report",
    "access_orthanc": "access_orthanc"
})
workflow.add_edge("access_orthanc", "comparison")
workflow.add_edge("comparison", "draft_report")
workflow.add_edge("draft_report", "review_report")
workflow.add_conditional_edges("review_report", route_after_review, {
    "draft_report": "draft_report",
    "save_report": "save_report"
})
workflow.add_edge("save_report", "digital_twin")
workflow.add_edge("digital_twin", END)

medical_agent = workflow.compile()

# ==========================================
# INTERFACE STREAMLIT
# ==========================================
st.set_page_config(page_title="Agent Radiology", layout="wide", page_icon="🏥")
st.title("🏥 AGENT RADIOLOGY : Multi-Agent Hub")

if "agent_results" not in st.session_state:
    st.session_state.agent_results = None

# ── SIDEBAR ──────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuration")
    etude_id = st.text_input("ID de l'étude (Orthanc)", value="1fc1a205-400007b2...")

    st.divider()
    st.header("🧬 Profil Patient")
    tabac        = st.selectbox("Tabagisme", ["Non", "Ancien", "Actif"])
    genetique    = st.selectbox("Antécédents génétiques cancer ?", ["Non", "Oui"])
    stress       = st.slider("Niveau de stress (1-10)", 1, 10, 5)
    sport        = st.selectbox("Activité physique", ["Sédentaire", "Modérée", "Régulière"])
    alimentation = st.selectbox("Alimentation", ["Déséquilibrée", "Correcte", "Saine"])

    st.divider()
    if st.button("🚀 Lancer l'IA (LangGraph)", type="primary"):
        with st.spinner("L'agent analyse le CT Scan et l'historique..."):
            try:
                config = {"configurable": {"thread_id": f"session_{etude_id}"}}
                inputs = {
                    "current_study_id": etude_id,
                    "excel_path": "/home/jovyan/work/FINAL/data/protected-clinical-data.xlsx",
                    "excel_password": "Trois petits cochons",  # ← AJOUTÉ
                    "history_texts": "",
                    "revision_count": 0,
                    "is_human_approved": True,
                    "lifestyle_data": {
                        "tabac": tabac,
                        "genetique": genetique,
                        "stress": stress,
                        "sport": sport,
                        "alimentation": alimentation
                    }
                }
                final_state = medical_agent.invoke(inputs, config=config)
                st.session_state.agent_results = final_state

            except Exception as e:
                st.error("❌ ERREUR FATALE DANS L'AGENT LANGGRAPH :")
                st.code(traceback.format_exc(), language="python")

# ── RÉSULTATS ─────────────────────────────
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
        draft = res.get("draft_report", "")
        edited_report = st.text_area("Review & Edit:", value=draft, height=300)
        if st.button("✅ VALIDATE & SAVE"):
            st.balloons()
            st.success("Rapport validé !")

    # — Jumeau Numérique —
    st.divider()
    st.header("🧬 Jumeau Numérique")

    twin_col1, twin_col2 = st.columns([1, 2])

    with twin_col1:
        st.subheader("📈 Projections")
        projections = res.get("twin_projections", {})
        if projections:
            st.metric("Volume actuel", f"{projections.get('V0')} mm³")
            st.metric("Dans 3 mois",   f"{projections.get('3_mois')} mm³")
            st.metric("Dans 6 mois",   f"{projections.get('6_mois')} mm³")

    with twin_col2:
        st.subheader("🤖 Analyse IA")
        twin_report = res.get("twin_analysis_report", "")
        if twin_report:
            if "PARTIE 2" in twin_report:
                parts = twin_report.split("PARTIE 2")
                with st.expander("👨‍⚕️ Pour le staff médical (RCP)", expanded=True):
                    st.markdown(parts[0].replace("PARTIE 1 :", "").strip())
                with st.expander("🙋 Pour le patient", expanded=True):
                    st.markdown("PARTIE 2" + parts[1])
            else:
                st.markdown(twin_report)

else:
    st.info("👈 Entrez un ID d'étude à gauche et cliquez sur 'Lancer l'IA' pour commencer l'analyse.")"""





import streamlit as st
from langgraph.graph import StateGraph, START, END
import traceback
from MedicalAgentState import MedicalAgentState
from etape_1 import node_process_current
from fetch_history import fetch_history
from access_orthanc import node_process_orthanc
from comparing import node_clinical_comparison
from RAG import prepare_documents
from agent_rapport import RAGAgent, draft_report
from agent_review import review_report
from savereport import save_report_1
from jumeau_numerique import node_digital_twin

# ==========================================
# GRAPH LANGGRAPH
# ==========================================
def route_after_review(state: MedicalAgentState) -> str:
    feedback = state.get("critic_feedback", "")
    retries = state.get("revision_count", 0)
    if feedback != "OK" and retries < 3:
        return "draft_report"
    return "save_report"

def route_after_history(state: MedicalAgentState) -> str:
    historique_excel = state.get("patient_history", "")
    if not historique_excel:
        return "draft_report"
    return "access_orthanc"

workflow = StateGraph(MedicalAgentState)
workflow.add_node("process_current", node_process_current)
workflow.add_node("fetch_history", fetch_history)
workflow.add_node("access_orthanc", node_process_orthanc)
workflow.add_node("comparison", node_clinical_comparison)
workflow.add_node("draft_report", draft_report)
workflow.add_node("review_report", review_report)
workflow.add_node("save_report", save_report_1)
workflow.add_node("digital_twin", node_digital_twin)

workflow.add_edge(START, "process_current")
workflow.add_edge("process_current", "fetch_history")
workflow.add_conditional_edges("fetch_history", route_after_history, {
    "draft_report": "draft_report",
    "access_orthanc": "access_orthanc"
})
workflow.add_edge("access_orthanc", "comparison")
workflow.add_edge("comparison", "draft_report")
workflow.add_edge("draft_report", "review_report")
workflow.add_conditional_edges("review_report", route_after_review, {
    "draft_report": "draft_report",
    "save_report": "save_report"
})
workflow.add_edge("save_report", "digital_twin")
workflow.add_edge("digital_twin", END)

medical_agent = workflow.compile()

# ==========================================
# INTERFACE STREAMLIT
# ==========================================
st.set_page_config(page_title="Agent Radiology", layout="wide", page_icon="🏥")
st.title("🏥 AGENT RADIOLOGY : Multi-Agent Hub")

if "agent_results" not in st.session_state:
    st.session_state.agent_results = None

# ── SIDEBAR ──────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuration")
    etude_id = st.text_input("Study ID (Orthanc)", value="1fc1a205-400007b2...")

    st.divider()
    st.header("🧬 Patient Profile")
    tabac        = st.selectbox("Smoking", ["No", "Former", "Active"])
    genetique    = st.selectbox("Family history of cancer?", ["No", "Yes"])
    stress       = st.slider("Stress level (1-10)", 1, 10, 5)
    sport        = st.selectbox("Physical activity", ["Sedentary", "Moderate", "Regular"])
    alimentation = st.selectbox("Diet", ["Poor", "Average", "Healthy"])

    st.divider()
    if st.button("🚀 Run AI Agent (LangGraph)", type="primary"):
        with st.spinner("Agent is analyzing the CT Scan and patient history..."):
            try:
                config = {"configurable": {"thread_id": f"session_{etude_id}"}}
                inputs = {
                    "current_study_id": etude_id,
                    "excel_path": "/home/jovyan/work/FINAL/data/protected-clinical-data.xlsx",
                    "excel_password": "Trois petits cochons",
                    "history_texts": "",
                    "revision_count": 0,
                    "is_human_approved": True,
                    "lifestyle_data": {
                        "tabac": tabac,
                        "genetique": genetique,
                        "stress": stress,
                        "sport": sport,
                        "alimentation": alimentation
                    }
                }
                final_state = medical_agent.invoke(inputs, config=config)
                st.session_state.agent_results = final_state

            except Exception as e:
                st.error("❌ FATAL ERROR IN LANGGRAPH AGENT:")
                st.code(traceback.format_exc(), language="python")

# ── RÉSULTATS ─────────────────────────────
if st.session_state.agent_results:
    res = st.session_state.agent_results

    col1, col2 = st.columns([1, 2])

    with col1:
        st.header("🔍 Patient Analysis")
        st.metric("Patient ID", res.get("patient_id", "N/A"))
        st.write(f"**Age:** {res.get('patient_age')} | **Sex:** {res.get('patient_sex')}")
        st.info(f"**Spatial Findings:** {res.get('anomaly_detected')}")

    with col2:
        st.header("📝 Preliminary Radiology Report")
        draft = res.get("draft_report", "")
        edited_report = st.text_area("Review & Edit:", value=draft, height=300)
        if st.button("✅ Validate & Save"):
            st.balloons()
            st.success("Report validated!")

    # — Digital Twin —
    st.divider()
    st.header("🧬 Digital Twin")

    twin_col1, twin_col2 = st.columns([1, 2])

    with twin_col1:
        st.subheader("📈 Projections")
        projections = res.get("twin_projections", {})
        if projections:
            st.metric("Current volume", f"{projections.get('V0')} mm³")
            st.metric("In 3 months",    f"{projections.get('3_mois')} mm³")
            st.metric("In 6 months",    f"{projections.get('6_mois')} mm³")

    with twin_col2:
        st.subheader("🤖 AI Analysis")
        twin_report = res.get("twin_analysis_report", "")
        if twin_report:
            if "PART 2" in twin_report:
                parts = twin_report.split("PART 2")
                with st.expander("👨‍⚕️ For the medical staff (MDT)", expanded=True):
                    st.markdown(parts[0].replace("PART 1:", "").strip())
                with st.expander("🙋 For the patient", expanded=True):
                    st.markdown("PART 2" + parts[1])
            else:
                st.markdown(twin_report)

else:
    st.info("👈 Enter a study ID and fill in the patient profile on the left, then click 'Run AI Agent'.")