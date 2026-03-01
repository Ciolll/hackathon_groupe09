import streamlit as st
from langgraph.graph import StateGraph, START, END
import traceback
from MedicalAgentState import MedicalAgentState
from etape_1 import node_process_current
from fetch_history import fetch_history
from access_orthanc import node_process_orthanc
from comparing import node_clinical_comparison
from agent_rapport import RAGAgent, draft_report
from agent_review import review_report
from savereport import save_report_1
from agent_rcp import consulter_rcp

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
workflow.add_edge("save_report", END)

medical_agent = workflow.compile()

# ==========================================
# INTERFACE STREAMLIT
# ==========================================
st.set_page_config(page_title="Agent Radiology — RCP", layout="wide", page_icon="🏥")
st.title("🏥 AGENT RADIOLOGY : MDT Hub")

if "agent_results" not in st.session_state:
    st.session_state.agent_results = None
if "rcp_response" not in st.session_state:
    st.session_state.rcp_response = None

# ── SIDEBAR ──────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuration")
    etude_id = st.text_input("Study ID (Orthanc)", value="1fc1a205-400007b2...")

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

    # — Agent RCP —
    st.divider()
    st.header("🏛️ MDT Agent (RCP)")

    rcp_col1, rcp_col2 = st.columns([1, 2])

    with rcp_col1:
        st.subheader("⚙️ MDT Configuration")
        patient_id_rcp = st.text_input("Patient ID", value=res.get("patient_id", "0301B7D6"))
        question = st.text_area(
            "Question for the MDT",
            value="What elements suggest local or regional nodal extension?",
            height=100
        )

        if st.button("🧠 Run MDT Analysis", type="primary"):
            with st.spinner("MDT Agent is analyzing the full patient history..."):
                try:
                    rcp_response = consulter_rcp(patient_id_rcp, question)
                    st.session_state.rcp_response = rcp_response
                except Exception as e:
                    st.error("❌ ERROR IN RCP AGENT:")
                    st.code(traceback.format_exc(), language="python")

    with rcp_col2:
        st.subheader("📋 MDT Report")
        if st.session_state.rcp_response:
            st.markdown(st.session_state.rcp_response)
        else:
            st.info("Configure the MDT on the left and click 'Run MDT Analysis'.")

else:
    st.info("👈 Enter a study ID on the left and click 'Run AI Agent' to start the analysis.")