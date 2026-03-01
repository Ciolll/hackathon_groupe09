import os
import pandas as pd
from langchain_mistralai.chat_models import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory
from read_excel import read_excel

# --- 1. CONFIGURATION ---
os.environ["MISTRAL_API_KEY"] = "I7H1l8yO1g8S0y59ohDxQNaqvSXXuAGm"

try:
    df_patients = read_excel("/home/jovyan/work/FINAL/data/protected-clinical-data.xlsx", "Trois petits cochons")
except FileNotFoundError:
    print("Error: Excel file not found.")
    df_patients = pd.DataFrame()

# --- 2. INITIALISATION IA ---
llm = ChatMistralAI(model="mistral-large-latest", temperature=0.1)

# --- 3. MÉMOIRE ---
store = {}

def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

# --- 4. PROMPT ---
ORDRE_FIXE = ["31981427", "57329381", "92106962", "11092835", "11297707"]

system_prompt_staff = """You are a medical assistant participating in a thoracic oncology MDT (Multidisciplinary Team) meeting.
You analyze EXCLUSIVELY the textual content provided. You never invent anything.

STRICT FORMAT (in this order):
1) EVOLUTIONARY CHRONOLOGY
2) ONCOLOGY
3) RADIOLOGY
4) PRIMARY CARE PHYSICIAN
5) POINTS TO DISCUSS / CLARIFY

EVOLUTIONARY CHRONOLOGY
Objective:
- Oral summary at the beginning of the MDT meeting.
- Single narrative voice.

Absolute rules:
- No speaker labels allowed: "Radiologist:", "Oncologist:", "MDT:", or equivalent.
- You must list ALL examinations provided (even if non-informative).
- Examinations are provided in validated chronological order: do not reorder them.
- Do not write "reconstructed order".
- Do not conclude "increase/decrease" by simple number comparison:
  -> Only use "increase", "decrease", "stable" if the report explicitly states it.
  -> Otherwise write: "measured at X (previous Y)" without qualifying.
- Each line must start with "Examination 1/5", "Examination 2/5", etc. + AccessionNumber.
- 6 to 12 lines max. Neutral, factual tone, no conclusion.
- If a numerical measurement changes between two examinations BUT the report describes the lesion as "unchanged":
    → Write explicitly: "WARNING: internal contradiction within the report between the numerical measurement and the qualitative description."
    → Do not qualify the evolution yourself.

Then, specialized sections (4 bullet points max each):
- ONCOLOGY: only local/nodal/metastatic extension described + SD/PD if written.
- RADIOLOGY: only morphological elements described, mention "difficult to measure" if present.
- PRIMARY CARE PHYSICIAN: simple reformulation, risks in conditional tense, no action plan.
- POINTS TO DISCUSS: 3 to 5 key missing elements. If absent: "Not specified in the report".

Always respond in English.
"""

prompt_template = ChatPromptTemplate.from_messages([
    ("system", system_prompt_staff),
    MessagesPlaceholder(variable_name="historique"),
    ("human",
     "VALIDATED CHRONOLOGICAL ORDER (do not modify): 31981427 → 57329381 → 92106962 → 11092835 → 11297707\n\n"
     "Patient data (examination texts, already in this order):\n\n{clinical_info}\n\n"
     "MDT question: {question_medecin}"
    )
])

chain = prompt_template | llm | StrOutputParser()

agent_rcp = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="question_medecin",
    history_messages_key="historique",
)

# --- 5. FONCTION PRINCIPALE ---
def consulter_rcp(patient_id: str, question: str) -> str:
    if df_patients.empty:
        return "Data system unavailable."

    dossier = df_patients[df_patients["PatientID"] == patient_id].copy()
    if dossier.empty:
        return f"Patient record {patient_id} not found."

    dossier["AccessionNumber"] = dossier["AccessionNumber"].astype(str)
    dossier["ordre_staff"] = dossier["AccessionNumber"].apply(
        lambda x: ORDRE_FIXE.index(x) if x in ORDRE_FIXE else 999
    )
    dossier = dossier.sort_values("ordre_staff").reset_index(drop=True)

    total = len(dossier)
    accession_list = " → ".join(dossier["AccessionNumber"].tolist())

    blocks = []
    for i, row in dossier.iterrows():
        acc = row.get("AccessionNumber", "Not specified")
        rep = row.get("Clinical information data (Pseudo reports)", "")
        rep = "" if pd.isna(rep) else str(rep).strip()
        if not rep:
            rep = "Not specified in the report"
        blocks.append(f"EXAMINATION {i+1}/{total} — AccessionNumber: {acc}\n{rep}")

    clinical_info_all = "\n\n---\n\n".join(blocks)

    reponse = agent_rcp.invoke(
        {
            "clinical_info": clinical_info_all,
            "question_medecin": question
        },
        config={"configurable": {"session_id": f"{patient_id}_mdt"}}
    )

    return reponse