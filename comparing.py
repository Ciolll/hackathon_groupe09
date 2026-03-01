from mistralai import Mistral

import MedicalAgentState

# Configuration du client Mistral
API_KEY = "I7H1l8yO1g8S0y59ohDxQNaqvSXXuAGm" 
client = Mistral(api_key=API_KEY)

def node_clinical_comparison(state: MedicalAgentState) -> dict:
    print("\n▶️ Node 4 : Cinical Comparison (Reports vs Computer vision)")
    
    # 1. Récupération des données du State
    rapports_humains = "\n".join(state.get("patient_history", "Aucun rapport précédent disponible."))
    ia_passee = state.get("past_metrics", "no pictures analyzed.")
    ia_actuelle = state.get("current_metrics", "No current data.")
    
    # 2. Le Prompt d'analyse croisée
    prompt = f"""You are an expert radiologist specializing in data reconciliation.
Your mission is to compare the number of nodules mentioned in previous human reports with the number of nodules detected by our algorithm (AI) on the prior images, and then establish a link with today's exam.

--- AVAILABLE DATA ---
PREVIOUS HUMAN REPORT(S):
{rapports_humains}

RETROSPECTIVE ANALYSIS (On prior images):
{ia_passee}

CURRENT ANALYSIS (Today):
{ia_actuelle}

--- ANALYSIS RULES ---
1. COUNTING: Strictly compare the number of nodules found by the AI in the past with the number described by the radiologist in the past.
2. RETROSPECTIVE DETECTION: If the AI found nodules on the prior image that the old human report did not mention, explicitly write: "Retrospective detection: the algorithm identified [X] nodules on the prior imaging, whereas the previous report only mentioned [Y]."

Write a direct and professional synthesis using bullet points. Do not include an introduction; return only the useful text for the final report.
"""
    
    try:
        # On utilise temperature=0 pour forcer le LLM à être factuel et bon en mathématiques
        response = client.chat.complete(
            model="mistral-medium-latest", 
            temperature=0.0,
            messages=[{"role": "user", "content": prompt}]
        )
        
        analyse_critique = response.choices[0].message.content.strip()
        
    except Exception as e:
        print(f"  ⚠️ Failure : {e}")
        analyse_critique = "Automatic Comparison failed."

    print(" Synthèse comparative générée.")
    print(analyse_critique)
    # 5. Mise à jour de l'état LangGraph
    return {"clinical_comparison": analyse_critique}