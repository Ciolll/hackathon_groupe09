



from mistralai import Mistral
import json
from typing import TypedDict, List, Optional
from MedicalAgentState import MedicalAgentState

API_KEY = "I7H1l8yO1g8S0y59ohDxQNaqvSXXuAGm"
client = Mistral(api_key=API_KEY)
    
def review_report(state: MedicalAgentState) -> dict:
    prompt = f"""You are an expert medical reviewer.
    Here is the patient's source data:
    - ID: {state['patient_id']}
    - Age: {state['patient_age']}, Sex: {state['patient_sex']}, Weight: {state['patient_weight']}
    - Current metrics: {state['anomaly_detected']}
    - History: {state['patient_history']}
    
    Here is the generated report:
    {state['draft_report']}
    
    Verify that the report is consistent with the source data and check for:
    1. Spelling and grammar errors
    2. Consistency of measurements with the source data
    3. Report structure (findings, evolution, conclusion)
    4. Missing or contradictory data
    
    Respond ONLY with this JSON structure, with no surrounding text:
    {{"is_valid": "true or false", "feedback": "list of issues or empty if perfect"}}
    """
    response = client.chat.complete(
        model="mistral-small-latest",
        messages=[{"role": "user", "content": prompt}]
    )
    
    content = response.choices[0].message.content
    content = content.replace("```json", "").replace("```", "").strip()
    result = json.loads(content)
    
    return {
        "critic_feedback": result["feedback"],
        "revision_count": state["revision_count"] + 1
    }