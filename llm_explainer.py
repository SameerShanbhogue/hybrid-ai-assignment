import os
import json
import google.generativeai as genai

# Configure Gemini
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

def explain_with_gemini(input_payload):
    """
    Generate structured explanation using Gemini LLM.
    Falls back to rule-based explanation if API key missing.
    """
    if not GEMINI_API_KEY:
        return fallback_explanation(input_payload)
    
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        prompt = f"""You are a business AI decision assistant for an e-commerce customer complaint system.

Given the following input and model outputs, generate a structured JSON response.

Rules:
- Return ONLY valid JSON.
- Do not change the model scores.
- Do not invent facts not present in the complaint.
- If risk_level is high, human_review_required must be true.
- If model scores disagree strongly (standard deviation > 0.15), human_review_required must be true.
- For medium risk, recommend manual_review; for high risk, escalate_immediately; for low risk, auto_reply.

Input:
{json.dumps(input_payload, indent=2)}

Return JSON with these fields:
- summary (string, brief 1-sentence summary of the complaint)
- recommended_action (string: "auto_reply", "manual_review", or "escalate_immediately")
- reason (string, why this action is recommended)
- human_review_required (boolean)
- risk_notes (list of strings, key risk indicators found in the complaint)

Example response:
{{
    "summary": "Customer reports a broken product with no support response.",
    "recommended_action": "escalate_immediately",
    "reason": "High risk score and complaint mentions both product damage and support delay.",
    "human_review_required": true,
    "risk_notes": ["Broken product", "Support delay", "Multiple previous complaints"]
}}
"""
        response = model.generate_content(prompt)
        # Parse JSON from response
        text = response.text.strip()
        # Remove markdown code blocks if present
        if text.startswith("```json"):
            text = text[7:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        result = json.loads(text)
        return result
        
    except Exception as e:
        print(f"LLM error: {e}, using fallback")
        return fallback_explanation(input_payload)

def fallback_explanation(input_payload):
    """Rule-based fallback when Gemini is unavailable"""
    model_outputs = input_payload.get("model_outputs", {})
    risk_level = model_outputs.get("risk_level", "low")
    complaint_text = input_payload.get("complaint_text", "").lower()
    
    # Simple rule-based logic
    risk_notes = []
    if "broken" in complaint_text or "damaged" in complaint_text:
        risk_notes.append("Product damage mentioned")
    if "ignored" in complaint_text or "no response" in complaint_text:
        risk_notes.append("Support delay indicated")
    if input_payload.get("previous_complaints", 0) > 2:
        risk_notes.append("Multiple previous complaints")
    
    if risk_level == "high":
        recommended_action = "escalate_immediately"
        human_review = True
        summary = "High-risk complaint requiring immediate escalation."
        reason = "High risk score and complaint contains escalation indicators."
    elif risk_level == "medium":
        recommended_action = "manual_review"
        human_review = True
        summary = "Medium-risk complaint that needs manual review."
        reason = "Medium risk score, further human assessment recommended."
    else:
        recommended_action = "auto_reply"
        human_review = False
        summary = "Low-risk complaint suitable for automated response."
        reason = "Low risk score, standard auto-reply is appropriate."
    
    return {
        "summary": summary,
        "recommended_action": recommended_action,
        "reason": reason,
        "human_review_required": human_review,
        "risk_notes": risk_notes
    }

if __name__ == "__main__":
    # Test
    test_payload = {
        "complaint_text": "Product broken and support ignored me",
        "customer_tenure": 12,
        "previous_complaints": 2,
        "model_outputs": {
            "classical_score": 0.75,
            "cnn_score": 0.70,
            "rnn_score": 0.82,
            "final_score": 0.76,
            "risk_level": "high"
        }
    }
    result = explain_with_gemini(test_payload)
    print(json.dumps(result, indent=2))