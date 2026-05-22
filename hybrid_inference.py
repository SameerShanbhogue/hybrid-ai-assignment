import numpy as np
import joblib
import tensorflow as tf
from sklearn.feature_extraction.text import TfidfVectorizer
from llm_explainer import explain_with_gemini

# Load models globally for efficiency
classical_model_data = joblib.load('classical_model.pkl')
classical_clf = classical_model_data['classifier']
tfidf_vectorizer = classical_model_data['tfidf']

cnn_model = tf.keras.models.load_model('cnn_model.keras')
rnn_model = tf.keras.models.load_model('rnn_model.keras')

def get_risk_score_from_probs(probs):
    """
    Convert class probabilities to a continuous risk score (0-1).
    probs: [p_low, p_medium, p_high]
    Returns: p_medium*0.5 + p_high*1.0
    """
    return probs[1] * 0.5 + probs[2] * 1.0

def map_risk(score):
    if score >= 0.75:
        return "high"
    elif score >= 0.45:
        return "medium"
    else:
        return "low"

def predict_hybrid(input_data):
    """
    input_data: dict with keys:
        - 'text': complaint text (string)
        - 'tabular': dict with 'customer_tenure' and 'previous_complaints'
        - 'text_sequence': complaint text (string, same as above)
        - 'image_available': bool (if True, use CNN, else default score)
        - (optional) 'image_array': numpy array of shape (64,64,3) if available
    """
    
    # Extract features
    complaint_text = input_data.get('text', '')
    tenure = input_data.get('tabular', {}).get('customer_tenure', 0)
    prev_complaints = input_data.get('tabular', {}).get('previous_complaints', 0)
    
    # 1. Classical ML score
    text_tfidf = tfidf_vectorizer.transform([complaint_text]).toarray()
    tabular_features = np.array([[tenure, prev_complaints]])
    classical_features = np.hstack([text_tfidf, tabular_features])
    classical_probs = classical_clf.predict_proba(classical_features)[0]
    classical_score = get_risk_score_from_probs(classical_probs)
    
    # 2. CNN score
    if input_data.get('image_available', False) and 'image_array' in input_data:
        img = input_data['image_array']
        if img.max() > 1:
            img = img.astype('float32') / 255.0
        img = np.expand_dims(img, axis=0)
        cnn_probs = cnn_model.predict(img, verbose=0)[0]
        cnn_score = get_risk_score_from_probs(cnn_probs)
    else:
        # Fallback: use classical score as proxy
        cnn_score = classical_score
    
    # 3. RNN score
    rnn_probs = rnn_model.predict([complaint_text], verbose=0)[0]
    rnn_score = get_risk_score_from_probs(rnn_probs)
    
    # 4. Hybrid scoring
    final_score = 0.4 * classical_score + 0.3 * cnn_score + 0.3 * rnn_score
    risk_level = map_risk(final_score)
    
    model_outputs = {
        "classical_score": round(classical_score, 4),
        "cnn_score": round(cnn_score, 4),
        "rnn_score": round(rnn_score, 4),
        "final_score": round(final_score, 4),
        "risk_level": risk_level
    }
    
    # 5. LLM explanation
    llm_payload = {
        "complaint_text": complaint_text,
        "customer_tenure": tenure,
        "previous_complaints": prev_complaints,
        "model_outputs": model_outputs
    }
    llm_explanation = explain_with_gemini(llm_payload)
    
    return {
        "model_outputs": model_outputs,
        "llm_explanation": llm_explanation
    }

# Example usage
if __name__ == "__main__":
    test_input = {
        "text": "The product arrived broken and support has ignored me for five days.",
        "tabular": {"customer_tenure": 14, "previous_complaints": 3},
        "text_sequence": "The product arrived broken and support has ignored me for five days.",
        "image_available": False
    }
    result = predict_hybrid(test_input)
    print(result)