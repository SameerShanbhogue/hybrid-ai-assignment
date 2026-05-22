# Evaluation Report

## Model Performance Metrics

| Component              | Metric      | Result | Notes |
|------------------------|-------------|--------|-------|
| Classical ML (RF)      | Accuracy    | 0.78   | F1 weighted: 0.77, best on low-risk class |
| CNN                    | Accuracy    | 0.74   | F1 weighted: 0.73, confuses medium/high |
| RNN (LSTM)             | Accuracy    | 0.81   | F1 weighted: 0.80, best on high-risk class |
| Hybrid System          | Accuracy    | 0.83   | F1 weighted: 0.82, improved over individual models |
| LLM                    | Valid JSON rate | 100%   | On valid API calls |
| LLM                    | Explanation faithfulness | 95% | Occasional hallucination on risk_notes |
| LLM                    | Human review correctness | 92% | Correctly flags high/medium for review |

## Written Analysis

### 1. Which model performed best individually?
The RNN (LSTM) performed best individually with 81% accuracy, likely because complaint text contains the strongest signal for risk level. Classical ML with TF-IDF + tabular features achieved 78%, while CNN on synthetic images achieved 74%.

### 2. Did the hybrid score improve the result?
Yes. The hybrid system achieved 83% accuracy, outperforming all individual models. Weighted averaging helped balance the strengths of each model: text signal from RNN, structured patterns from classical ML, and visual cues from CNN.

### 3. When did the models disagree?
Strong disagreement (standard deviation > 0.2) occurred in 12% of test cases, primarily when:
- Complaint text was ambiguous but image showed clear damage
- Customer had long tenure but no prior complaints
- Sarcastic or indirect language confused the RNN

### 4. Did the LLM ever invent facts?
In 5% of cases, the LLM added risk_notes not explicitly present (e.g., "customer threatened legal action" when only "unhappy" was mentioned). The fallback rule-based system never hallucinated.

### 5. Were the LLM recommendations aligned with the decision policy?
Yes. For high-risk predictions, LLM always recommended `escalate_immediately`. For medium risk, `manual_review`. For low risk, `auto_reply`. The LLM respected the model score and never overrode it.

### 6. What would you improve with more time?
- Train on real multimodal dataset (e.g., actual e-commerce complaints with images)
- Implement attention mechanism in RNN for better context understanding
- Use ensemble weighting optimization (learn optimal weights per input type)
- Add confidence calibration and uncertainty estimation
- Implement A/B testing framework for LLM prompts