# Mini Model Card

## System Name
Hybrid Complaint Escalation Risk System v1.0

## Intended Use
Predict escalation risk (low/medium/high) for e-commerce customer complaints to automate triage and prioritize support responses.

## Not Intended For
- Medical or healthcare complaints
- Legal arbitration or binding decisions
- Financial fraud detection
- Real-time safety-critical systems
- Complaints requiring immediate physical action

## Model Components

**Classical ML:** Random Forest with TF-IDF (2000 features) + tabular features (tenure, previous complaints)

**CNN:** 3-conv-layer network (32/64/128 filters) trained on 64x64 synthetic RGB images

**RNN / LSTM:** LSTM with 64 units, embedding dimension 64, trained on complaint text

**LLM:** Google Gemini 1.5 Flash (fallback: rule-based)

## Input Data

**Tabular inputs:** customer_tenure (months), previous_complaints (integer)

**Text inputs:** complaint_text (free text, English only in training)

**Image inputs:** 64x64 RGB image (synthetic product/condition image)

**Sequence inputs:** same as complaint_text, tokenized to max 100 tokens

## Output

- Risk score (0-1 continuous)
- Risk level (low/medium/high)
- Recommended action (auto_reply/manual_review/escalate_immediately)
- Explanation summary and risk notes

## Metrics

| Component | Accuracy | F1 (weighted) |
|-----------|----------|---------------|
| Classical ML | 0.78 | 0.77 |
| CNN | 0.74 | 0.73 |
| RNN | 0.81 | 0.80 |
| Hybrid | 0.83 | 0.82 |
| LLM JSON validity | 100% (with valid API key) | - |

## Threshold Policy

- Low: score < 0.45
- Medium: 0.45 ≤ score < 0.75
- High: score ≥ 0.75

## Known Failure Modes

- Sarcastic complaint text ("Great product, it broke in 2 days")
- Poor quality images (dark, blurry, non-product)
- Biased labels (if synthetic distribution differs from real)
- Model disagreement on ambiguous cases
- LLM hallucination of risk_notes (5% of cases)
- Out-of-distribution input (non-English text, unusual formats)

## Human Review Rules

Human review required when risk is medium/high, model scores disagree, or LLM flags human review. All high-risk predictions must be reviewed within 1 hour.

## Deployment Assumptions

- **Runtime:** < 500ms per inference (CPU), < 200ms with GPU
- **API:** REST endpoint accepting JSON, returning JSON
- **Input schema:** complaint_text (required), image_base64 (optional), tenure, prev_complaints
- **Privacy:** PII (customer_id) should be logged separately, not stored in inference logs
- **Latency SLA:** 95th percentile < 1 second

## Monitoring Suggestions

- Prediction distribution shift (risk levels over time)
- Invalid LLM JSON rate (alert if > 5%)
- Model disagreement rate (alert if > 20%)
- Escalation rate by customer segment
- Human override rate (when system says low but human escalates)
- End-to-end latency percentiles