# Hybrid AI System: Classical ML + CNN + RNN + LLM

## Problem Framing

**Business scenario:**
An e-commerce company wants to identify customer complaints that may escalate into serious issues, causing customer churn or reputational damage.

**Prediction target:**
Complaint risk level: low, medium, or high.

**Who will use the output:**
Customer support team managers and escalation desk.

**Business action:**
- Low-risk complaints → Auto-reply with standard resolution steps
- Medium-risk complaints → Manual review by support agent within 4 hours
- High-risk complaints → Immediate escalation to senior support + compensation review

**False positive impact:**
A normal complaint may be unnecessarily escalated, wasting senior support time and potentially over-compensating customers.

**False negative impact:**
A serious complaint may be missed, causing prolonged customer dissatisfaction, potential social media backlash, or customer churn.

**Human review required when:**
- Risk level is medium or high
- Model scores disagree by more than 0.2 (standard deviation > 0.15)
- LLM flags `human_review_required: true`
- Input image quality is poor or text contains ambiguous language

## Setup Instructions

1. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate