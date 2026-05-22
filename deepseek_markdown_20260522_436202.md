# Decision Policy

## Risk Thresholds

**Low risk:** final_score < 0.45
**Medium risk:** 0.45 ≤ final_score < 0.75
**High risk:** final_score ≥ 0.75

## Actions

| Risk Level | Action |
|------------|--------|
| Low | Auto-reply allowed. Send standard resolution steps. No human review required. |
| Medium | Send to manual support review queue. Agent must respond within 4 hours. |
| High | Escalate immediately to senior support. Notify manager. Prioritize within 1 hour. |

## Human Review Rules

Human review is **required** when:
- Risk level is medium or high
- Model scores disagree strongly (standard deviation > 0.15)
- Input text is unclear or contains ambiguous language
- Image quality is poor (detected by CNN confidence < 0.6)
- LLM marks `human_review_required` as true
- Customer has > 5 previous complaints regardless of score

## LLM Restrictions

The LLM may:
- Summarize the complaint
- Explain the reasoning behind the recommendation
- Suggest a business action (auto_reply, manual_review, escalate_immediately)
- Flag human review requirement

The LLM may **NOT**:
- Override or change the model scores
- Make final refund, compensation, or legal decisions
- Reject a claim without human oversight
- Make hiring, credit, or medical decisions
- Approve discounts > $50 without manager approval