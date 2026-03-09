# Compliance & Regulatory Alignment

## Standards Implemented

### SEC Regulation Best Interest (Reg BI)
- Suitability checks run on every session via `PolicyEngine`
- Risk profile enforced: concentration limits vary by `RiskTolerance`
- All responses include jurisdiction-specific disclaimers
- No specific investment recommendations without suitability caveats

### FINRA Rule 2111 (Suitability)
- User `risk_tolerance` drives all concentration and sector limits
- `PolicyEngine.run_pre_trade_checks()` sweeps every proposed action
- KYC clearance required before regulated actions

### NIST AI Risk Management Framework
- **Govern**: Policy rules version-controlled in `config/policy_rules.json`
- **Map**: All agent outputs include `reasoning_chain` and `data_sources`
- **Measure**: Shadow eval scoring on 10% of traffic; confidence thresholds enforced
- **Manage**: HITL gate blocks low-confidence/high-risk outputs; meta-critic reviews all

### OWASP LLM Top 10
- **LLM01 Prompt Injection**: PII redaction before all LLM calls; system prompts fixed
- **LLM02 Insecure Output**: Meta-critic validates all agent outputs before synthesis
- **LLM06 Sensitive Info Disclosure**: 7-pattern PII redaction; name minimisation
- **LLM08 Excessive Agency**: HITL gate for all high-value / low-confidence actions
- **LLM09 Overreliance**: Confidence scores surfaced; low confidence routes to human

### SOC 2 Type II
- Audit trail: SHA-256 hash-chained, append-only
- Chain integrity verified at end of every session
- `AuditTrail.export()` produces WORM-ready JSON for external storage
- Access control via RBAC (`security/rbac.py`)

### GDPR / CCPA
- PII minimisation: names stored as first name + last initial only
- PII redaction on all LLM inputs
- `right_to_erasure` hook: delete by `user_id` (implement in persistence layer)

## Compliance Checklist for Production Deployment

- [ ] Replace JWT stub with real IdP integration (`security/rbac.py`)
- [ ] Set `ALPHA_VANTAGE_API_KEY` for true dual-source verification
- [ ] Configure `AWS_S3_AUDIT_BUCKET` for WORM audit log persistence
- [ ] Set `HITL_WEBHOOK_URL` for real ticketing system integration
- [ ] Run `safety check -r requirements.txt` before every release
- [ ] Complete model validation documentation for regulators
- [ ] Conduct red-team exercise on prompt injection vectors
- [ ] Register as Investment Adviser (if providing advice in US)

## Disclaimer Jurisdiction Map

| Jurisdiction | Regulatory Basis |
|---|---|
| US | SEC Reg BI, FINRA Rule 2111 |
| UK | FCA COBS, Consumer Duty |
| EU | MiFID II Article 25 |
| CA | OSC, provincial securities acts |
