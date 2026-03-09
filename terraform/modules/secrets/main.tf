# ─────────────────────────────────────────────────────────────────────────────
# Secrets Module — Secrets Manager for API keys
# ─────────────────────────────────────────────────────────────────────────────

resource "aws_secretsmanager_secret" "anthropic" {
  name        = "${var.project_name}/${var.environment}/anthropic-api-key"
  description = "Anthropic API key for LLM calls (Bedrock fallback)"

  recovery_window_in_days = 7

  tags = { Name = "${var.project_name}-anthropic-key" }
}

resource "aws_secretsmanager_secret_version" "anthropic" {
  secret_id     = aws_secretsmanager_secret.anthropic.id
  secret_string = var.anthropic_api_key
}

# IAM policy for ECS to read secrets
resource "aws_iam_policy" "secrets_read" {
  name        = "${var.project_name}-${var.environment}-secrets-read"
  description = "Allow reading the Anthropic API key from Secrets Manager"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect   = "Allow"
        Action   = ["secretsmanager:GetSecretValue"]
        Resource = [aws_secretsmanager_secret.anthropic.arn]
      }
    ]
  })
}
