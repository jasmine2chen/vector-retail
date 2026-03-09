# ─────────────────────────────────────────────────────────────────────────────
# AI Module — Bedrock model access (Claude in-VPC) + IAM
# ─────────────────────────────────────────────────────────────────────────────

resource "aws_iam_policy" "bedrock_invoke" {
  name        = "${var.project_name}-${var.environment}-bedrock-invoke"
  description = "Allow ECS tasks to invoke Bedrock models (Claude) for LLM inference"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "InvokeModels"
        Effect = "Allow"
        Action = [
          "bedrock:InvokeModel",
          "bedrock:InvokeModelWithResponseStream"
        ]
        # Scoped to Claude models only — least privilege
        Resource = [
          "arn:aws:bedrock:${var.aws_region}::foundation-model/anthropic.claude-*"
        ]
      },
      {
        Sid    = "ListModels"
        Effect = "Allow"
        Action = ["bedrock:ListFoundationModels"]
        Resource = ["*"]
      }
    ]
  })

  tags = { Name = "${var.project_name}-bedrock-policy" }
}
