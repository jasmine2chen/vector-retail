output "secret_arn" {
  value = aws_secretsmanager_secret.anthropic.arn
}

output "secrets_policy_arn" {
  value = aws_iam_policy.secrets_read.arn
}
