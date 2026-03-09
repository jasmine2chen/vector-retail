# ─────────────────────────────────────────────────────────────────────────────
# Outputs
# ─────────────────────────────────────────────────────────────────────────────

output "api_endpoint" {
  description = "API Gateway endpoint URL"
  value       = module.api.api_endpoint
}

output "ecr_repository_url" {
  description = "ECR repository URL for docker push"
  value       = module.registry.repository_url
}

output "ecs_cluster_name" {
  description = "ECS cluster name"
  value       = module.compute.cluster_name
}

output "database_endpoint" {
  description = "Aurora PostgreSQL endpoint"
  value       = module.database.cluster_endpoint
}

output "redis_endpoint" {
  description = "ElastiCache Redis endpoint"
  value       = module.cache.redis_endpoint
}

output "documents_bucket" {
  description = "S3 bucket for RAG documents"
  value       = module.storage.documents_bucket_name
}

output "log_group" {
  description = "CloudWatch log group name"
  value       = module.monitoring.log_group_name
}
