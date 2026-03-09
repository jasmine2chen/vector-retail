variable "project_name" { type = string }
variable "environment" { type = string }
variable "vpc_id" { type = string }
variable "private_subnet_ids" { type = list(string) }
variable "ecr_repository_url" { type = string }
variable "image_tag" { type = string }

variable "secret_arn" { type = string }
variable "bedrock_policy_arn" { type = string }
variable "documents_bucket_arn" { type = string }
variable "audit_bucket_arn" { type = string }
variable "db_endpoint" { type = string }
variable "db_secret_arn" { type = string }
variable "redis_endpoint" { type = string }

variable "alb_target_group_arn" { type = string }
variable "ecs_security_group_id" { type = string }
variable "log_group_name" { type = string }

variable "container_cpu" {
  type    = number
  default = 1024
}

variable "container_memory" {
  type    = number
  default = 2048
}

variable "desired_count" {
  type    = number
  default = 2
}
