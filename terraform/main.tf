# ─────────────────────────────────────────────────────────────────────────────
# Vector Retail — Root Terraform Configuration
# Production financial AI agent on AWS (cloud-portable module design)
# ─────────────────────────────────────────────────────────────────────────────

terraform {
  required_version = ">= 1.5.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.0"
    }
  }

  # Uncomment to use S3 backend for team state management:
  # backend "s3" {
  #   bucket         = "vector-retail-terraform-state"
  #   key            = "prod/terraform.tfstate"
  #   region         = "us-east-1"
  #   encrypt        = true
  #   dynamodb_table = "terraform-locks"
  # }
}

provider "aws" {
  region = var.aws_region

  default_tags {
    tags = {
      Project     = "vector-retail"
      Environment = var.environment
      ManagedBy   = "terraform"
    }
  }
}

# ── Data Sources ─────────────────────────────────────────────────────────────

data "aws_caller_identity" "current" {}
data "aws_region" "current" {}

# ── Modules ──────────────────────────────────────────────────────────────────

module "networking" {
  source = "./modules/networking"

  project_name = var.project_name
  environment  = var.environment
  vpc_cidr     = var.vpc_cidr
  az_count     = var.az_count
}

module "registry" {
  source = "./modules/registry"

  project_name = var.project_name
  environment  = var.environment
}

module "secrets" {
  source = "./modules/secrets"

  project_name    = var.project_name
  environment     = var.environment
  anthropic_api_key = var.anthropic_api_key
}

module "database" {
  source = "./modules/database"

  project_name       = var.project_name
  environment        = var.environment
  vpc_id             = module.networking.vpc_id
  private_subnet_ids = module.networking.private_subnet_ids
  ecs_security_group_id = module.networking.ecs_security_group_id
}

module "cache" {
  source = "./modules/cache"

  project_name       = var.project_name
  environment        = var.environment
  vpc_id             = module.networking.vpc_id
  private_subnet_ids = module.networking.private_subnet_ids
  ecs_security_group_id = module.networking.ecs_security_group_id
}

module "storage" {
  source = "./modules/storage"

  project_name = var.project_name
  environment  = var.environment
  account_id   = data.aws_caller_identity.current.account_id
}

module "ai" {
  source = "./modules/ai"

  project_name = var.project_name
  environment  = var.environment
  aws_region   = var.aws_region
  account_id   = data.aws_caller_identity.current.account_id
}

module "compute" {
  source = "./modules/compute"

  project_name       = var.project_name
  environment        = var.environment
  vpc_id             = module.networking.vpc_id
  private_subnet_ids = module.networking.private_subnet_ids
  ecr_repository_url = module.registry.repository_url
  image_tag          = var.image_tag

  # Secrets & IAM
  secret_arn              = module.secrets.secret_arn
  bedrock_policy_arn      = module.ai.bedrock_policy_arn
  documents_bucket_arn    = module.storage.documents_bucket_arn
  audit_bucket_arn        = module.storage.audit_bucket_arn
  db_endpoint             = module.database.cluster_endpoint
  db_secret_arn           = module.database.db_secret_arn
  redis_endpoint          = module.cache.redis_endpoint

  # Networking
  ecs_security_group_id = module.networking.ecs_security_group_id
  alb_target_group_arn  = module.api.alb_target_group_arn

  # Logging
  log_group_name = module.monitoring.log_group_name

  container_cpu    = var.container_cpu
  container_memory = var.container_memory
  desired_count    = var.desired_count
}

module "api" {
  source = "./modules/api"

  project_name       = var.project_name
  environment        = var.environment
  vpc_id             = module.networking.vpc_id
  private_subnet_ids = module.networking.private_subnet_ids
  ecs_security_group_id = module.networking.ecs_security_group_id
}

module "monitoring" {
  source = "./modules/monitoring"

  project_name     = var.project_name
  environment      = var.environment
  ecs_cluster_name = module.compute.cluster_name
  ecs_service_name = module.compute.service_name
  alb_arn_suffix   = module.api.alb_arn_suffix
}
