# ─────────────────────────────────────────────────────────────────────────────
# Database Module — Aurora PostgreSQL Serverless v2 + pgvector
# Combined relational (sessions, audit, users) + vector (embeddings) store
# ─────────────────────────────────────────────────────────────────────────────

resource "aws_db_subnet_group" "main" {
  name       = "${var.project_name}-${var.environment}-db"
  subnet_ids = var.private_subnet_ids

  tags = { Name = "${var.project_name}-db-subnet-group" }
}

resource "aws_security_group" "aurora" {
  name_prefix = "${var.project_name}-aurora-"
  vpc_id      = var.vpc_id

  ingress {
    description     = "PostgreSQL from ECS"
    from_port       = 5432
    to_port         = 5432
    protocol        = "tcp"
    security_groups = [var.ecs_security_group_id]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = { Name = "${var.project_name}-aurora-sg" }
}

# Generate database credentials
resource "random_password" "db_password" {
  length  = 32
  special = false
}

resource "aws_secretsmanager_secret" "db_credentials" {
  name        = "${var.project_name}/${var.environment}/db-credentials"
  description = "Aurora PostgreSQL credentials"

  recovery_window_in_days = 7
}

resource "aws_secretsmanager_secret_version" "db_credentials" {
  secret_id = aws_secretsmanager_secret.db_credentials.id
  secret_string = jsonencode({
    username = "vectorretail"
    password = random_password.db_password.result
    dbname   = "vectorretail"
  })
}

# ── Aurora Cluster ───────────────────────────────────────────────────────────

resource "aws_rds_cluster" "main" {
  cluster_identifier = "${var.project_name}-${var.environment}"
  engine             = "aurora-postgresql"
  engine_mode        = "provisioned"
  engine_version     = "16.4"

  database_name   = "vectorretail"
  master_username = "vectorretail"
  master_password = random_password.db_password.result

  db_subnet_group_name   = aws_db_subnet_group.main.name
  vpc_security_group_ids = [aws_security_group.aurora.id]

  # Serverless v2 scaling
  serverlessv2_scaling_configuration {
    min_capacity = 0.5   # Scale to near-zero when idle
    max_capacity = 4.0   # Scale up under load
  }

  # Security
  storage_encrypted = true
  deletion_protection = true

  # Backup
  backup_retention_period      = 35
  preferred_backup_window      = "03:00-04:00"
  preferred_maintenance_window = "sun:04:00-sun:05:00"

  # Logging
  enabled_cloudwatch_logs_exports = ["postgresql"]

  skip_final_snapshot = false
  final_snapshot_identifier = "${var.project_name}-${var.environment}-final"

  tags = { Name = "${var.project_name}-aurora-cluster" }
}

resource "aws_rds_cluster_instance" "main" {
  count = 2  # Writer + reader for HA

  identifier         = "${var.project_name}-${var.environment}-${count.index}"
  cluster_identifier = aws_rds_cluster.main.id
  instance_class     = "db.serverless"
  engine             = aws_rds_cluster.main.engine
  engine_version     = aws_rds_cluster.main.engine_version

  tags = { Name = "${var.project_name}-aurora-instance-${count.index}" }
}
