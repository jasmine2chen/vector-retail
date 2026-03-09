output "documents_bucket_name" {
  value = aws_s3_bucket.documents.id
}

output "documents_bucket_arn" {
  value = aws_s3_bucket.documents.arn
}

output "audit_bucket_name" {
  value = aws_s3_bucket.audit_archive.id
}

output "audit_bucket_arn" {
  value = aws_s3_bucket.audit_archive.arn
}
