output "site_url" {
  value = "https://${var.site_domains[0]}"
}

output "site_bucket" {
  description = "GitHub repo variable SITE_BUCKET"
  value       = aws_s3_bucket.site.bucket
}

output "site_distribution_id" {
  description = "GitHub repo variable SITE_DISTRIBUTION_ID"
  value       = aws_cloudfront_distribution.site.id
}

output "site_deploy_role_arn" {
  description = "GitHub repo variable AWS_SITE_DEPLOY_ROLE_ARN"
  value       = aws_iam_role.site_deploy.arn
}

output "api_instance_id" {
  description = "For `aws ssm start-session --target <id>`"
  value       = aws_instance.api.id
}

output "api_public_ip" {
  value = aws_eip.api.public_ip
}
