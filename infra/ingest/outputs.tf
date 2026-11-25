output "worker_instance_ids" {
  description = "For `aws ssm start-session --region us-west-2 --target <id>`"
  value       = aws_instance.worker[*].id
}

output "log_group" {
  description = "CloudWatch log group; one stream per worker, named server-<id>"
  value       = aws_cloudwatch_log_group.worker.name
}
