output "builder_instance_ids" {
  description = "Builder instance by index, for `aws ec2 get-console-output --region us-west-2 --instance-id <id> --latest`"
  value       = { for index, builder in aws_instance.builder : index => builder.id }
}

output "log_group" {
  description = "CloudWatch log group; one stream per index, named index-<index>"
  value       = aws_cloudwatch_log_group.builder.name
}
