# A single EC2 instance runs the API container. CloudFront reaches it over HTTP
# on port 8080 through a DNS name pointing at its Elastic IP; the security group
# only admits CloudFront's origin-facing IP ranges. Shell access is through SSM
# Session Manager, so there is no SSH port or key pair.

locals {
  api_port      = 8080
  origin_domain = "origin.${var.site_domains[0]}"
}

data "aws_ssm_parameter" "al2023_ami" {
  name = "/aws/service/ami-amazon-linux-latest/al2023-ami-kernel-default-x86_64"
}

data "aws_ec2_managed_prefix_list" "cloudfront_origin_facing" {
  name = "com.amazonaws.global.cloudfront.origin-facing"
}

data "aws_iam_policy_document" "ec2_assume_role" {
  statement {
    actions = ["sts:AssumeRole"]

    principals {
      type        = "Service"
      identifiers = ["ec2.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "api" {
  name               = "${local.name}-api"
  assume_role_policy = data.aws_iam_policy_document.ec2_assume_role.json
}

resource "aws_iam_role_policy_attachment" "api_ssm" {
  role       = aws_iam_role.api.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"
}

resource "aws_iam_instance_profile" "api" {
  name = "${local.name}-api"
  role = aws_iam_role.api.name
}

resource "aws_security_group" "api" {
  name        = "${local.name}-api"
  description = "GovScape API: HTTP from CloudFront only"
  vpc_id      = aws_vpc.main.id
}

resource "aws_vpc_security_group_ingress_rule" "api_from_cloudfront" {
  security_group_id = aws_security_group.api.id
  description       = "API from CloudFront origin-facing servers"
  prefix_list_id    = data.aws_ec2_managed_prefix_list.cloudfront_origin_facing.id
  ip_protocol       = "tcp"
  from_port         = local.api_port
  to_port           = local.api_port
}

resource "aws_vpc_security_group_egress_rule" "api_all" {
  security_group_id = aws_security_group.api.id
  description       = "Image pulls, source.coop data, Hugging Face models, SSM"
  cidr_ipv4         = "0.0.0.0/0"
  ip_protocol       = "-1"
}

resource "aws_instance" "api" {
  ami                    = data.aws_ssm_parameter.al2023_ami.insecure_value
  instance_type          = var.instance_type
  subnet_id              = aws_subnet.public.id
  vpc_security_group_ids = [aws_security_group.api.id]
  iam_instance_profile   = aws_iam_instance_profile.api.name

  user_data = templatefile("${path.module}/templates/cloud-init.yaml.tftpl", {
    app_image        = var.app_image
    api_port         = local.api_port
    serving_data_dir = var.serving_data_dir
    server_args      = var.server_args
    gunicorn_workers = var.gunicorn_workers
    gunicorn_threads = var.gunicorn_threads
  })
  # Deploys (a new app_image or server settings) replace the instance.
  user_data_replace_on_change = true

  root_block_device {
    volume_size = var.root_volume_gb
    volume_type = "gp3"
    encrypted   = true
  }

  metadata_options {
    http_tokens = "required"
    # Keep the container from reaching instance metadata/credentials; it reads
    # source.coop anonymously.
    http_put_response_hop_limit = 1
  }

  tags = {
    Name = "${local.name}-api"
  }

  lifecycle {
    # Don't replace the instance whenever AWS publishes a new AMI; a replacement
    # for any other reason picks up the latest one.
    ignore_changes = [ami]
  }
}

resource "aws_eip" "api" {
  domain   = "vpc"
  instance = aws_instance.api.id

  tags = {
    Name = "${local.name}-api"
  }

  depends_on = [aws_internet_gateway.main]
}

resource "aws_route53_record" "api_origin" {
  zone_id = data.aws_route53_zone.main.zone_id
  name    = local.origin_domain
  type    = "A"
  ttl     = 60
  records = [aws_eip.api.public_ip]
}
