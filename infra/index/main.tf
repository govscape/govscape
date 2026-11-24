terraform {
  required_version = ">= 1.10"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 6.0"
    }
  }

  # Same state bucket as the hosting stack (see ../bootstrap), separate key.
  backend "s3" {
    bucket       = "govscape-tofu-state"
    key          = "index/terraform.tfstate"
    region       = "us-east-2"
    encrypt      = true
    use_lockfile = true
  }
}

locals {
  name   = "govscape-index"
  region = "us-west-2"

  # The script and arguments that build each index. num_pages_to_process is
  # each script's cap on how much input it reads; 10^12 means no cap.
  index_commands = {
    txt      = "scripts/indexing/generate_index_embedding.py --embedding_type txt --forward_index_type ${var.forward_index_type} --num_pages_to_process 1000000000000"
    img_pg   = "scripts/indexing/generate_index_embedding.py --embedding_type img_pg --forward_index_type ${var.forward_index_type} --num_pages_to_process 1000000000000"
    keyword  = "scripts/indexing/generate_index_keyword.py --keyword_index_type ${var.keyword_index_type} --num_pages_to_process 1000000000000 --batch_size 1000000"
    metadata = "scripts/indexing/generate_index_metadata.py --num_pages_to_process 1000000000000"
  }

  # One builder per index, so the indexes are built concurrently.
  instance_types = {
    for index in var.indexes : index => lookup(var.instance_types, index, var.instance_type)
  }
}

# us-west-2, next to source.coop's bucket.
provider "aws" {
  region = local.region

  default_tags {
    tags = {
      Project   = "govscape"
      Component = "index"
      ManagedBy = "opentofu"
    }
  }
}

# Index building needs no GPU: plain Amazon Linux 2023, with Docker installed at
# boot.
data "aws_ssm_parameter" "ami" {
  name = "/aws/service/ami-amazon-linux-latest/al2023-ami-kernel-default-x86_64"
}

# Each builder runs in a default subnet of the default VPC, in a zone that
# offers its instance type.
data "aws_vpc" "default" {
  default = true
}

data "aws_ec2_instance_type_offerings" "builder" {
  for_each      = toset(values(local.instance_types))
  location_type = "availability-zone"

  filter {
    name   = "instance-type"
    values = [each.key]
  }
}

data "aws_subnets" "builder" {
  for_each = toset(values(local.instance_types))

  filter {
    name   = "vpc-id"
    values = [data.aws_vpc.default.id]
  }

  filter {
    name = "availability-zone"
    values = (
      var.availability_zones == null
      ? data.aws_ec2_instance_type_offerings.builder[each.key].locations
      : tolist(setintersection(var.availability_zones, data.aws_ec2_instance_type_offerings.builder[each.key].locations))
    )
  }

  filter {
    name   = "default-for-az"
    values = ["true"]
  }
}

resource "aws_security_group" "builder" {
  name        = local.name
  description = "GovScape index builders: outbound only"
  vpc_id      = data.aws_vpc.default.id
}

resource "aws_vpc_security_group_egress_rule" "builder_all" {
  security_group_id = aws_security_group.builder.id
  description       = "source.coop, image pulls, packages, logs"
  cidr_ipv4         = "0.0.0.0/0"
  ip_protocol       = "-1"
}

resource "aws_cloudwatch_log_group" "builder" {
  name              = "/govscape/index"
  retention_in_days = 30
}

# The builders use the existing role that source.coop's bucket policy lets write
# to the govscape repositories. This stack only adds write access to its own log
# group; it doesn't attach SSM access, which the ingest stack manages on the same
# role and would lose if this stack were destroyed.
data "aws_iam_instance_profile" "builder" {
  name = var.instance_profile
}

data "aws_iam_policy_document" "builder_logs" {
  statement {
    actions   = ["logs:CreateLogStream", "logs:PutLogEvents"]
    resources = ["${aws_cloudwatch_log_group.builder.arn}:*"]
  }
}

resource "aws_iam_role_policy" "builder_logs" {
  name   = "${local.name}-logs"
  role   = data.aws_iam_instance_profile.builder.role_name
  policy = data.aws_iam_policy_document.builder_logs.json
}

resource "aws_instance" "builder" {
  for_each = local.instance_types

  ami                    = data.aws_ssm_parameter.ami.insecure_value
  instance_type          = each.value
  subnet_id              = data.aws_subnets.builder[each.value].ids[0]
  vpc_security_group_ids = [aws_security_group.builder.id]
  iam_instance_profile   = data.aws_iam_instance_profile.builder.name

  # A builder powers itself off when it's done; stopped (not terminated)
  # keeps it inspectable and costs only its root volume until `tofu destroy`.
  # AWS won't set this on spot instances, but persistent spot instances that
  # stop on interruption already stop on shutdown.
  instance_initiated_shutdown_behavior = var.use_spot ? null : "stop"

  dynamic "instance_market_options" {
    for_each = var.use_spot ? [1] : []
    content {
      market_type = "spot"
      spot_options {
        # After an interruption AWS restarts the builder when capacity returns,
        # and its index script resumes from its checkpoint.
        spot_instance_type             = "persistent"
        instance_interruption_behavior = "stop"
      }
    }
  }

  user_data = templatefile("${path.module}/templates/cloud-init.yaml.tftpl", {
    app_image       = var.app_image
    log_group       = aws_cloudwatch_log_group.builder.name
    region          = local.region
    remote_data_dir = var.remote_data_dir
    index           = each.key
    command         = local.index_commands[each.key]
  })
  user_data_replace_on_change = true

  root_block_device {
    volume_size = var.root_volume_gb
    volume_type = "gp3"
    encrypted   = true
  }

  metadata_options {
    http_tokens = "required"
    # The container reaches the instance role's credentials through Docker's
    # bridge network, one hop further than the host.
    http_put_response_hop_limit = 2
  }

  tags = {
    Name = "${local.name}-${each.key}"
  }

  lifecycle {
    ignore_changes = [ami]
  }
}
