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
    key          = "ingest/terraform.tfstate"
    region       = "us-east-2"
    encrypt      = true
    use_lockfile = true
  }
}

locals {
  name   = "govscape-ingest"
  region = "us-west-2"
}

# us-west-2, next to source.coop's bucket.
provider "aws" {
  region = local.region

  default_tags {
    tags = {
      Project   = "govscape"
      Component = "ingest"
      ManagedBy = "opentofu"
    }
  }
}

# Base AMI with the NVIDIA driver, Docker and the NVIDIA container toolkit.
data "aws_ssm_parameter" "gpu_ami" {
  name = "/aws/service/deeplearning/ami/x86_64/base-oss-nvidia-driver-gpu-amazon-linux-2023/latest/ami-id"
}

data "aws_ec2_instance_type" "worker" {
  instance_type = var.instance_type
}

# Workers run in the default VPC's public subnets, spread across the zones that
# offer the instance type, which also widens the pool of spot capacity.
data "aws_vpc" "default" {
  default = true
}

data "aws_ec2_instance_type_offerings" "worker" {
  location_type = "availability-zone"

  filter {
    name   = "instance-type"
    values = [var.instance_type]
  }
}

data "aws_subnets" "worker" {
  filter {
    name   = "vpc-id"
    values = [data.aws_vpc.default.id]
  }

  filter {
    name = "availability-zone"
    values = (
      var.availability_zones == null
      ? data.aws_ec2_instance_type_offerings.worker.locations
      : tolist(setintersection(var.availability_zones, data.aws_ec2_instance_type_offerings.worker.locations))
    )
  }

  filter {
    name   = "default-for-az"
    values = ["true"]
  }
}

resource "aws_security_group" "worker" {
  name        = local.name
  description = "GovScape ingestion workers: outbound only"
  vpc_id      = data.aws_vpc.default.id
}

resource "aws_vpc_security_group_egress_rule" "worker_all" {
  security_group_id = aws_security_group.worker.id
  description       = "source.coop, image pulls, Hugging Face models, logs, SSM"
  cidr_ipv4         = "0.0.0.0/0"
  ip_protocol       = "-1"
}

resource "aws_cloudwatch_log_group" "worker" {
  name              = "/govscape/ingest"
  retention_in_days = 30
}

# The workers use an existing role because source.coop's bucket policy grants
# that role write access; it can't be created here. These attachments add
# shell access and logging, and are removed again on destroy.
data "aws_iam_instance_profile" "worker" {
  name = var.instance_profile
}

resource "aws_iam_role_policy_attachment" "worker_ssm" {
  role       = data.aws_iam_instance_profile.worker.role_name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"
}

data "aws_iam_policy_document" "worker_logs" {
  statement {
    actions   = ["logs:CreateLogStream", "logs:PutLogEvents"]
    resources = ["${aws_cloudwatch_log_group.worker.arn}:*"]
  }
}

resource "aws_iam_role_policy" "worker_logs" {
  name   = "${local.name}-logs"
  role   = data.aws_iam_instance_profile.worker.role_name
  policy = data.aws_iam_policy_document.worker_logs.json
}

resource "aws_instance" "worker" {
  count = var.num_servers

  ami                    = data.aws_ssm_parameter.gpu_ami.insecure_value
  instance_type          = var.instance_type
  subnet_id              = data.aws_subnets.worker.ids[count.index % length(data.aws_subnets.worker.ids)]
  vpc_security_group_ids = [aws_security_group.worker.id]
  iam_instance_profile   = data.aws_iam_instance_profile.worker.name

  # A worker powers itself off when it's done; stopped (not terminated) keeps
  # it inspectable and costs only its root volume until `tofu destroy`. AWS
  # won't set this on spot instances, but persistent spot instances that stop
  # on interruption already stop on shutdown.
  instance_initiated_shutdown_behavior = var.use_spot ? null : "stop"

  dynamic "instance_market_options" {
    for_each = var.use_spot ? [1] : []
    content {
      market_type = "spot"
      spot_options {
        # After an interruption AWS restarts the worker when capacity returns,
        # and it resumes from its shards' checkpoints.
        spot_instance_type             = "persistent"
        instance_interruption_behavior = "stop"
      }
    }
  }

  user_data = templatefile("${path.module}/templates/cloud-init.yaml.tftpl", {
    app_image                   = var.app_image
    log_group                   = aws_cloudwatch_log_group.worker.name
    region                      = local.region
    server_id                   = count.index
    num_servers                 = var.num_servers
    remote_data_dir             = var.remote_data_dir
    vllm_image                  = var.vllm_image == null ? "" : var.vllm_image
    vllm_gpu_memory_utilization = var.vllm_gpu_memory_utilization
    vllm_extra_args             = var.vllm_extra_args
    pipeline_args = join(" ", compact([
      var.pipeline_args,
      var.max_pdfs == null ? "" : "--max_pdfs ${var.max_pdfs}",
    ]))
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
    Name = "${local.name}-${count.index}"
  }

  lifecycle {
    precondition {
      condition     = length(data.aws_ec2_instance_type.worker.gpus) > 0
      error_message = "instance_type must have a GPU."
    }

    ignore_changes = [ami]
  }
}
