terraform {
  # 1.10+ for S3 native state locking (use_lockfile).
  required_version = ">= 1.10"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 6.0"
    }
  }

  # The bucket is created by ./bootstrap and stays in us-east-2; its region is
  # independent of the stack's. Backend blocks can't use variables; override
  # with `tofu init -backend-config=bucket=...` if the name differs.
  backend "s3" {
    bucket       = "govscape-tofu-state"
    key          = "site/next.tfstate"
    region       = "us-east-2"
    encrypt      = true
    use_lockfile = true
  }
}

locals {
  name = "govscape-${var.environment}"

  default_tags = {
    Project     = "govscape"
    Environment = var.environment
    ManagedBy   = "opentofu"
  }
}

# us-west-2, next to source.coop's data.
provider "aws" {
  region = "us-west-2"

  default_tags {
    tags = local.default_tags
  }
}

# CloudFront only accepts ACM certificates issued in us-east-1.
provider "aws" {
  alias  = "us_east_1"
  region = "us-east-1"

  default_tags {
    tags = local.default_tags
  }
}

data "aws_route53_zone" "main" {
  name = "govscape.net"
}
