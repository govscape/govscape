terraform {
  # 1.10+ for S3 native state locking (use_lockfile).
  required_version = ">= 1.10"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 6.0"
    }
  }

  # The bucket is created by ./bootstrap. Backend blocks can't use variables;
  # override with `tofu init -backend-config=bucket=...` if the name differs.
  backend "s3" {
    bucket       = "govscape-tofu-state"
    key          = "site/next.tfstate"
    region       = "us-east-2"
    encrypt      = true
    use_lockfile = true
  }
}
