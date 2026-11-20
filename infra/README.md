# GovScape hosting (OpenTofu)

This directory manages the GovScape website on AWS with [OpenTofu](https://opentofu.org) (>= 1.10).
All data (page images, indices, PDFs) is served from source.coop; AWS only hosts the site and the API.

```
viewer ──HTTPS──► CloudFront (site_domains, ACM cert)
                    ├─ default  ──► S3 bucket (SvelteKit static build, private via OAC)
                    │               + spa_rewrite function: extension-less paths → /index.html
                    └─ /api/*   ──► origin.<site domain>:8080 (HTTP) ──► EC2 instance (Elastic IP)
                                     Docker: ghcr.io/govscape/govscape, run_gunicorn
                                     indices pulled anonymously from source.coop on start
browser ──► data.source.coop/govscape/eota-derivative-data/<serving dir>/img  (page images)
        ──► data.source.coop/govscape/eota-pdf-archive/pdfs                    (PDF downloads)
```

The stack starts out serving `next.govscape.net` (see `terraform.tfvars`). The existing govscape.net site was built by hand and is not managed here. See [Cutover](#cutover-to-govscapenet).

| File | Contents |
| --- | --- |
| `bootstrap/` | S3 bucket for this stack's state (applied once, keeps its own local state) |
| `network.tf` | VPC, one public subnet, internet gateway |
| `api.tf` | API instance, Elastic IP, security group (CloudFront origin-facing IPs only), SSM role, `origin.` DNS record |
| `frontend.tf` | Site bucket, CloudFront distribution and SPA function, ACM certificate, DNS aliases |
| `github.tf` | GitHub Actions OIDC provider and the frontend deploy role |
| `templates/cloud-init.yaml.tftpl` | Instance bootstrap: Docker + `govscape-api` systemd service |

## Prerequisites

- OpenTofu >= 1.10 and the AWS CLI, with credentials for the AWS account that holds the `govscape.net` Route 53 zone.
- The `ghcr.io/govscape/govscape` package must be public, so the instance can pull it without credentials (GitHub → package settings → visibility). It is published by `.github/workflows/publish-app-image.yml`.
- A populated serving directory on source.coop (`serving_data_dir`, default `prod-serving`). Generate one with `scripts/generate_serving_data.sh`.

## First-time setup

```bash
# 1. State bucket (once per account). If the name is taken, pick another and
#    pass it to `tofu init -backend-config=bucket=<name>` below.
cd infra/bootstrap
tofu init && tofu apply

# 2. The site stack
cd ..
tofu init
tofu plan -out=site.tfplan     # review
tofu apply site.tfplan
```

If the account already has a GitHub Actions OIDC provider, `apply` fails with `EntityAlreadyExists`. Set `create_github_oidc_provider = false` in `terraform.tfvars` and apply again.

3. Set the deploy workflow's repository variables from the outputs:

```bash
gh variable set AWS_SITE_DEPLOY_ROLE_ARN --body "$(tofu output -raw site_deploy_role_arn)"
gh variable set SITE_BUCKET              --body "$(tofu output -raw site_bucket)"
gh variable set SITE_DISTRIBUTION_ID     --body "$(tofu output -raw site_distribution_id)"
```

4. Run the **Deploy Site** workflow (Actions → Deploy Site → Run workflow), or push a change under `interface/` to `main`.

## Deploying

- **Frontend:** `deploy-site.yml` runs on every push to `main` that touches `interface/`. It syncs the build to the site bucket and invalidates `/index.html`.
- **API:** `publish-app-image.yml` publishes `ghcr.io/govscape/govscape:sha-<short sha>` (and `:main`) on every push to `main`. To deploy, set `app_image` in `terraform.tfvars` to the new `sha-` tag and run `tofu apply`. This replaces the instance, so expect a few minutes of API downtime while the new one pulls the image, downloads the indices and loads the models. Changing other server settings (`serving_data_dir`, `server_args`, gunicorn settings) works the same way.
- **New data:** after regenerating the serving directory, restart the service (below). The service downloads the indices again every time it starts.

## Operating the instance

There is no SSH; use SSM Session Manager (requires the [Session Manager plugin](https://docs.aws.amazon.com/systems-manager/latest/userguide/session-manager-working-with-install-plugin.html)):

```bash
aws ssm start-session --region us-east-2 --target "$(tofu output -raw api_instance_id)"
sudo journalctl -u govscape-api -f       # logs
sudo systemctl restart govscape-api      # re-pull the image tag and the indices
```

Health check: `curl https://next.govscape.net/api/health/`.

Security note: traffic between CloudFront and the instance is plain HTTP. The security group only allows CloudFront's origin-facing IP ranges, and the API serves only public data.

## Sizing

The FAISS indices are held in memory by every gunicorn worker (shared copy-on-write through `preload_app`), and the text and visual models run on the CPU. `t3.xlarge` (16 GiB) is enough for a 100-PDF serving directory. Before serving the full dataset, size `instance_type` (a memory-optimized `r7i.*` instance is a good start) and `root_volume_gb` to the sizes of `index/`, `index_img_pg/`, `index_keyword/` and `index_metadata/`. Then tune `gunicorn_workers`.

## Cutover to govscape.net

Do this once the full serving directory exists and the stack has been checked on next.govscape.net.

1. Size the instance for the full data (see above) and apply.
2. Free the names from the old CloudFront distribution: remove `govscape.net` and `www.govscape.net` from its alternate domain names in the console. A hostname can belong to only one distribution at a time. (Alternatively, move them without a gap using `aws cloudfront associate-alias`, which requires a `_govscape.net` TXT record pointing at the new distribution.)
3. In `terraform.tfvars`, set `site_domains = ["govscape.net", "www.govscape.net"]`. Add `import` blocks for the existing apex and `www` A/AAAA records, so they are adopted rather than duplicated:

   ```hcl
   import {
     to = aws_route53_record.site["govscape.net-A"]
     id = "<zone id>_govscape.net_A"
   }
   ```

   Run `tofu plan`, check that the old records are updated in place, then `tofu apply`. The ACM certificate is reissued for the new names.
4. Retire `.github/workflows/deploy-frontend.yml` (it deploys the old site from the `prod-serving` branch).
5. Check the site, then delete the old stack by hand: the old distribution, the `govscape-frontend-prod` bucket, the old EC2 instance, load balancer and certificates, and the deploy IAM user whose keys are in the `AWS_ACCESS_KEY_ID`/`AWS_SECRET_ACCESS_KEY` repository secrets. Archive and retire `bcgl-public-bucket` once nothing reads from it.

Rolling back before step 5 means reverting `site_domains` and re-adding the aliases to the old distribution.
