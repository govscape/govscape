environment  = "next"
site_domains = ["next.govscape.net"]

# Pin to an immutable tag from the publish-app-image workflow (sha-<short sha>)
# once one exists; applying a new tag replaces the API instance.
app_image = "ghcr.io/govscape/govscape:main"

serving_data_dir = "prod-serving"
instance_type    = "t3.xlarge"
