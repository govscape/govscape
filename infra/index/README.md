# GovScape index builders (OpenTofu)

This stack builds a serving directory's indexes on EC2, from the page text,
embeddings and metadata that the ingestion workers (`../ingest`) wrote to the
derivative repository on source.coop. It launches **one instance per index**, so
the indexes are built concurrently. Each builder runs its index script from the
app image, uploads the index next to the data, and **powers itself off**. It is
separate from the hosting and ingest stacks and has its own state.

| Index | Script | Written to |
| --- | --- | --- |
| `txt` | `scripts/indexing/generate_index_embedding.py --embedding_type txt` | `<remote_data_dir>/index/` |
| `img_pg` | `scripts/indexing/generate_index_embedding.py --embedding_type img_pg` | `<remote_data_dir>/index_img_pg/` |
| `keyword` | `scripts/indexing/generate_index_keyword.py` (Tantivy by default) | `<remote_data_dir>/index_keyword/` |
| `metadata` | `scripts/indexing/generate_index_metadata.py` (also reads the CDX from the PDF archive) | `<remote_data_dir>/index_metadata/` |

## How a builder runs

- **AMI:** Amazon Linux 2023, with Docker installed at boot. No GPU is needed.
- **Credentials:** the existing `SourceCoopAccess` instance profile, which
  source.coop lets write to the `govscape/` repositories. The stack adds write
  access to its log group to that role, and removes it on destroy. It doesn't
  attach SSM access: the ingest stack manages that on the same role.
- **Progress:** each index script checkpoints in `<remote_data_dir>/checkpoints/`
  and adds to the index already in the serving directory, so a restarted builder
  resumes, and building again later adds only what's new. To rebuild an index
  from scratch, delete it and its checkpoint first.
- **Exits and failures:** a build is retried twice after 60 s; the builder powers
  off when it finishes or after 3 failed attempts. Its messages also go to the
  serial console (`aws ec2 get-console-output --instance-id <id> --latest`).
- **Spot (the default):** a persistent spot request that stops the builder on
  interruption; AWS starts it again when capacity returns.
- **Logs:** CloudWatch log group `/govscape/index`, one stream per index
  (`index-<index>`).

## Usage

```sh
cd infra/index
tofu init
tofu apply \
  -var remote_data_dir=prod-serving/ \
  -var app_image=ghcr.io/govscape/govscape:sha-<commit>
```

Build only some indexes with e.g. `-var 'indexes=["keyword"]'`, and give an index
a bigger instance with e.g. `-var 'instance_types={ txt = "r7i.8xlarge" }'`. The
builders stay stopped once done; `tofu destroy` removes them.
