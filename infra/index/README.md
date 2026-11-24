# GovScape index builders (OpenTofu)

This stack builds a serving directory's indexes on EC2, from the page text,
embeddings and metadata that the ingestion workers (`../ingest`) wrote to the
derivative repository on source.coop. It launches **one instance per index**, so
the indexes are built concurrently. Each builder runs its index script from the
app image, uploads the index next to the data, and **powers itself off**. It is
separate from the hosting and ingest stacks and has its own state.

| Index | Script | Written to |
| --- | --- | --- |
| `txt` | `scripts/indexing/build_ivfpq_index.py --embedding_type txt` (on a GPU) | `<remote_data_dir>/index/` |
| `img_pg` | `scripts/indexing/build_ivfpq_index.py --embedding_type img_pg` (on a GPU) | `<remote_data_dir>/index_img_pg/` |
| `keyword` | `scripts/indexing/generate_index_keyword.py` (Tantivy by default) | `<remote_data_dir>/index_keyword/` |
| `metadata` | `scripts/indexing/generate_index_metadata.py` (also reads the CDX from the PDF archive) | `<remote_data_dir>/index_metadata/` |

The vector indexes are disk-based IVF-PQ indexes (`govscape/indexing/ivfpq.py`),
served with `--vector_index_type Disk`.

## How a builder runs

- **AMI and instance:** the vector builders run on `vector_instance_type`
  (g7.8xlarge by default) with the Deep Learning Base GPU AMI, and the container
  gets the GPU: FAISS trains the index and assigns vectors to its lists there.
  Before building, the builder checks that FAISS can see the GPU. The keyword
  and metadata builders run on `instance_type` with Amazon Linux 2023, with
  Docker installed at boot.
- **Credentials:** the existing `SourceCoopAccess` instance profile, which
  source.coop lets write to the `govscape/` repositories. The stack adds write
  access to its log group to that role, and removes it on destroy. It doesn't
  attach SSM access: the ingest stack manages that on the same role.
- **Progress:** the vector builders checkpoint every 100M vectors in their
  working directory on the root volume (`vector_root_volume_gb`, 3.5 TB by
  default, with 1,000 MB/s of throughput), which survives retries and spot
  interruptions, and upload the finished index; each build starts from scratch
  on a new instance. The keyword and metadata scripts checkpoint in
  `<remote_data_dir>/checkpoints/` and add to the index already in the serving
  directory, so building them again later adds only what's new; to rebuild one
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

Build only some indexes with e.g. `-var 'indexes=["keyword"]'`, give an index a
different instance with e.g. `-var 'instance_types={ txt = "g7.4xlarge" }'`, and
pass the IVF-PQ build options with e.g. `-var 'ivfpq_args=--nlist 65536'`. For a
small serving directory, shrink the vector builders' disks with e.g.
`-var vector_root_volume_gb=200`. The
builders stay stopped once done; `tofu destroy` removes them.
