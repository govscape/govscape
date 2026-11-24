# GovScape ingestion workers (OpenTofu)

This stack launches GPU instances in us-west-2 that run the embedding pipeline
(`scripts/pipeline/run_embedding_pipeline.py`) over the PDF archive and write
page text, images, embeddings and metadata to the derivative repository on
source.coop. It is separate from the hosting stack in `..` and has its own state.

Each worker is one pipeline server. Worker `i` of `num_servers` processes every
`num_servers`-th shard of the PDFs (shards are the 1,024 two-character digest
prefixes) and checkpoints each shard in `<remote_data_dir>/checkpoints/`. When it
has finished its shards, the worker **powers itself off**. It stays stopped,
not terminated, so its root volume and logs remain until `tofu destroy`.

| File | Contents |
| --- | --- |
| `main.tf` | State backend, provider, GPU AMI, default-VPC subnets, security group, log group, role attachments, workers |
| `variables.tf` | Run settings: `num_servers`, `remote_data_dir`, `app_image`, `max_pdfs`, `vllm_image`, ... |
| `templates/cloud-init.yaml.tftpl` | The `govscape-ingest` systemd service and its retry script |

## How a worker runs

- **AMI:** the AWS Deep Learning Base GPU AMI (Amazon Linux 2023), which has the
  NVIDIA driver, Docker and the NVIDIA container toolkit.
- **Credentials:** the existing `SourceCoopAccess` instance profile.
  source.coop's bucket policy lets that role write to the `govscape/` repositories
  directly in `us-west-2.opendata.source.coop`. Its credentials renew
  automatically, unlike source.coop API keys. The stack attaches SSM access and
  write access to the log group to that role, and removes them on destroy.
- **Working directory:** `/var/lib/govscape-ingest` on the root volume. It only
  holds the batch in progress; progress is kept in the shard checkpoints.
- **Before each start:** the service pulls the image and checks that PyTorch in
  the container can use the GPU. If the driver were too old for the image's CUDA
  build, the models would otherwise quietly fall back to the CPU.
- **Exits and failures:**

  | What happens | Result |
  | --- | --- |
  | The pipeline exits 0 | The worker powers off |
  | The image pull, GPU check or pipeline fails | It is retried after 60 s |
  | It fails 5 times in a row | The worker powers off as well |

  The retry script's own messages (attempts and failures) go to the serial
  console, so `aws ec2 get-console-output --instance-id <id> --latest` shows them
  even after the worker has stopped.

- **Spot (the default):** a persistent spot request that stops the worker on
  interruption. AWS starts it again when capacity returns, and the service resumes
  from the shard checkpoints. It loses at most the batch in progress.
- **Logs:** CloudWatch log group `/govscape/ingest`, with one stream per worker
  (`server-<id>`), plus `server-<id>-vllm` when vLLM is on.

## Qwen3 text embeddings with vLLM

Set `vllm_image` (e.g. `vllm/vllm-openai:v0.34.0`) and pass
`--text_model_type Qwen3vLLM` in `pipeline_args`. Before each attempt the worker
starts `vllm serve Qwen/Qwen3-Embedding-0.6B --runner pooling --max-model-len 2048`
in its own container and waits until it's healthy. The pipeline container then
sends page text to it over a private Docker network (`VLLM_URL`). vLLM
tokenizes and truncates each page to its first 2,048 tokens. On a g7.2xlarge
this embedded about 3× as many pages per second as running Qwen3 inside the
pipeline, with effectively identical embeddings.

The vLLM server starts first and takes `vllm_gpu_memory_utilization` of the GPU's
memory (default 0.5). The pipeline's image models use the rest. `vllm_extra_args`
adds arguments to `vllm serve`, e.g. `-var 'vllm_extra_args=--max-num-batched-tokens 32768'`.

## Prerequisites

- The hosting stack's state bucket (`../bootstrap`).
- A public `ghcr.io/govscape/govscape` image that includes the sharded pipeline.
  Pass an immutable `sha-*` tag as `app_image`.
- **GPU quota in us-west-2.** The account allows 64 spot G-instance vCPUs, which
  is 8× g6.2xlarge, and 768 on-demand. To use more spot workers, request a quota
  increase for "All G and VT Spot Instance Requests", or set `use_spot=false`.

## Run

```bash
cd infra/ingest
tofu init

# Pilot: one worker, 2,000 PDFs, into dev-serving/. It measures real throughput
# and checks the GPU, credentials and output layout.
tofu apply -var num_servers=1 -var max_pdfs=2000 \
  -var remote_data_dir=dev-serving/ -var app_image=ghcr.io/govscape/govscape:sha-<commit>

# The same pilot with Qwen3 text embeddings served by vLLM.
tofu apply -var num_servers=1 -var max_pdfs=2000 \
  -var remote_data_dir=dev-serving/ -var app_image=ghcr.io/govscape/govscape:sha-<commit> \
  -var vllm_image=vllm/vllm-openai:v0.34.0 \
  -var 'pipeline_args=--text_model_type Qwen3vLLM --visual_model_type CLIP'

# Full run, after the pilot.
tofu apply -var num_servers=8 \
  -var remote_data_dir=prod-serving/ -var app_image=ghcr.io/govscape/govscape:sha-<commit>
```

Keep `num_servers` the same while a run is in progress. Changing it re-deals the
shards, which is safe because finished shards are skipped, but it replaces every
worker.

## Monitor

```bash
# Follow one worker's log.
aws logs tail /govscape/ingest --region us-west-2 --log-stream-names server-0 --follow

# Worker states: "stopped" means finished, or gave up after repeated failures.
aws ec2 describe-instances --region us-west-2 \
  --filters Name=tag:Component,Values=ingest \
  --query 'Reservations[].Instances[].[Tags[?Key==`Name`]|[0].Value,State.Name]' --output text

# Shell on a worker.
aws ssm start-session --region us-west-2 --target <instance id from `tofu output`>
```

A stopped worker finished only if its log ends with the pipeline's
`TOTAL TIME TO LOAD` summary. Otherwise it gave up, and the log shows why. Fix the
problem and run `aws ec2 start-instances` on it, or `tofu apply` with a new image.

## Clean up

When every worker is stopped and the run looks complete, run `tofu destroy`. That
removes the workers, the security group, the log group and the role attachments.
The output on source.coop is not affected.
