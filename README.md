# MinerU2.5 on Databricks — 2 Deployment Patterns

Benchmarks `opendatalab/MinerU2.5-2509-1.2B` across 2 vLLM serving patterns on Databricks.

| Option | Pattern | Inference engine |
|--------|---------|-----------------|
| vLLM_Batch | Triggered batch job | `vLLM` (pay-per-use, cold start) |
| vLLM_RT | Continuous job + vLLM driver proxy | `vLLM` (always-hot, low latency) |

---

## Cluster Requirements

| Notebook | GPU? | Instance | Runtime | Notes |
|----------|------|----------|---------|-------|
| `setup/00_download_model` | No | Any CPU | 15.4 ML GPU | Downloads ~3 GB model |
| `setup/01_prepare_test_cases` | No | Any CPU | 15.4 ML GPU | Generates PDF test fixtures |
| `vllm_batch/notebook` | **Yes** | g5.2xlarge | **15.4 ML GPU** | Runs vLLM batch inference |
| `vllm_batch/tests` | No | Serverless | 15.4 ML GPU | Enqueues PDFs + polls Delta queue |
| `vllm_rt/notebook` | **Yes** | g5.2xlarge | **15.4 ML GPU** | Continuous job; launches vLLM HTTP server |
| `vllm_rt/tests` | No | Serverless | 15.4 ML GPU | HTTP requests to vLLM driver proxy |
| `comparison/notebook` | No | Serverless | 15.4 ML GPU | Reads perf table, renders charts |

**GPU cluster spec** (all GPU notebooks):
- Instance: `g5.2xlarge` (AWS) — 1x NVIDIA A10G, 24 GB GPU RAM, 32 GB RAM
- Runtime: `15.4.x-gpu-ml-scala2.12` (Databricks ML GPU, Python 3.11)
- Workers: 0 (single-node; `spark.master = local[*, 4]`)
- No cluster-level libraries — all packages installed via `%pip install` inside notebooks

---

## Setup

Run in order:

1. **`setup/00_download_model`** — GPU cluster — downloads model to Unity Catalog Volume
2. **`setup/01_prepare_test_cases`** — CPU/serverless — generates TC1-TC4 PDFs to Volume
3. For each option: run the `notebook` to deploy, then `tests` to benchmark

Or use the orchestrator to run everything end-to-end:

```bash
python3 -u scripts/orchestrate.py
```

---

## Config

Edit `config.yaml` to change catalog, schema, volume, or model settings. All notebooks read from this file — no hardcoded values.

```yaml
catalog:            <your_catalog>
schema:             <your_schema>
volume:             data
model_subpath:      models
hf_model_id:        opendatalab/MinerU2.5-2509-1.2B
test_cases_subpath: test_cases
queue_table:        mineru_queue
perf_table:         mineru_perf_results
vllm_port:          7777
vllm_model_name:    mineru2.5
```

---

## Upload to Databricks

```bash
databricks sync . /Workspace/Users/<your_email>/doc_parsing_mineru_databricks \
  --profile=<your_profile> --exclude .git --exclude .DS_Store --exclude __pycache__ --full
```
