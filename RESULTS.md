# Performance Results — MinerU2.5 on Databricks

Benchmark run: **2026-03-07**

Model: `opendatalab/MinerU2.5-2509-1.2B` on g5.2xlarge or equivalent (1x NVIDIA A10G, 24 GB VRAM)

Runtime: Databricks ML Runtime 15.4 (`15.4.x-gpu-ml-scala2.12`), vLLM 0.7.3

---

## Test Cases

| ID | Description | Pages |
|----|-------------|-------|
| TC1 | Simple single-page text | 1 |
| TC2 | Multi-page with headings and bullets | 3 |
| TC3 | Single-page with tabular data | 1 |
| TC4 | Long document with structured sections | 10 |

---

## Per-Document Latency

| Option | TC1 (1pg) | TC2 (3pg) | TC3 (1pg) | TC4 (10pg) |
|--------|-----------|-----------|-----------|------------|
| vLLM_Batch | 844.9s | 845.2s | 845.5s | 845.8s |
| vLLM_Real_Time | 3.6s | 3.0s | 1.0s | 10.4s |

> **Note:** vLLM_Batch latency includes cold start (cluster startup + model loading). All four test cases complete within ~1s of each other because the cold start dominates — actual inference is only a few seconds.

---

## Summary

| Dimension | vLLM_Batch | vLLM_Real_Time |
|-----------|------------|---------|
| **Pattern** | Triggered batch job | Continuous job + driver proxy |
| **Cold Start** | ~845s (cluster + model load) | None (always running) |
| **Per-Page Latency** | ~1s (after warm) | ~1s |
| **Throughput** | ~60 pages/min (after warm) | ~60 pages/min |
| **Scalability** | Single job per cluster | Single GPU, not scalable |
| **Cost** | Low (pay-per-use, idle = $0) | High (always-on GPU) |
| **Best For** | Scheduled batch workloads | Demos, prototyping, interactive use |

---

## Recommendations

- **vLLM_Batch**: Use for scheduled/recurring workloads where latency tolerance is high and cost efficiency matters. The cluster spins up only when there are pending requests — no idle cost.
- **vLLM_Real_Time**: Use for demos, prototyping, or interactive applications that need sub-second per-page latency. Pause the job when not in use to stop GPU billing.
