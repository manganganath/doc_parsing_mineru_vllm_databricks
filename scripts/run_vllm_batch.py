"""
vLLM_Batch local runner — enqueues TC1-TC4 to Delta queue via Databricks Connect,
triggers the vLLM batch job, polls for results.

Run:
    python scripts/run_vllm_batch.py --job-id <JOB_ID>

Prerequisites:
    - pip install databricks-connect databricks-sdk pymupdf pyyaml
    - Databricks CLI profile configured (~/.databrickscfg) or DATABRICKS_CONFIG_PROFILE env var
    - setup/01_prepare_test_cases notebook already run (TC PDFs in Volume)
    - mineru-vllm-batch-job created in Databricks (see vllm_batch/notebook.py)
"""
import argparse, base64, os, time, uuid, yaml
from datetime import datetime, timezone
from pathlib import Path

from databricks.connect import DatabricksSession
from databricks.sdk import WorkspaceClient
from pyspark.sql import functions as F
import fitz

PROFILE       = os.environ.get("DATABRICKS_CONFIG_PROFILE", "DEFAULT")
POLL_INTERVAL = 15
JOB_TIMEOUT   = 1800

_root = Path(__file__).parent.parent
cfg   = yaml.safe_load(open(_root / "config.yaml"))

CATALOG     = cfg["catalog"]
SCHEMA      = cfg["schema"]
QUEUE_TABLE = f"{CATALOG}.{SCHEMA}.{cfg['queue_table']}"
TC_PATH     = f"/Volumes/{CATALOG}/{SCHEMA}/{cfg['volume']}/{cfg['test_cases_subpath']}"

spark = DatabricksSession.builder.profile(PROFILE).serverless(True).getOrCreate()
w     = WorkspaceClient(profile=PROFILE)
TOKEN = w.config.token
HOST  = w.config.host.rstrip("/")
print(f"Connected : {HOST}")
print(f"Queue     : {QUEUE_TABLE}")


def read_volume_file(path: str) -> bytes:
    import requests
    url = f"{HOST}/api/2.0/fs/files{path}"
    r   = requests.get(url, headers={"Authorization": f"Bearer {TOKEN}"})
    r.raise_for_status()
    return r.content


def enqueue_pdf(pdf_bytes: bytes) -> tuple:
    doc   = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages = len(doc); doc.close()
    rid   = str(uuid.uuid4())
    now   = datetime.now(timezone.utc)
    spark.createDataFrame([{
        "request_id": rid,
        "pdf_base64": base64.b64encode(pdf_bytes).decode(),
        "status": "pending", "markdown": "", "char_count": 0,
        "error": "", "created_at": now, "updated_at": now,
    }]).write.mode("append").saveAsTable(QUEUE_TABLE)
    return rid, pages


def poll_result(rid: str, deadline: float) -> dict:
    while time.time() < deadline:
        row = (spark.table(QUEUE_TABLE)
               .filter(F.col("request_id") == rid)
               .select("status", "markdown", "char_count", "error")
               .first())
        if row and row.status in ("done", "error"):
            return row.asDict()
        time.sleep(POLL_INTERVAL)
    return {"status": "timeout", "markdown": "", "char_count": 0, "error": "timed out"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-id", type=int, required=True, help="Databricks job ID for mineru-vllm-batch-job")
    args = parser.parse_args()

    tc_files    = {"TC1": "tc1.pdf", "TC2": "tc2.pdf", "TC3": "tc3.pdf", "TC4": "tc4.pdf"}
    request_ids = {}
    page_counts = {}

    print("\nEnqueuing test PDFs ...")
    for tc_id, fname in tc_files.items():
        pdf_bytes          = read_volume_file(f"{TC_PATH}/{fname}")
        rid, pages         = enqueue_pdf(pdf_bytes)
        request_ids[tc_id] = rid
        page_counts[tc_id] = pages
        print(f"  {tc_id} ({pages} pages) → {rid}")

    print(f"\nTriggering job {args.job_id} ...")
    t_trigger = time.time()
    run       = w.jobs.run_now(job_id=args.job_id)
    print(f"  Run ID  : {run.run_id}")
    print(f"  Monitor : {HOST}/jobs/{args.job_id}/runs/{run.run_id}")

    print(f"\nPolling (timeout={JOB_TIMEOUT}s) ...")
    deadline = time.time() + JOB_TIMEOUT
    results  = {}

    for tc_id, rid in request_ids.items():
        result  = poll_result(rid, deadline)
        latency = time.time() - t_trigger
        if result["status"] == "done":
            ok           = result["char_count"] > 5
            results[tc_id] = "PASS" if ok else f"FAIL: {result['char_count']} chars"
        else:
            results[tc_id] = f"FAIL: {result['status']} — {result['error']}"
        print(f"  {tc_id}: {results[tc_id]} ({latency:.1f}s)")

    passed = sum(1 for v in results.values() if v == "PASS")
    print(f"\n{passed}/{len(results)} tests passed")


if __name__ == "__main__":
    main()
