"""
vLLM_RT local runner — ensures the continuous vLLM job is running,
waits for vLLM readiness on the driver proxy, then queries TC1-TC4.

Run:
    python scripts/run_vllm_rt.py --job-id <JOB_ID>

Prerequisites:
    - pip install databricks-connect databricks-sdk pymupdf pyyaml requests
    - Databricks CLI profile configured (~/.databrickscfg) or DATABRICKS_CONFIG_PROFILE env var
    - setup/01_prepare_test_cases notebook already run (TC PDFs in Volume)
    - mineru-vllm-rt-job created as a continuous Databricks Job

Notes:
    - HTTP 401 from driver proxy means the cluster is terminated, not an auth error
    - The job keeps running after this script completes; pause it manually when done
"""
import argparse, base64, os, time, requests, yaml
from pathlib import Path

from databricks.connect import DatabricksSession
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.jobs import RunLifeCycleState
import fitz

PROFILE        = os.environ.get("DATABRICKS_CONFIG_PROFILE", "DEFAULT")
POLL_INTERVAL  = 15
READY_TIMEOUT  = 1200
TEST_TIMEOUT   = 120

_root = Path(__file__).parent.parent
cfg   = yaml.safe_load(open(_root / "config.yaml"))

CATALOG    = cfg["catalog"]
SCHEMA     = cfg["schema"]
TC_PATH    = f"/Volumes/{CATALOG}/{SCHEMA}/{cfg['volume']}/{cfg['test_cases_subpath']}"
VLLM_PORT  = cfg["vllm_port"]
MODEL_NAME = cfg["vllm_model_name"]

spark = DatabricksSession.builder.profile(PROFILE).serverless(True).getOrCreate()
w     = WorkspaceClient(profile=PROFILE)
TOKEN = w.config.token
HOST  = w.config.host.rstrip("/")
print(f"Connected : {HOST}")


def ensure_job_running(job_id: int) -> int:
    """Return run_id of a RUNNING run; start one if none is running.
    Filters to RUNNING only — list_runs(active_only=True) can return
    PENDING, INTERNAL_ERROR etc. for continuous jobs."""
    runs    = list(w.jobs.list_runs(job_id=job_id, active_only=True))
    running = [r for r in runs if r.state.life_cycle_state == RunLifeCycleState.RUNNING]
    if running:
        print(f"  Found running run: {running[0].run_id}")
        return running[0].run_id
    print(f"  No running run — triggering job {job_id} ...")
    run = w.jobs.run_now(job_id=job_id)
    print(f"  Started run: {run.run_id}")
    return run.run_id


def get_cluster_id(run_id: int, timeout: int = 120) -> str:
    """Poll until the run's cluster_id is assigned."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        r = w.jobs.get_run(run_id=run_id)
        for t in (r.tasks or []):
            if t.cluster_instance and t.cluster_instance.cluster_id:
                return t.cluster_instance.cluster_id
        time.sleep(5)
    raise TimeoutError(f"cluster_id not assigned within {timeout}s for run {run_id}")


def wait_for_vllm(base_url: str, run_id: int, timeout: int) -> None:
    """Poll /v1/models until vLLM returns 200.
    HTTP 401 = cluster is dead (not an auth error) — raise immediately."""
    headers          = {"Authorization": f"Bearer {TOKEN}"}
    deadline         = time.time() + timeout
    last_liveness    = time.time()
    print(f"  Waiting for vLLM at {base_url}/models ...")
    while time.time() < deadline:
        try:
            r = requests.get(f"{base_url}/models", headers=headers, timeout=10)
            if r.status_code == 200:
                print(f"  vLLM ready ({r.json()['data'][0]['id']})")
                return
            if r.status_code == 401:
                raise RuntimeError(f"HTTP 401 from driver proxy — cluster {run_id} is terminated")
            status_msg = f"HTTP {r.status_code}"
        except RuntimeError:
            raise
        except Exception as e:
            status_msg = type(e).__name__

        # Check run liveness every ~60s
        if time.time() - last_liveness > 60:
            run   = w.jobs.get_run(run_id=run_id)
            state = run.state.life_cycle_state
            if state not in (RunLifeCycleState.RUNNING, RunLifeCycleState.PENDING):
                raise RuntimeError(f"Job run {run_id} ended with state {state} — vLLM never started")
            last_liveness = time.time()

        print(f"  {status_msg} — waiting ...")
        time.sleep(POLL_INTERVAL)
    raise TimeoutError(f"vLLM not ready within {timeout}s")


def read_volume_file(path: str) -> bytes:
    url = f"{HOST}/api/2.0/fs/files{path}"
    r   = requests.get(url, headers={"Authorization": f"Bearer {TOKEN}"})
    r.raise_for_status()
    return r.content


def rasterize_pdf(pdf_bytes: bytes, dpi: int = 150) -> list:
    doc   = fitz.open(stream=pdf_bytes, filetype="pdf")
    scale = dpi / 72.0
    mat   = fitz.Matrix(scale, scale)
    pages = [page.get_pixmap(matrix=mat).tobytes("png") for page in doc]
    doc.close()
    return pages


def query_page(base_url: str, img_bytes: bytes) -> str:
    headers = {"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json"}
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64.b64encode(img_bytes).decode()}"}},
            {"type": "text", "text": "Extract all text, tables, and structure from this page. Output as clean markdown."},
        ]}],
        "max_tokens": 2048,
        "temperature": 0.0,
    }
    r = requests.post(f"{base_url}/chat/completions", json=payload, headers=headers, timeout=TEST_TIMEOUT)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-id", type=int, required=True, help="Databricks job ID for mineru-vllm-rt-job")
    args = parser.parse_args()

    print(f"\nEnsuring job {args.job_id} is running ...")
    run_id     = ensure_job_running(args.job_id)
    cluster_id = get_cluster_id(run_id)
    base_url   = f"{HOST}/driver-proxy-api/o/0/{cluster_id}/{VLLM_PORT}/v1"
    print(f"  Cluster ID : {cluster_id}")
    print(f"  Proxy URL  : {base_url}")
    print(f"  Monitor    : {HOST}/jobs/{args.job_id}/runs/{run_id}")

    print(f"\nWaiting for vLLM (timeout={READY_TIMEOUT}s) ...")
    wait_for_vllm(base_url, run_id, READY_TIMEOUT)

    tc_files = {"TC1": "tc1.pdf", "TC2": "tc2.pdf", "TC3": "tc3.pdf", "TC4": "tc4.pdf"}
    results  = {}

    print(f"\nRunning test cases ...")
    for tc_id, fname in tc_files.items():
        print(f"  {tc_id} ...")
        try:
            pdf_bytes  = read_volume_file(f"{TC_PATH}/{fname}")
            page_imgs  = rasterize_pdf(pdf_bytes)
            page_texts = [query_page(base_url, img) for img in page_imgs]
            markdown   = "\n\n---\n\n".join(page_texts) if len(page_texts) > 1 else page_texts[0]
            ok         = len(markdown) > 5
            results[tc_id] = "PASS" if ok else f"FAIL: {len(markdown)} chars"
            print(f"    {results[tc_id]} ({len(markdown)} chars)")
        except Exception as e:
            results[tc_id] = f"FAIL: {e}"
            print(f"    {results[tc_id]}")

    passed = sum(1 for v in results.values() if v == "PASS")
    print(f"\n{passed}/{len(results)} tests passed")
    print(f"\nNote: Job {args.job_id} is still running. Pause it manually when done testing.")


if __name__ == "__main__":
    main()
