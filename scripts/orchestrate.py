#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/orchestrate.py — Parallel end-to-end pipeline for mineru-databricks.

Flow:
  1  Delete old jobs (by name)
  2  Re-upload workspace (config.yaml + all notebooks)
  3  setup/01_prepare_test_cases (CPU, sequential — all options depend on it)
  4  [PARALLEL] Deploy + test vLLM_Batch and vLLM_RT simultaneously:
       vLLM_Batch: create job → vllm_batch/tests (CPU, job_id param)
       vLLM_RT: create continuous job → resolve cluster_id → vllm_rt/tests (CPU)
  5  comparison/notebook (CPU, after all options complete)

Usage:
    python3 -u scripts/orchestrate.py

Runtime versions:
    CPU: Serverless compute
    GPU (vLLM_Batch, vLLM_RT): 15.4.x-gpu-ml-scala2.12  (vLLM 0.7.3 compatible)

Key design decisions:
  - Each submit_notebook / poll_run call creates its OWN WorkspaceClient.
    The SDK is not guaranteed thread-safe; sharing one client across threads
    causes the .result() polling timeout to be silently ignored (hits 5-min
    default instead of the specified hours-long timeout).
  - Manual polling loop replaces .result() for full control and diagnostics.
  - On any run failure the error_trace is fetched and printed immediately.
"""

import os, subprocess, sys, time, yaml
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from databricks.sdk import WorkspaceClient
from databricks.sdk.service import jobs as j
from databricks.sdk.service import compute as c
from databricks.sdk.service.jobs import RunLifeCycleState, RunResultState

PROFILE       = os.environ.get("DATABRICKS_CONFIG_PROFILE", "DEFAULT")
os.environ["DATABRICKS_CONFIG_PROFILE"] = PROFILE
os.environ["PYTHONUNBUFFERED"] = "1"
WORKSPACE_DIR = os.environ.get("DATABRICKS_WORKSPACE_DIR", "/Users/<your_email>/doc_parsing_mineru_databricks")
USER_EMAIL    = os.environ.get("DATABRICKS_USER_EMAIL", "<your_email>")

_root = Path(__file__).parent.parent
cfg   = yaml.safe_load(open(_root / "config.yaml"))
CATALOG = cfg["catalog"]
SCHEMA  = cfg["schema"]

# Main client for single-threaded operations (steps 1-3, step 5)
w    = WorkspaceClient(profile=PROFILE)
HOST = w.config.host.rstrip("/")

print(f"Connected : {HOST}", flush=True)
print(f"Catalog   : {CATALOG}.{SCHEMA}", flush=True)
print(f"Model at  : /Volumes/{CATALOG}/{SCHEMA}/{cfg['volume']}/{cfg['model_subpath']}", flush=True)

# ── Cluster specs ─────────────────────────────────────────────────────────────

# vLLM options need MLR 15.4 (vLLM has package conflicts with 16.4's Python 3.12)
VLLM_GPU_CLUSTER = c.ClusterSpec(
    spark_version="15.4.x-gpu-ml-scala2.12",
    node_type_id="g5.2xlarge",
    num_workers=0,
    spark_conf={"spark.master": "local[*,4]"},
    data_security_mode=c.DataSecurityMode.SINGLE_USER,
    single_user_name=USER_EMAIL,
    aws_attributes=c.AwsAttributes(
        availability=c.AwsAvailability.ON_DEMAND,
    ),
)

# CPU tasks use serverless compute (classic clusters have Spark SQL hangs).
CPU_CLUSTER = None   # sentinel — submit_notebook uses serverless when cluster=None

# ── Resources to clean up ─────────────────────────────────────────────────────

CLEANUP_JOB_NAMES = {
    "mineru-vllm-batch-job",
    "mineru-vllm-rt-job",
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def banner(msg: str, prefix: str = "") -> None:
    tag = f"[{prefix}] " if prefix else ""
    print(f"\n{'='*64}\n{tag}{msg}\n{'='*64}", flush=True)


def log(msg: str, prefix: str = "") -> None:
    tag = f"[{prefix}] " if prefix else ""
    print(f"  {tag}{msg}", flush=True)


def _make_client() -> WorkspaceClient:
    """Create a fresh WorkspaceClient for the calling thread."""
    return WorkspaceClient(profile=PROFILE)


def _fetch_run_output(wc: WorkspaceClient, run_id: int, prefix: str = "") -> None:
    """Fetch and print run output/error_trace for post-failure diagnostics."""
    try:
        run = wc.jobs.get_run(run_id=run_id)
        for task in (run.tasks or []):
            task_run_id = getattr(task, "run_id", None)
            if task_run_id is None:
                continue
            try:
                out = wc.jobs.get_run_output(run_id=task_run_id)
                if out and out.error:
                    log(f"  Task '{task.task_key}' error : {out.error}", prefix)
                if out and out.error_trace:
                    trace = out.error_trace[:4000]
                    log(f"  Error trace:\n{trace}", prefix)
                if out and out.notebook_output and out.notebook_output.result:
                    log(f"  Notebook result: {out.notebook_output.result[:800]}", prefix)
            except Exception as e2:
                log(f"  Could not get output for task '{task.task_key}': {e2}", prefix)
    except Exception as e:
        log(f"  Could not fetch run {run_id} details: {e}", prefix)


def poll_run(
    wc: WorkspaceClient,
    run_id: int,
    timeout_seconds: int,
    prefix: str = "",
) -> j.BaseRun:
    """
    Poll run until it reaches a terminal state.
    Raises RuntimeError on failure, TimeoutError on timeout.
    Prints progress every 60 s.
    """
    terminal = {
        RunLifeCycleState.TERMINATED,
        RunLifeCycleState.SKIPPED,
        RunLifeCycleState.INTERNAL_ERROR,
    }
    deadline    = time.time() + timeout_seconds
    interval    = 60   # poll every 60 s
    next_log    = time.time() + interval

    while True:
        run = wc.jobs.get_run(run_id=run_id)
        lc  = run.state.life_cycle_state if run.state else None

        if lc in terminal:
            result_state = run.state.result_state if run.state else None
            if lc == RunLifeCycleState.TERMINATED and result_state == RunResultState.SUCCESS:
                return run
            # Failed — fetch diagnostics before raising
            _fetch_run_output(wc, run_id, prefix)
            state_msg = run.state.state_message if run.state else ""
            raise RuntimeError(
                f"failed to reach TERMINATED/SUCCESS, got {lc} "
                f"({result_state}): {state_msg}"
            )

        remaining = deadline - time.time()
        if remaining <= 0:
            raise TimeoutError(f"Run {run_id} timed out after {timeout_seconds}s")

        now = time.time()
        if now >= next_log:
            log(f"  run {run_id}: {lc} — {int(remaining)}s remaining", prefix)
            next_log = now + interval

        time.sleep(min(30, max(remaining, 0)))


def submit_notebook(
    name: str,
    notebook: str,
    cluster: c.ClusterSpec = None,
    params: dict = None,
    timeout_seconds: int = 7200,
    prefix: str = "",
) -> j.BaseRun:
    """
    Submit a one-time notebook run and poll until completion.
    Creates its own WorkspaceClient (thread-safe).
    If cluster is None, uses serverless compute.
    """
    wc      = _make_client()
    nb_path = f"{WORKSPACE_DIR}/{notebook}"
    mode    = "GPU cluster" if cluster else "serverless"
    log(f"Submitting {notebook} ({mode}) ...", prefix)

    if cluster:
        # Classic cluster (GPU)
        task = j.SubmitTask(
            task_key="main",
            notebook_task=j.NotebookTask(
                notebook_path=nb_path,
                base_parameters=params or {},
            ),
            new_cluster=cluster,
            timeout_seconds=timeout_seconds,
        )
        submitted = wc.jobs.submit(run_name=name, tasks=[task])
    else:
        # Serverless compute
        task = j.SubmitTask(
            task_key="main",
            notebook_task=j.NotebookTask(
                notebook_path=nb_path,
                base_parameters=params or {},
            ),
            environment_key="default",
            timeout_seconds=timeout_seconds,
        )
        submitted = wc.jobs.submit(
            run_name=name,
            tasks=[task],
            environments=[j.JobEnvironment(environment_key="default")],
        )
    run_id = submitted.run_id
    log(f"  Run {run_id} → {HOST}/jobs/runs/{run_id}", prefix)

    result = poll_run(wc, run_id, timeout_seconds, prefix)
    log(f"Done → {HOST}/jobs/runs/{result.run_id}", prefix)
    return result


def get_running_run_id(job_id: int, wait_seconds: int = 600, prefix: str = "") -> int:
    """Wait for a job to have a RUNNING run; return its run_id."""
    wc       = _make_client()
    deadline = time.time() + wait_seconds
    while time.time() < deadline:
        runs = list(wc.jobs.list_runs(job_id=job_id, active_only=True))
        for run in runs:
            if run.state and run.state.life_cycle_state == RunLifeCycleState.RUNNING:
                return run.run_id
        log(f"Waiting for running run (job {job_id}) ...", prefix)
        time.sleep(15)
    raise TimeoutError(f"Job {job_id} not RUNNING within {wait_seconds}s")


def get_cluster_id(run_id: int, timeout: int = 600, prefix: str = "") -> str:
    """Wait for a cluster_id to be assigned to the first task of the run."""
    wc       = _make_client()
    deadline = time.time() + timeout
    while time.time() < deadline:
        run = wc.jobs.get_run(run_id=run_id)
        for t in (run.tasks or []):
            if t.cluster_instance and t.cluster_instance.cluster_id:
                return t.cluster_instance.cluster_id
        time.sleep(10)
    raise TimeoutError(f"cluster_id not assigned for run {run_id} within {timeout}s")


# ── Step 1: Cleanup ───────────────────────────────────────────────────────────

banner("Step 1: Delete old resources")

# Clean up previous-run jobs by name (filter per name to avoid listing all jobs)
for job_name in CLEANUP_JOB_NAMES:
    try:
        for job in w.jobs.list(name=job_name):
            actual_name = job.settings.name if job.settings else ""
            if actual_name != job_name:
                continue
            try:
                for run in w.jobs.list_runs(job_id=job.job_id, active_only=True):
                    try:
                        w.jobs.cancel_run(run_id=run.run_id)
                        log(f"Cancelled run {run.run_id}")
                    except Exception:
                        pass
                w.jobs.delete(job_id=job.job_id)
                log(f"Deleted job       : '{job_name}' (id={job.job_id})")
            except Exception as e:
                log(f"Skipped job       : '{job_name}' ({e})")
    except Exception as e:
        log(f"Skipped job search: '{job_name}' ({e})")

# ── Step 2: Upload workspace ──────────────────────────────────────────────────

banner("Step 2: Upload workspace (config.yaml + all notebooks)")
subprocess.run(
    [
        "databricks", "sync",
        str(_root),
        f"/Workspace{WORKSPACE_DIR}",
        "--profile", PROFILE,
        "--exclude", ".git",
        "--exclude", ".DS_Store",
        "--exclude", "__pycache__",
        "--exclude", "mlflow.db",
        "--exclude", ".databricks",
        "--full",
    ],
    check=True,
    cwd=str(_root),
)
log(f"Uploaded → /Workspace{WORKSPACE_DIR}")

# ── Step 3: Setup test cases ────────────────────────────────────────────────

banner("Step 3: setup/01_prepare_test_cases (CPU — generates TC PDFs)")
submit_notebook(
    "mineru-setup-test-cases",
    "setup/01_prepare_test_cases",
    CPU_CLUSTER,
    timeout_seconds=600,
)

# ── Steps 4-5: Deploy + test C and D IN PARALLEL ─────────────────────────────

banner("Steps 4-5: Deploy + test vLLM_Batch and vLLM_RT IN PARALLEL")
log("Starting vLLM_Batch, vLLM_RT simultaneously. Total time = slowest option.")

results: dict = {}   # disjoint keys written from each thread — no lock needed


def run_vllm_batch() -> str:
    px = "vLLM_Batch"
    try:
        banner("vLLM_Batch: Create batch-vLLM job", px)
        wc = _make_client()
        job_c = wc.jobs.create(
            name="mineru-vllm-batch-job",
            tasks=[j.Task(
                task_key="main",
                notebook_task=j.NotebookTask(
                    notebook_path=f"{WORKSPACE_DIR}/vllm_batch/notebook",
                ),
                new_cluster=VLLM_GPU_CLUSTER,
                timeout_seconds=7200,
            )],
        )
        log(f"Created job ID: {job_c.job_id}", px)
        results["job_batch_id"] = job_c.job_id

        banner("vLLM_Batch: Tests (enqueue + trigger + poll)", px)
        submit_notebook(
            "mineru-vllm-batch-tests", "vllm_batch/tests", CPU_CLUSTER,
            params={"job_id": str(job_c.job_id)},
            timeout_seconds=7200, prefix=px,
        )
        results["vLLM_Batch"] = "DONE"
        return "vLLM_Batch: OK"
    except Exception as e:
        results["vLLM_Batch"] = f"FAILED: {e}"
        log(f"FAILED: {e}", px)
        return "vLLM_Batch: FAILED"


def run_vllm_rt() -> str:
    px = "vLLM_RT"
    try:
        banner("vLLM_RT: Create continuous vLLM driver proxy job", px)
        wc = _make_client()
        job_d = wc.jobs.create(
            name="mineru-vllm-rt-job",
            tasks=[j.Task(
                task_key="main",
                notebook_task=j.NotebookTask(
                    notebook_path=f"{WORKSPACE_DIR}/vllm_rt/notebook",
                ),
                new_cluster=VLLM_GPU_CLUSTER,
                timeout_seconds=0,
            )],
            continuous=j.Continuous(pause_status=j.PauseStatus.UNPAUSED),
        )
        log(f"Created continuous job ID: {job_d.job_id}", px)
        results["job_rt_id"] = job_d.job_id

        log("Waiting for RUNNING run (may take 10 min for g5.2xlarge) ...", px)
        run_id_d = get_running_run_id(job_d.job_id, wait_seconds=600, prefix=px)
        log(f"RUNNING run ID: {run_id_d}", px)

        log("Waiting for cluster_id ...", px)
        cluster_id_d = get_cluster_id(run_id_d, timeout=600, prefix=px)
        log(f"Cluster ID: {cluster_id_d}", px)
        log(f"Proxy URL: {HOST}/driver-proxy-api/o/0/{cluster_id_d}/{cfg['vllm_port']}/v1", px)
        results["cluster_rt_id"] = cluster_id_d

        banner("vLLM_RT: Tests (wait for vLLM + query)", px)
        submit_notebook(
            "mineru-vllm-rt-tests", "vllm_rt/tests", CPU_CLUSTER,
            params={"cluster_id": cluster_id_d},
            timeout_seconds=7200, prefix=px,
        )
        results["vLLM_RT"] = "DONE"
        log(f"NOTE: Job {job_d.job_id} is still running. Pause it when done.", px)
        return "vLLM_RT: OK"
    except Exception as e:
        results["vLLM_RT"] = f"FAILED: {e}"
        log(f"FAILED: {e}", px)
        return "vLLM_RT: FAILED"


with ThreadPoolExecutor(max_workers=2) as executor:
    futures = {
        executor.submit(run_vllm_batch): "vLLM_Batch",
        executor.submit(run_vllm_rt): "vLLM_RT",
    }
    for future in as_completed(futures):
        opt = futures[future]
        try:
            outcome = future.result()
            log(f"{opt} completed: {outcome}")
        except Exception as e:
            log(f"{opt} raised exception: {e}")

# ── Step 6: Comparison ────────────────────────────────────────────────────────

banner("Step 6: comparison/notebook (reads perf table + renders charts)")
try:
    submit_notebook(
        "mineru-comparison",
        "comparison/notebook",
        CPU_CLUSTER,
        timeout_seconds=600,
    )
except Exception as e:
    log(f"Comparison skipped: {e}")
    log("  If this is an IP ACL error, ask a workspace admin to allowlist your IP")
    log(f"  or run the notebook manually at {HOST}/browse/folders/")

# ── Summary ───────────────────────────────────────────────────────────────────

banner("PIPELINE COMPLETE")
print(f"\nPerf table : {CATALOG}.{SCHEMA}.{cfg['perf_table']}", flush=True)
print(f"Workspace  : {HOST}/browse/folders/", flush=True)
for opt, job_key in [("vLLM_Batch", "job_batch_id"), ("vLLM_RT", "job_rt_id")]:
    status  = results.get(opt, "NOT RUN")
    job_id  = results.get(job_key, "")
    job_str = f"  (job {job_id})" if job_id else ""
    print(f"  {opt}: {status}{job_str}", flush=True)
if results.get("job_rt_id"):
    print(
        f"\nNOTE: vLLM_RT (job {results['job_rt_id']}) is still running. "
        "Pause it manually when done to avoid GPU charges.",
        flush=True,
    )
