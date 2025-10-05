#!/usr/bin/env python3
import time
from datetime import datetime, timezone
import requests

API_BASE = "http://localhost:5015"

def main():
    url = "https://openreview.net/forum?id=H3at5y8VFW"

    # Kick off the job
    resp = requests.post(f"{API_BASE}/process", json={"openreview_url": url})
    resp.raise_for_status()
    data = resp.json()
    job_id = data["job_id"]
    print(f"Started job: {job_id}")

    # Poll for status and percent
    last_percent = -1.0
    job_started_at = None
    while True:
        r = requests.get(f"{API_BASE}/jobs/{job_id}", timeout=30)
        r.raise_for_status()
        s = r.json()

        status = s.get("status", "unknown")
        phase = s.get("phase", "unknown")
        message = s.get("message", phase)
        percent = s.get("percent", 0.0)
        started_at = s.get("started_at")
        now = datetime.now(timezone.utc).isoformat()
        if started_at and job_started_at is None:
            job_started_at = started_at

        if percent != last_percent:
            print(f"[{now}] [{status}] {message} percent={percent:.1f}% (phase={phase}) started_at={started_at}")
            last_percent = percent

        if status in ("completed", "failed"):
            break

        time.sleep(0.5)

    if status == "completed":
        results = s.get("results", [])
        submission_id = s.get("submission_id")
        pdf_url = s.get("pdf_url")
        # Summaries
        num_claims = len(results)
        label_counts = {}
        total_evidence_refs = 0
        for item in results:
            ver = (item.get("verification") or {}).get("result")
            if ver:
                label_counts[ver] = label_counts.get(ver, 0) + 1
            ev_ids = item.get("evidence_ids") or item.get("evidence") or []
            total_evidence_refs += len(ev_ids)
        hl = s.get("highlighting_map") or {}
        evidence_registry = (hl.get("evidence_registry") or {})
        num_unique_evidence = len(evidence_registry)
        # Print concise summary line
        print(
            f"[{datetime.now(timezone.utc).isoformat()}] Completed in {s.get('processing_time', '?')}s; "
            f"submission={submission_id}; claims={num_claims}; unique_evidence={num_unique_evidence}; "
            f"evidence_refs={total_evidence_refs}; labels={label_counts}; pdf={pdf_url}; started_at={job_started_at}"
        )
    else:
        print(f"[{datetime.now(timezone.utc).isoformat()}] Job failed: {s.get('error', 'unknown error')}")

if __name__ == "__main__":
    main()