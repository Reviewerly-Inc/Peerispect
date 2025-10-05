#!/usr/bin/env python3
import time
import requests

API_BASE = "http://localhost:5015"

def main():
    url = "https://openreview.net/forum?id=odjMSBSWRt"

    # Kick off the job
    resp = requests.post(f"{API_BASE}/process", json={"openreview_url": url})
    resp.raise_for_status()
    data = resp.json()
    job_id = data["job_id"]
    print(f"Started job: {job_id}")

    # Poll for status and percent
    last_percent = -1.0
    while True:
        r = requests.get(f"{API_BASE}/jobs/{job_id}", timeout=30)
        r.raise_for_status()
        s = r.json()

        status = s.get("status", "unknown")
        phase = s.get("phase", "unknown")
        percent = s.get("percent", 0.0)

        if percent != last_percent:
            print(f"[{status}] phase={phase} percent={percent:.1f}%")
            last_percent = percent

        if status in ("completed", "failed"):
            break

        time.sleep(3)

    if status == "completed":
        results = s.get("results", [])
        print(f"Completed in {s.get('processing_time', '?')}s with {len(results)} verification items.")
    else:
        print(f"Job failed: {s.get('error', 'unknown error')}")

if __name__ == "__main__":
    main()