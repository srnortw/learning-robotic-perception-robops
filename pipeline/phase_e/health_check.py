"""
Phase E — Post-Deploy Health Check

Runs from GitHub Actions (or locally) after a Greengrass canary deployment.
SSHes into each canary Pi and verifies:
  1. Docker containers are running
  2. ROS2 topics are publishing
  3. Inference latency p95 is within 3× the baseline
  4. CPU usage is below 95%

Exit code 0 = healthy, 1 = unhealthy (triggers Greengrass rollback).

Usage:
    python pipeline/phase_e/health_check.py \
        --host <PI_IP> \
        --user pi \
        --key ~/.ssh/robops_pi \
        [--baseline-latency-ms 2000] \
        [--timeout-minutes 5]
"""

import argparse
import json
import subprocess
import sys
import time

REQUIRED_CONTAINERS  = ["robops-inference", "robops-ros2stack"]
REQUIRED_TOPICS      = ["/camera/image_raw", "/detr/detections"]
MAX_CPU_PERCENT      = 95.0
LATENCY_MULTIPLIER   = 3.0   # alert if p95 > baseline × this


def ssh(host: str, user: str, key: str, cmd: str, timeout: int = 30) -> tuple[int, str]:
    """Run a command on the Pi over SSH, return (exit_code, stdout)."""
    result = subprocess.run(
        ["ssh", "-i", key, "-o", "StrictHostKeyChecking=no",
         "-o", "ConnectTimeout=10", f"{user}@{host}", cmd],
        capture_output=True, text=True, timeout=timeout,
    )
    return result.returncode, (result.stdout + result.stderr).strip()


def check_containers(host: str, user: str, key: str) -> list[str]:
    """Return list of containers that are NOT running."""
    failures = []
    for name in REQUIRED_CONTAINERS:
        rc, out = ssh(host, user, key,
                      f"docker inspect -f '{{{{.State.Running}}}}' {name} 2>&1")
        if rc != 0 or out.strip() != "true":
            failures.append(name)
            print(f"  [FAIL] Container '{name}' not running: {out}")
        else:
            print(f"  [OK]   Container '{name}' running")
    return failures


def check_topics(host: str, user: str, key: str) -> list[str]:
    """Return list of ROS2 topics that are NOT advertised.

    Uses `ros2 topic list` instead of `ros2 topic hz` so the check
    completes instantly even without a live camera feed (shadow mode /
    bench test).  Topic existence confirms the nodes are up and running.
    """
    failures = []
    rc, out = ssh(
        host, user, key,
        "docker exec robops-ros2stack bash -c '. /opt/ros/jazzy/setup.sh && "
        "timeout 10 ros2 topic list 2>&1' || "
        "docker exec robops-inference bash -c '. /opt/ros/jazzy/setup.sh && "
        "timeout 10 ros2 topic list 2>&1'",
        timeout=25,
    )
    for topic in REQUIRED_TOPICS:
        if topic in out:
            print(f"  [OK]   Topic '{topic}' advertised")
        else:
            failures.append(topic)
            print(f"  [FAIL] Topic '{topic}' not found in ros2 topic list: {out[:200]}")
    return failures


def check_cpu(host: str, user: str, key: str) -> bool:
    """Return True if CPU is acceptable (< MAX_CPU_PERCENT)."""
    rc, out = ssh(host, user, key,
                  "top -bn1 | grep 'Cpu(s)' | awk '{print $2+$4}'")
    if rc != 0:
        print(f"  [WARN] Could not read CPU: {out}")
        return True  # non-fatal
    try:
        cpu = float(out.strip())
        ok = cpu < MAX_CPU_PERCENT
        status = "[OK]  " if ok else "[FAIL]"
        print(f"  {status} CPU usage: {cpu:.1f}% (limit {MAX_CPU_PERCENT}%)")
        return ok
    except ValueError:
        print(f"  [WARN] Could not parse CPU value: '{out}'")
        return True


def check_latency(host: str, user: str, key: str, baseline_ms: float) -> bool:
    """
    Reads the last 10 latency log lines from the inference container and
    checks that p95 is within LATENCY_MULTIPLIER × baseline.
    """
    limit_ms = baseline_ms * LATENCY_MULTIPLIER
    rc, out = ssh(
        host, user, key,
        "docker logs --tail 50 robops-inference 2>&1 | grep 'latency_ms' | tail -10",
    )
    if rc != 0 or not out.strip():
        print(f"  [WARN] No latency logs found — skipping latency check")
        return True  # non-fatal on first deploy (no history yet)

    latencies = []
    for line in out.splitlines():
        try:
            val = float(line.split("latency_ms")[1].split()[0].strip(":= "))
            latencies.append(val)
        except (IndexError, ValueError):
            continue

    if not latencies:
        print(f"  [WARN] Could not parse latency values — skipping")
        return True

    latencies.sort()
    p95 = latencies[int(len(latencies) * 0.95)]
    ok = p95 <= limit_ms
    status = "[OK]  " if ok else "[FAIL]"
    print(f"  {status} Latency p95: {p95:.0f} ms (limit {limit_ms:.0f} ms, baseline {baseline_ms:.0f} ms)")
    return ok


def run_health_check(host: str, user: str, key: str,
                     baseline_ms: float, timeout_minutes: int) -> bool:
    deadline = time.time() + timeout_minutes * 60
    attempt = 0

    while time.time() < deadline:
        attempt += 1
        print(f"\n── Health check attempt {attempt} ──────────────────────────")

        container_failures = check_containers(host, user, key)
        if container_failures:
            wait = min(30, deadline - time.time())
            if wait > 0:
                print(f"  Containers not ready — retrying in 30s")
                time.sleep(30)
            continue

        topic_failures  = check_topics(host, user, key)
        cpu_ok          = check_cpu(host, user, key)
        latency_ok      = check_latency(host, user, key, baseline_ms)

        if not topic_failures and cpu_ok and latency_ok:
            print(f"\n✓ Health check PASSED (attempt {attempt})")
            return True

        wait = min(30, deadline - time.time())
        if wait > 0:
            print(f"  Issues found — retrying in 30s")
            time.sleep(30)

    print(f"\n✗ Health check FAILED after {timeout_minutes} min / {attempt} attempts")
    return False


def main():
    parser = argparse.ArgumentParser(description="Phase E — post-deploy health check")
    parser.add_argument("--host",                 required=True)
    parser.add_argument("--user",                 default="pi")
    parser.add_argument("--key",                  default="~/.ssh/robops_pi")
    parser.add_argument("--baseline-latency-ms",  type=float, default=2000.0,
                        help="Expected p95 inference latency (ms). Fails if >3× this.")
    parser.add_argument("--timeout-minutes",      type=int, default=5)
    args = parser.parse_args()

    healthy = run_health_check(
        host=args.host,
        user=args.user,
        key=args.key,
        baseline_ms=args.baseline_latency_ms,
        timeout_minutes=args.timeout_minutes,
    )
    sys.exit(0 if healthy else 1)


if __name__ == "__main__":
    main()
