#!/usr/bin/env bash
# Diagnose GPU/CPU/memory for RLlib training (host WSL + container).
set -u
cd "$(dirname "$0")/.."

echo '--- torch in container (rllib-dev) ---'
docker exec rllib-dev python /workspace/tools/check_gpu.py 2>&1 | tail -8

echo '--- ray: pending / resource warnings (container) ---'
docker exec rllib-dev bash -c '
  grep -ih "pending\|insufficient\|cannot be scheduled\|GPU" /tmp/ray/session_latest/logs/python-core-driver-*.log /tmp/ray/session_latest/logs/*.err 2>/dev/null | tail -8
  echo "--- raylet resources ---"
  grep -ih "resources\|GPU\|CPU" /tmp/ray/session_latest/logs/raylet.out 2>/dev/null | head -5
  echo "--- processes by CPU ---"
  ps -eo pid,etime,pcpu,comm --sort=-pcpu | head -10
' 2>&1

echo '--- container stats ---'
docker stats --no-stream --format 'table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}' 2>&1

echo '--- WSL memory ---'
free -g

echo '--- .wslconfig ---'
cat /mnt/c/Users/*/.wslconfig 2>/dev/null || echo "(none)"

echo '--- top mem procs ---'
ps -eo pid,rss,comm --sort=-rss | head -8
