"""Terminate a RunPod pod by id. Usage: runpod_kill.py <pod_id> [<pod_id> ...]"""
import os, sys
import runpod
runpod.api_key = open(os.path.expanduser("~/.run.pod")).read().strip()
for pod_id in sys.argv[1:]:
    try:
        runpod.terminate_pod(pod_id)
        print(f"terminated {pod_id}")
    except Exception as e:
        print(f"  {pod_id}: {e}")
