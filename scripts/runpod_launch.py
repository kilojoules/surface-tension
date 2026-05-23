"""Spin up a RunPod pod and write its SSH info to an env file.

Usage:
    python3 scripts/runpod_launch.py --gpu "NVIDIA A100-SXM4-80GB" \
        --name st-rank-curve --env-file vast_rank_curve.env \
        --disk 150 --cloud COMMUNITY

Writes "<pod_id> <ip> <port>" to --env-file so the existing rsync+ssh+watchdog
flow can read it. Use scripts/runpod_kill.py <pod_id> when done.
"""
import argparse, os, sys, time
import runpod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", default="NVIDIA A100-SXM4-80GB",
                    help="GPU type id (e.g. 'NVIDIA A100-SXM4-80GB', 'NVIDIA A100 80GB PCIe', 'NVIDIA H100 80GB HBM3')")
    ap.add_argument("--name", required=True, help="pod name (for the UI)")
    ap.add_argument("--env-file", required=True, help="path to write '<pod_id> <ip> <port>'")
    # RunPod's own PyTorch images have openssh-server + the account-pubkey injection
    # baked in; the bare pytorch/pytorch:* images from Docker Hub do not (SSH never starts).
    ap.add_argument("--image", default="runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04")
    ap.add_argument("--disk", type=int, default=150, help="container disk GB")
    ap.add_argument("--volume-gb", type=int, default=0, help="persistent volume GB (0 = none)")
    ap.add_argument("--cloud", default="COMMUNITY", choices=["COMMUNITY", "SECURE"])
    ap.add_argument("--wait-min", type=int, default=15, help="minutes to wait for RUNNING status")
    args = ap.parse_args()

    key_path = os.path.expanduser("~/.run.pod")
    runpod.api_key = open(key_path).read().strip()

    print(f"deploying pod name={args.name!r} gpu={args.gpu!r} cloud={args.cloud}...", flush=True)
    pod = runpod.create_pod(
        name=args.name,
        image_name=args.image,
        gpu_type_id=args.gpu,
        cloud_type=args.cloud,
        gpu_count=1,
        volume_in_gb=args.volume_gb,
        container_disk_in_gb=args.disk,
        ports="22/tcp",
        start_ssh=True,
        docker_args="",
        support_public_ip=True,
    )
    pod_id = pod["id"]
    print(f"pod id: {pod_id}", flush=True)

    deadline = time.time() + args.wait_min * 60
    last_status = None
    while time.time() < deadline:
        time.sleep(8)
        try:
            info = runpod.get_pod(pod_id)
        except Exception as e:
            print(f"  get_pod error: {e}"); continue
        status = (info.get("desiredStatus") or info.get("lastStatusChange") or "?")
        if status != last_status:
            print(f"  status: {status}", flush=True); last_status = status
        runtime = info.get("runtime") or {}
        ports = runtime.get("ports") or []
        ssh_port = next((p for p in ports if p.get("privatePort") == 22 and p.get("isIpPublic")), None)
        if ssh_port:
            ip = ssh_port["ip"]; port = ssh_port["publicPort"]
            line = f"{pod_id} {ip} {port}\n"
            with open(args.env_file, "w") as f:
                f.write(line)
            print(f"\n=== READY ===")
            print(f"  pod_id: {pod_id}")
            print(f"  ssh: ssh root@{ip} -p {port}")
            print(f"  env file: {args.env_file}")
            return
    print("FAIL: pod never reached SSH-ready state", file=sys.stderr)
    try:
        runpod.terminate_pod(pod_id); print(f"terminated {pod_id}")
    except Exception as e:
        print(f"terminate failed: {e}")
    sys.exit(1)


if __name__ == "__main__":
    main()
