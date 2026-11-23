#!/usr/bin/env python3
"""Fetch the Docker images a SWE-bench run needs, before the run starts.

Every SWE-bench task needs its own image (``swebench/sweb.eval.x86_64.<task>``).
Pulling them inside a session wastes the session budget and hides failures, so
do it up front:

    ~/.exgentic/benchmarks/swebench/venv/bin/python \
        scripts/pull-swebench-images.py --num-tasks 20

Run it with the benchmark venv (it needs ``swebench`` and ``datasets``).

Image selection mirrors ``RunConfig.get_session_configs``: all task ids are
shuffled with the benchmark seed and the first ``--num-tasks`` are kept, so the
pulled images line up with the run.

``docker pull`` is used first. If it fails (registry stalls behind a proxy), the
layers are fetched straight from the registry API and assembled into a tar that
``docker load`` accepts; downloaded layers stay cached under
``~/.exgentic/swebench-images/<instance>/``.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import subprocess
import sys
import tarfile
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

SUBSET = "princeton-nlp/SWE-bench_Verified"
SEED = 42
NAMESPACE = "swebench"
IMAGE_PREFIX = "sweb.eval.x86_64"
CACHE_ROOT = Path(os.environ.get("EXGENTIC_SWEBBENCH_IMAGE_CACHE", str(Path.home() / ".exgentic" / "swebench-images")))
REGISTRY = "https://registry-1.docker.io"
AUTH_URL = "https://auth.docker.io/token"
ACCEPT = ", ".join(
    [
        "application/vnd.docker.distribution.manifest.v2+json",
        "application/vnd.docker.distribution.manifest.list.v2+json",
        "application/vnd.oci.image.manifest.v1+json",
        "application/vnd.oci.image.index.v1+json",
    ]
)


def image_name(instance_id: str) -> str:
    return f"{NAMESPACE}/{IMAGE_PREFIX}.{instance_id.replace('__', '_1776_')}:latest"


def docker_image_exists(name: str) -> bool:
    return subprocess.run(
        ["docker", "image", "inspect", name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    ).returncode == 0


def docker_pull(name: str) -> tuple[bool, str]:
    proc = subprocess.run(["docker", "pull", name], capture_output=True, text=True, timeout=1800)
    if proc.returncode == 0:
        return True, "docker pull"
    return False, (proc.stderr or proc.stdout).strip().splitlines()[-1:] and (proc.stderr or proc.stdout).strip().splitlines()[-1] or "docker pull failed"


def _fetch(url: str, headers: dict, dest: Path | None = None) -> bytes:
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=120) as response:
        if dest is None:
            return response.read()
        partial = dest.with_name(dest.name + ".part")
        with partial.open("wb") as fh:
            shutil.copyfileobj(response, fh)
    partial.replace(dest)
    return b""


def registry_pull(name: str) -> tuple[bool, str]:
    """Download image layers through the registry API and ``docker load`` them."""
    repository, _, tag = name.partition(":")
    tag = tag or "latest"
    cache_dir = CACHE_ROOT / repository.split("/")[-1]
    cache_dir.mkdir(parents=True, exist_ok=True)

    token = json.loads(
        _fetch(f"{AUTH_URL}?{urllib.parse.urlencode({'service': 'registry.docker.io', 'scope': f'repository:{repository}:pull'})}", {})
    )["token"]
    headers = {"Authorization": f"Bearer {token}", "Accept": ACCEPT}

    manifest = json.loads(_fetch(f"{REGISTRY}/v2/{repository}/manifests/{tag}", headers))
    if "manifests" in manifest:
        digest = next(m["digest"] for m in manifest["manifests"] if m.get("platform", {}).get("architecture") == "amd64")
        manifest = json.loads(_fetch(f"{REGISTRY}/v2/{repository}/manifests/{digest}", headers))

    config_digest = manifest["config"]["digest"]
    config_file = cache_dir / "config.json"
    if not config_file.is_file():
        _fetch(f"{REGISTRY}/v2/{repository}/blobs/{config_digest}", headers, config_file)

    layer_files = []
    for index, layer in enumerate(manifest["layers"]):
        layer_file = cache_dir / f"layer-{index:02d}.tar.gz"
        if layer_file.is_file() and layer_file.stat().st_size != layer.get("size"):
            layer_file.unlink()
        if not layer_file.is_file():
            _fetch(f"{REGISTRY}/v2/{repository}/blobs/{layer['digest']}", headers, layer_file)
        layer_files.append(layer_file)

    tar_path = cache_dir / "image.tar"
    with tarfile.open(tar_path, "w") as tar:
        tar.add(config_file, arcname=config_digest.replace(":", "_") + ".json")
        names = []
        for layer_file in layer_files:
            arcname = layer_file.name
            tar.add(layer_file, arcname=arcname)
            names.append(arcname)
        manifest_json = json.dumps(
            [{"Config": config_digest.replace(":", "_") + ".json", "RepoTags": [name], "Layers": names}]
        ).encode()
        info = tarfile.TarInfo("manifest.json")
        info.size = len(manifest_json)
        tar.addfile(info, __import__("io").BytesIO(manifest_json))

    proc = subprocess.run(["docker", "load", "-i", str(tar_path)], capture_output=True, text=True)
    tar_path.unlink(missing_ok=True)
    if proc.returncode != 0:
        return False, (proc.stderr or proc.stdout).strip()[:200]
    return True, "registry fallback"


def resolve_instances(num_tasks: int, seed: int) -> list[str]:
    from exgentic.benchmarks.swebench.swebench_eval import SWEBenchEvaluator

    evaluator = SWEBenchEvaluator(subset=SUBSET)
    try:
        task_ids = [str(t) for t in evaluator.list_tasks()]
    finally:
        try:
            evaluator.close()
        except Exception:
            pass
    random.Random(seed).shuffle(task_ids)
    return task_ids[:num_tasks]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--num-tasks", type=int, default=20, help="how many tasks the run will use (default: 20)")
    parser.add_argument("--seed", type=int, default=SEED, help="benchmark seed used for task selection (default: 42)")
    parser.add_argument("--parallel", type=int, default=3, help="concurrent pulls (default: 3)")
    parser.add_argument("--instances", nargs="*", default=None, help="explicit instance ids instead of the sampled ones")
    parser.add_argument("--dry-run", action="store_true", help="only print the selected instances")
    parser.add_argument("--no-registry-fallback", action="store_true", help="fail instead of using the registry API")
    args = parser.parse_args()

    instances = args.instances or resolve_instances(args.num_tasks, args.seed)
    print(f"{len(instances)} instance(s):")
    for instance_id in instances:
        print(f"  {instance_id}")
    if args.dry_run:
        return 0

    missing = [i for i in instances if not docker_image_exists(image_name(i))]
    print(f"\n{len(instances) - len(missing)} already present, {len(missing)} to pull\n")
    if not missing:
        return 0

    failures: list[tuple[str, str]] = []

    def worker(instance_id: str) -> tuple[str, bool, str]:
        name = image_name(instance_id)
        ok, detail = docker_pull(name)
        if not ok and not args.no_registry_fallback:
            print(f"[{instance_id}] docker pull failed ({detail}); trying registry API")
            ok, detail = registry_pull(name)
        return instance_id, ok, detail

    with ThreadPoolExecutor(max_workers=max(1, args.parallel)) as pool:
        futures = {pool.submit(worker, i): i for i in missing}
        for done, future in enumerate(as_completed(futures), start=1):
            instance_id, ok, detail = future.result()
            status = "ok" if ok else "FAILED"
            print(f"[{done}/{len(missing)}] {status:<6} {instance_id} ({detail})")
            if not ok:
                failures.append((instance_id, detail))

    if failures:
        print("\nfailed:")
        for instance_id, detail in failures:
            print(f"  {instance_id}: {detail}")
        return 1
    print("\nall images ready")
    return 0


if __name__ == "__main__":
    sys.exit(main())