#!/usr/bin/env python3
"""Modal runner for CBP full-trace benchmarking.

This script is separate from the existing GPU-focused `run_modal.py`.
It targets the CBP CPU simulator in this repository.

One-time setup example:
    pip install modal
    modal setup
    modal volume create cbp-traces
    modal volume put cbp-traces ./cbp-ng_training_traces/

Typical usage:
    python cbp_modal.py --help
    modal run cbp_modal.py --predictor-type 'Tage<>' --trace-subdir cbp-ng_training_traces \
        --output-dir modal_out/tageparam_full

Notes:
- Traces are expected in a Modal volume, not bundled into the image.
- Each trace is run as an independent Modal task.
- Results are materialized locally as `.out`, then aggregated with
  `predictor_metrics.py` and `vfs.py` from this repo.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import modal

APP_NAME = "cbp-fulltrace"
TRACE_VOLUME_NAME = "cbp-traces"
BUILD_VOLUME_NAME = "cbp-build-cache"
REMOTE_PROJECT = "/root/cbp7"
REMOTE_TRACE_ROOT = "/traces"
REMOTE_BUILD_ROOT = "/build-cache"
DEFAULT_JOBS = 32
REPO_ROOT = Path(__file__).resolve().parent.parent

app = modal.App(APP_NAME)
trace_volume = modal.Volume.from_name(TRACE_VOLUME_NAME, create_if_missing=False)
build_volume = modal.Volume.from_name(BUILD_VOLUME_NAME, create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("build-essential", "g++", "make", "zlib1g-dev", "bash", "findutils")
    .add_local_file(REPO_ROOT / "cbp.cpp", remote_path=f"{REMOTE_PROJECT}/cbp.cpp")
    .add_local_file(REPO_ROOT / "cbp.hpp", remote_path=f"{REMOTE_PROJECT}/cbp.hpp")
    .add_local_file(REPO_ROOT / "branch_predictor.hpp", remote_path=f"{REMOTE_PROJECT}/branch_predictor.hpp")
    .add_local_file(REPO_ROOT / "harcom.hpp", remote_path=f"{REMOTE_PROJECT}/harcom.hpp")
    .add_local_file(REPO_ROOT / "Makefile", remote_path=f"{REMOTE_PROJECT}/Makefile")
    .add_local_file(REPO_ROOT / "params.yaml", remote_path=f"{REMOTE_PROJECT}/params.yaml")
    .add_local_file(REPO_ROOT / "trace_files" / "trace_reader.hpp", remote_path=f"{REMOTE_PROJECT}/trace_files/trace_reader.hpp")
    .add_local_dir(REPO_ROOT / "scripts", remote_path=f"{REMOTE_PROJECT}/scripts")
    .add_local_dir(REPO_ROOT / "predictors", remote_path=f"{REMOTE_PROJECT}/predictors")
)


def _run(cmd: list[str], cwd: str | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=cwd, text=True, capture_output=True, check=True)


def _source_hash() -> str:
    root = REPO_ROOT
    inputs = [
        "Makefile",
        "params.yaml",
        "cbp.cpp",
        "cbp.hpp",
        "branch_predictor.hpp",
        "harcom.hpp",
        "scripts/gen_predictor_config.py",
        "trace_files/trace_reader.hpp",
    ]
    inputs.extend(sorted(str(p.relative_to(root)) for p in (root / "predictors").glob("*.hpp")))
    inputs.extend(sorted(str(p.relative_to(root)) for p in (root / "predictors" / "custom").glob("*.hpp")))

    digest = hashlib.sha256()
    for relpath in inputs:
        path = root / relpath
        if not path.exists():
            continue
        digest.update(relpath.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()[:16]


@app.function(
    image=image,
    cpu=1,
    timeout=60 * 60,
    volumes={REMOTE_TRACE_ROOT: trace_volume},
)
def list_traces(trace_subdir: str) -> list[str]:
    trace_subdir = trace_subdir.strip()
    trace_dir = Path(REMOTE_TRACE_ROOT) if trace_subdir in ("", ".") else Path(REMOTE_TRACE_ROOT) / trace_subdir
    if not trace_dir.exists():
        fallback_dir = Path(REMOTE_TRACE_ROOT)
        if fallback_dir.exists():
            trace_dir = fallback_dir
        else:
            raise FileNotFoundError(f"Trace directory not found in volume: {trace_dir}")
    return sorted(str(p.relative_to(Path(REMOTE_TRACE_ROOT))) for p in trace_dir.rglob("*_trace.gz"))


@app.function(
    image=image,
    cpu=2,
    timeout=60 * 60,
    volumes={REMOTE_BUILD_ROOT: build_volume},
)
def build_cbp(
    build_key: str,
    predictor_type: str = "tage<>",
    extra_common_flags: str = "",
    extra_cbp_flags: str = "",
) -> dict:
    project = Path(REMOTE_PROJECT)
    build_dir = Path(REMOTE_BUILD_ROOT) / build_key
    binary_path = build_dir / "cbp"

    build_volume.reload()
    if binary_path.exists():
        return {"build_key": build_key, "binary": str(binary_path), "cached": True}

    build_dir.mkdir(parents=True, exist_ok=True)
    make_cmd = [
        "make",
        "cbp",
        f"PREDICTOR_TYPE={predictor_type}",
    ]
    if extra_common_flags:
        make_cmd.append(f"EXTRA_COMMON_FLAGS={extra_common_flags}")
    if extra_cbp_flags:
        make_cmd.append(f"EXTRA_CBP_FLAGS={extra_cbp_flags}")
    _run(make_cmd, cwd=str(project))
    binary_path.write_bytes((project / "cbp").read_bytes())
    binary_path.chmod(0o755)
    (build_dir / "build_meta.json").write_text(
        json.dumps(
            {
                "build_key": build_key,
                "predictor_type": predictor_type,
                "extra_common_flags": extra_common_flags,
                "extra_cbp_flags": extra_cbp_flags,
            },
            indent=2,
        )
    )
    build_volume.commit()
    return {"build_key": build_key, "binary": str(binary_path), "cached": False}


@app.function(
    image=image,
    cpu=1,
    timeout=60 * 60,
    volumes={REMOTE_TRACE_ROOT: trace_volume, REMOTE_BUILD_ROOT: build_volume},
)
def run_trace(
    trace_relpath: str,
    build_key: str,
    warmup: int = 1_000_000,
    measure: int = 40_000_000,
) -> dict:
    build_volume.reload()
    trace_path = Path(REMOTE_TRACE_ROOT) / trace_relpath
    binary_path = Path(REMOTE_BUILD_ROOT) / build_key / "cbp"
    if not trace_path.exists():
        raise FileNotFoundError(f"Missing trace in volume: {trace_path}")
    if not binary_path.exists():
        raise FileNotFoundError(f"Missing cached binary: {binary_path}")

    trace_name = trace_path.stem.replace("_trace", "")

    proc = _run(
        [
            str(binary_path),
            str(trace_path),
            trace_name,
            str(warmup),
            str(measure),
        ],
    )

    line = proc.stdout.strip().splitlines()[-1]
    return {
        "trace": trace_relpath,
        "name": trace_name,
        "out": line,
    }


@app.local_entrypoint()
def main(
    predictor_type: str = "tage<>",
    trace_subdir: str = ".",
    output_dir: str = "modal_out/cbp_full",
    warmup: int = 1_000_000,
    measure: int = 40_000_000,
    jobs: int = DEFAULT_JOBS,
    extra_common_flags: str = "",
    extra_cbp_flags: str = "",
    limit: int = 0,
    trace_filter: str = "",
    json_out: str = "",
):
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    traces = list_traces.remote(trace_subdir)
    if trace_filter:
        traces = [t for t in traces if trace_filter in t]
    if limit > 0:
        traces = traces[:limit]
    if not traces:
        raise SystemExit("No traces selected")

    print(f"Selected {len(traces)} traces from volume subdir '{trace_subdir}'")
    print(f"Predictor: {predictor_type}")
    print(f"Parallel jobs requested: {jobs}")

    source_hash = _source_hash()
    build_key = hashlib.sha256(
        "|".join([source_hash, predictor_type, extra_common_flags, extra_cbp_flags]).encode("utf-8")
    ).hexdigest()[:16]
    build_info = build_cbp.remote(
        build_key,
        predictor_type=predictor_type,
        extra_common_flags=extra_common_flags,
        extra_cbp_flags=extra_cbp_flags,
    )
    cache_state = "hit" if build_info["cached"] else "miss"
    print(f"Build cache: {cache_state} ({build_key})")

    inputs = [
        (trace_relpath, build_key, warmup, measure)
        for trace_relpath in traces
    ]

    batch_size = max(1, jobs)
    results: list[dict] = []
    for batch_start in range(0, len(inputs), batch_size):
        batch = inputs[batch_start:batch_start + batch_size]
        for result in run_trace.starmap(batch, order_outputs=True):
            results.append(result)
            out_path = out_dir / f"{result['name']}.out"
            out_path.write_text(result["out"] + "\n")
            print(f"[{len(results)}/{len(traces)}] {result['name']}")

    metrics = subprocess.check_output(
        ["python3", "predictor_metrics.py", str(out_dir)],
        cwd=REPO_ROOT,
        text=True,
    ).strip()
    vfs = subprocess.check_output(
        ["python3", "vfs.py", metrics],
        cwd=REPO_ROOT,
        text=True,
    ).strip()

    metrics_path = out_dir.with_suffix(".metrics")
    vfs_path = out_dir.with_suffix(".vfs")
    metrics_path.write_text(metrics + "\n")
    vfs_path.write_text(vfs + "\n")

    summary = {
        "predictor_type": predictor_type,
        "trace_subdir": trace_subdir,
        "trace_count": len(traces),
        "metrics": metrics,
        "vfs": vfs,
    }
    print()
    print("Summary")
    print(json.dumps(summary, indent=2))

    if json_out:
        json_path = Path(json_out)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps({"summary": summary, "results": results}, indent=2))
        print(f"Saved JSON to {json_path}")
