# Modal Runner

This folder contains the Modal-based cloud runner for CBP full-trace benchmarking.

## Files

- `cbp_modal.py`: runs CBP on Modal using a trace volume and a build-cache volume.

## What It Does

`cbp_modal.py` uploads only the minimum source set needed to build `cbp`, compiles the predictor once on Modal, stores the binary in a build-cache volume, and then runs traces in parallel as independent Modal tasks. After the run, it writes local `.out` files and computes `.metrics` and `.vfs`.

## Required Volumes

Create these once:

```bash
modal volume create cbp-traces
modal volume create cbp-build-cache
```

Upload traces:

```bash
modal volume put cbp-traces ./cbp-ng_training_traces/
```

If your traces end up at the volume root instead of under `cbp-ng_training_traces/`, that is fine. The script defaults to scanning the volume root.

## Conda Setup

Create the recommended conda environment and install Modal:

```bash
conda create -y -n cbp-modal python=3.11
conda activate cbp-modal
python -m pip install --upgrade pip
python -m pip install modal
```

Verify the install:

```bash
modal --version
python -c "import modal; print(modal.__version__)"
```

## What Is Modal?

Modal is a serverless cloud platform for running compute jobs without managing machines directly. In this project, we use it to run many CBP traces in parallel and collect the results locally.

To get started, create a Modal account here:

- https://modal.com/signup

You can also run `modal setup` after installing the CLI and follow the login flow in your terminal.

As of March 2026, Modal's Starter plan includes **$30/month free compute credits**. Pricing and plan details can change, so check the official pricing page for the latest information:

- https://modal.com/pricing

## Environment
Recommended local environment:

```bash
conda activate cbp-modal
modal setup
```

## Smoke Test

Run one trace first:

```bash
modal run modal_runner/cbp_modal.py --predictor-type 'Tage<>' --limit 1 --output-dir modal_out/smoke
```

## Full Run

Run all available traces:

```bash
modal run modal_runner/cbp_modal.py --predictor-type 'Tage<>' --output-dir modal_out/tage_full
```

## Useful Options

- `--predictor-type 'Tage<>'`
- `--limit 10`
- `--trace-filter gcc`
- `--output-dir modal_out/whatever`
- `--extra-common-flags '...'`
- `--extra-cbp-flags '...'`

## Build Cache

The runner computes a cache key from:

- relevant CBP source files
- `predictor_type`
- extra compile flags

If those inputs do not change, later runs reuse the cached binary instead of recompiling.

## Changing The Build Command

Cloud compilation happens inside `build_cbp()` in `modal_runner/cbp_modal.py`.

The current build command is assembled as:

```python
make_cmd = [
    "make",
    "cbp",
    f"PREDICTOR_TYPE={predictor_type}",
]
```

If you want to change how CBP is compiled on Modal, this is the place to modify. Common examples:

- change the build target
- add or remove compile flags
- pass different `EXTRA_COMMON_FLAGS`
- pass different `EXTRA_CBP_FLAGS`

For example, you can already inject extra flags from the CLI:

```bash
modal run modal_runner/cbp_modal.py \
  --predictor-type 'Tage<>' \
  --extra-cbp-flags '-DTAGE_PRINT_PARAMS' \
  --limit 1 \
  --output-dir modal_out/debug
```

If you make structural changes to `make_cmd`, the build cache key may also need to be updated so Modal does not incorrectly reuse an older binary.

## Current Notes

- `jobs` is currently informational only. Actual parallelism is mainly determined by Modal scheduling and account limits, usually 100 containers at sametime.
- The script uploads only the source files needed to build `cbp`; old benchmark output files are not mounted into Modal.
