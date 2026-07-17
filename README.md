# Reusable CUDA training image

An amd64 training base built on CUDA 12.8.1/cuDNN development libraries, CPython
3.12, and the official PyTorch cu128 wheels. It contains tools and caches, but no
model weights, datasets, credentials, or project source.

Pinned core versions: PyTorch 2.9.1, torchvision 0.24.1, torchaudio 2.9.1.

## Build locally

Build the image from any host with Docker or a compatible container builder that
can produce Linux amd64 images. This includes Linux, macOS, Windows, CI runners,
and remote builders.

```bash
docker build --platform linux/amd64 -t image-trainer:cuda12.8.1-torch2.9.1 .
```

Run the import-only verifier when a CUDA GPU is not available:

```bash
docker run --rm image-trainer:cuda12.8.1-torch2.9.1 verify-gpu --allow-no-cuda
```

## GPU runtime test

CUDA runtime testing requires a host with an NVIDIA GPU, a compatible NVIDIA
driver, and NVIDIA Container Runtime support. On Windows, use Docker Desktop with
WSL integration and the Windows NVIDIA driver; do not install a Linux display
driver inside WSL.

```bash
nvidia-smi
docker run --rm --gpus all image-trainer:cuda12.8.1-torch2.9.1 verify-gpu
docker run --rm --gpus all -it \
  -v "$PWD:/workspace" \
  image-trainer:cuda12.8.1-torch2.9.1
```

The verifier prints the installed/runtime versions, GPU name, compute capability,
VRAM, and runs a synchronized 512x512 CUDA matrix multiplication. A failure is
intentional when no GPU is available; CI uses `--allow-no-cuda` solely to validate
imports on its CPU runner.

## Publish to Docker Hub

Use an immutable release tag rather than `latest`:

```bash
docker login
docker tag image-trainer:cuda12.8.1-torch2.9.1 \
  <dockerhub-user>/image-trainer:cuda12.8.1-torch2.9.1
docker push <dockerhub-user>/image-trainer:cuda12.8.1-torch2.9.1
```

## Cloud GPU usage

Launch the image on a GPU host with NVIDIA Container Runtime support. Mount
persistent project data and caches at `/workspace`; the image keeps Hugging Face,
PyTorch, uv, and other runtime caches under that path. The default command is
`bash`, so SSH or web terminals should open an interactive shell by default.

Run `verify-gpu` before starting a training job. The `hf` command is available for
model downloads; authenticate at runtime with `hf auth login` when a gated
repository requires it. Keep Hugging Face, registry, and cloud tokens in your
provider's secret manager or runtime environment variables, never in the image or
Docker build arguments.

## Blackwell compatibility and remaining uncertainty

CUDA 12.8 is the first toolkit generation with native Blackwell support, and these
official cu128 wheels are intended for Blackwell-era GPUs. Compatibility still
depends on the host NVIDIA driver, WSL/Docker GPU passthrough, and every optional
compiled extension used by a mounted training project. Third-party extensions may
lack kernels for the GPU's compute capability or bundled PTX even when core PyTorch
passes. Test each target host with `verify-gpu`, followed by a small workload using
the project's optional extensions. The image cannot validate physical GPUs during a
CPU-only build.

## Maintenance and security

The CI workflow uses ordinary GitHub-hosted runners, CPU-only PyTorch wheels, and
BuildKit's GitHub Actions cache; Docker Build Cloud is not required. The release
build keeps the Dockerfile's default official cu128 index. Rebuild regularly to
pick up base-image and Ubuntu security updates. `.dockerignore` excludes common
credentials, model formats, data, caches, and outputs. Mounted `/workspace` content
remains outside the image, although containers run as root by default for broad
cloud-provider compatibility. The CUDA development base and cu128 libraries make
this image intentionally large; the build uses a disposable uv cache mount so
downloaded wheels are not duplicated in the final layers.

Treat checkpoints as untrusted input: prefer official safetensors artifacts and
never load unknown pickle-based weights.

The release candidate was scanned with:

```bash
docker scout cves image-trainer:cuda12.8.1-torch2.9.1 \
  --only-severity critical,high --only-fixed --format packages
```

The image deliberately omits Git LFS: Hugging Face model artifacts are downloaded
with `hf download`, while project source should be mounted or cloned normally. This
also avoids shipping Git LFS's embedded Go dependencies. Nsight Compute is removed
from the runnable filesystem, although layer-aware scanners may still report its
base-layer bytes. The no-LFS release candidate scan reported 3 critical and 23 high
findings: 3 critical and 20 high in those inherited Nsight bytes, 2 high in the
current pinned `uv` binary, and 1 high in PyTorch 2.9.1. Reassess all findings before
each public rebuild; do not interpret a passing runtime smoke test as a clean
vulnerability scan.
