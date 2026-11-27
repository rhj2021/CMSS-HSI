"""Install the pinned RoMa and RCDIFO packages for the current CUDA environment."""

from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

PACKAGES = [
    ("causal-conv1d", "causal_conv1d", "Dao-AILab/causal-conv1d", "v1.5.4", "1.5.4"),
    ("mamba-ssm", "mamba_ssm", "state-spaces/mamba", "v2.3.1", "2.3.1"),
]
SOURCE_COMMITS = {
    "causal-conv1d": "547dee6bab0dbe5c5954d7b8f0c813a0d5ff97c8",
    "mamba-ssm": "c5afbdf3bda1a09d68f65181ae3a43ec71079820",
}
KNOWN = {
    "causal_conv1d-1.5.4+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl": "c62338f9b3a914d79326e0f5c127361d6d7f8add87e896d01beb8291c95aa745",
    "mamba_ssm-2.3.1+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl": "f9e375b47cee1ef27463c9249073f9065f0f8f0f98b32fabfbb8df4bf75e2b66",
}


def environment():
    import torch

    if str(torch.__version__).split("+", 1)[0] != "2.8.0":
        raise RuntimeError("Install PyTorch 2.8.0 before running this installer.")
    if importlib.metadata.version("torchvision").split("+", 1)[0] != "0.23.0":
        raise RuntimeError(
            "PyTorch 2.8.0 requires the matching torchvision 0.23.0 build."
        )
    if sys.platform != "linux" or platform.machine() != "x86_64":
        raise RuntimeError("The pinned CUDA extensions target Linux x86_64.")
    if not torch.cuda.is_available() or not torch.version.cuda:
        raise RuntimeError(
            "Select a CUDA-enabled PyTorch installation with an available GPU."
        )
    for module in [
        "torchvision",
        "einops",
        "packaging",
        "transformers",
        "triton",
        "kornia",
        "loguru",
    ]:
        try:
            __import__(module)
        except (ImportError, OSError, RuntimeError) as exc:
            raise RuntimeError(
                f"Dependency {module} is unavailable; install requirements.txt with the matching PyTorch build: {exc}"
            ) from exc
    return dict(
        torch=str(torch.__version__),
        torch_mm="2.8",
        cuda=str(torch.version.cuda),
        cuda_major=str(torch.version.cuda).split(".")[0],
        abi=str(bool(torch._C._GLIBCXX_USE_CXX11_ABI)).upper(),
        python=f"cp{sys.version_info.major}{sys.version_info.minor}",
        platform="linux_x86_64",
    )


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def select_asset(pkg, env):
    dist, module, repo, tag, ver = pkg
    fname = f'{module}-{ver}+cu{env["cuda_major"]}torch{env["torch_mm"]}cxx11abi{env["abi"]}-{env["python"]}-{env["python"]}-{env["platform"]}.whl'
    if fname in KNOWN:
        return dict(
            name=fname,
            url=f"https://github.com/{repo}/releases/download/{tag}/{fname}",
            sha256=KNOWN[fname],
        )
    req = urllib.request.Request(
        f"https://api.github.com/repos/{repo}/releases/tags/{tag}",
        headers={
            "User-Agent": "CMSS-dependency-installer",
            "Accept": "application/vnd.github+json",
        },
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        release = json.load(r)
    hits = [a for a in release.get("assets", []) if a["name"] == fname]
    if len(hits) != 1:
        raise RuntimeError(
            f"No verified exact wheel in {repo}/{tag}: {fname}. Use --build-from-source with the CUDA toolkit to build Mamba and causal-conv1d."
        )
    asset = hits[0]
    digest = asset.get("digest", "") or ""
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", digest):
        raise RuntimeError(
            "Official release did not provide a SHA256 digest. Use --build-from-source with the CUDA toolkit."
        )
    return dict(
        name=fname, url=asset["browser_download_url"], sha256=digest.split(":", 1)[1]
    )


def fetch(asset, cache):
    path = cache / asset["name"]
    part = Path(str(path) + ".part")

    def valid(p):
        if not p.is_file() or sha(p) != asset["sha256"]:
            return False
        try:
            with zipfile.ZipFile(p) as z:
                return z.testzip() is None
        except zipfile.BadZipFile:
            return False

    if valid(path):
        print("[CACHE]", path.name, flush=True)
        return path
    if path.exists():
        path.rename(Path(str(path) + f".invalid-{time.time_ns()}"))
    curl = shutil.which("curl")
    if curl is None:
        raise RuntimeError(
            "curl is required for resumable downloads; install curl or put an independently verified wheel in --cache-dir."
        )
    cache.mkdir(parents=True, exist_ok=True)
    cmd = [
        curl,
        "--fail",
        "--location",
        "--continue-at",
        "-",
        "--retry",
        "8",
        "--retry-delay",
        "5",
        "--connect-timeout",
        "30",
        "--speed-limit",
        "1024",
        "--speed-time",
        "120",
        "--output",
        str(part),
        asset["url"],
    ]
    print("[DOWNLOAD]", asset["name"], flush=True)
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise RuntimeError(
            "Download interrupted; partial file retained. Repeat this command to resume."
        )
    if not valid(part):
        part.rename(Path(str(part) + f".invalid-{time.time_ns()}"))
        raise RuntimeError(
            "Wheel SHA256/ZIP check failed. Invalid bytes retained as .invalid; repeat to download again."
        )
    os.replace(part, path)
    return path


def pypi_asset(package, version):
    from packaging.tags import sys_tags
    from packaging.utils import parse_wheel_filename

    with urllib.request.urlopen(
        f"https://pypi.org/pypi/{package}/{version}/json", timeout=30
    ) as response:
        metadata = json.load(response)
    supported = set(sys_tags())
    for item in metadata["urls"]:
        if item["filename"].endswith(".whl"):
            _, _, _, tags = parse_wheel_filename(item["filename"])
            if supported & tags:
                return dict(
                    name=item["filename"],
                    url=item["url"],
                    sha256=item["digests"]["sha256"],
                )
    raise RuntimeError(
        f"No compatible {package} {version} wheel. Use Python 3.12 on Linux x86_64."
    )


def install_wheel(asset, cache):
    wheel = fetch(asset, cache)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-deps",
            "--force-reinstall",
            str(wheel),
        ],
        check=True,
    )


def build_package(package):
    distribution, _, repository, _, _ = package
    if shutil.which("nvcc") is None or shutil.which("git") is None:
        raise RuntimeError("Source builds require nvcc from the CUDA toolkit and git.")
    build_env = os.environ.copy()
    build_env["MAMBA_FORCE_BUILD"] = "TRUE"
    build_env["CAUSAL_CONV1D_FORCE_BUILD"] = "TRUE"
    source = f"git+https://github.com/{repository}.git@{SOURCE_COMMITS[distribution]}"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-deps",
            "--no-build-isolation",
            "--force-reinstall",
            source,
        ],
        env=build_env,
        check=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check compatibility and list packages without installing",
    )
    parser.add_argument(
        "--build-from-source",
        action="store_true",
        help="Build Mamba and causal-conv1d from pinned commits using the CUDA toolkit",
    )
    parser.add_argument(
        "--cache-dir", type=Path, default=Path.home() / ".cache/cmss_wheels"
    )
    args = parser.parse_args()
    env = environment()
    print(json.dumps(env, indent=2))
    selected = []
    for package in PACKAGES:
        asset = None if args.build_from_source else select_asset(package, env)
        selected.append((package, asset))
        print(
            package[0], package[4], "source build" if asset is None else asset["name"]
        )
    extras = [pypi_asset("romatch", "0.1.2"), pypi_asset("fused-local-corr", "0.3.28")]
    for asset in extras:
        print(asset["name"])
    if args.check:
        return
    for package, asset in selected:
        if asset is None:
            build_package(package)
        else:
            install_wheel(asset, args.cache_dir)
    for asset in extras:
        install_wheel(asset, args.cache_dir)
    check = "import torch, torchvision, causal_conv1d_cuda, selective_scan_cuda, local_corr, romatch; from mamba_ssm.modules.mamba_simple import Mamba; print('CUDA packages import successfully.')"
    subprocess.run([sys.executable, "-c", check], check=True)
    import torch

    if str(torch.__version__) != env["torch"]:
        raise RuntimeError("Unexpected PyTorch version change.")
    print(
        "Installed RoMa 0.1.2, fused-local-corr 0.3.28, Mamba 2.3.1 and causal-conv1d 1.5.4."
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("ERROR:", exc, file=sys.stderr)
        sys.exit(1)
