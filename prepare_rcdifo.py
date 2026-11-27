"""Prepare the pinned upstream RCDIFO model for local use."""

from __future__ import annotations
import argparse
import ast
import hashlib
import json
from pathlib import Path
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parent


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def prepare(archive, root=ROOT):
    root = Path(root)
    settings = json.loads((root / "configs/rcdifo_sources.json").read_text())
    archive = Path(archive)
    target = root / "experiments/rcdifo/_vendor"
    converted, files = {}, {}
    with zipfile.ZipFile(archive) as source:
        for relative, expected in settings["files"].items():
            matches = [
                name
                for name in source.namelist()
                if name == relative or name.endswith("/" + relative)
            ]
            if len(matches) != 1:
                raise ValueError(
                    f"Expected one copy of {relative}, found {len(matches)}."
                )
            raw = source.read(matches[0])
            if sha256(raw) != expected:
                raise ValueError("RCDIFO source checksum mismatch: " + relative)
            text = raw.decode("utf-8-sig").replace("\r\n", "\n")
            text = text.replace("from model.", "from experiments.rcdifo._vendor.")
            if relative.endswith("/NET.py"):
                for line in [
                    "from torchnet.dataset import TensorDataset\n",
                    "from data_loader.data_load import HSICD_Dataset\n",
                    "from thop import profile, clever_format\n",
                    "import matplotlib.pyplot as plt\n",
                ]:
                    if text.count(line) != 1:
                        raise ValueError("Unexpected RCDIFO import: " + line.strip())
                    text = text.replace(line, "")
                # Positions must follow the same feedback filter as the patches.
                original = "x1, x2, position = self.ss(x1, x2, pos)"
                if text.count(original) != 1:
                    raise ValueError("Unexpected RCDIFO position-filter call.")
                text = text.replace(
                    original, "x1, x2, position = self.ss(x1, x2, position)"
                )
            if relative.endswith("/SS2D.py"):
                for original, replacement in [
                    (
                        "from mamba_ssm.ops.selective_scan_interface import selective_scan_fn",
                        "from experiments.rcdifo.mamba_backend import selective_scan_fn",
                    ),
                    (
                        "from mamba_ssm.modules.mamba_simple import Mamba",
                        "from experiments.rcdifo.mamba_backend import Mamba",
                    ),
                ]:
                    if text.count(original) != 1:
                        raise ValueError("Unexpected RCDIFO Mamba import.")
                    text = text.replace(original, replacement)
            tree = ast.parse(text)
            lines = text.splitlines(keepends=True)
            examples = [
                (node.lineno - 1, node.end_lineno)
                for node in tree.body
                if isinstance(node, ast.If) and "__name__" in ast.unparse(node.test)
            ]
            for start, stop in reversed(examples):
                del lines[start:stop]
            text = "".join(lines)
            compile(text, relative, "exec")
            destination = relative.removeprefix("model/")
            converted[destination] = text
            files[relative] = {
                "source_sha256": expected,
                "adapted_sha256": sha256(text.encode()),
            }
    for relative, text in converted.items():
        path = target / relative
        if path.exists() and path.read_text() != text:
            raise FileExistsError("Prepared file has local changes: " + str(path))
    for relative, text in converted.items():
        path = target / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    for path in [target, target / "change_model", target / "gegistration"]:
        (path / "__init__.py").write_text("")
    manifest = {
        "repository": settings["repository"],
        "commit": settings["commit"],
        "archive_sha256": sha256(archive.read_bytes()),
        "files": files,
    }
    (target / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("Prepared RCDIFO model in", target)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", default="downloads/RCDIFO.zip", type=Path)
    parser.add_argument(
        "--download", action="store_true", help="Download the pinned upstream archive"
    )
    args = parser.parse_args()
    settings = json.loads((ROOT / "configs/rcdifo_sources.json").read_text())
    if args.download and not args.zip.exists():
        with urllib.request.urlopen(settings["archive_url"], timeout=60) as response:
            data = response.read()
        if sha256(data) != settings["archive_sha256"]:
            raise ValueError("Downloaded RCDIFO archive checksum mismatch.")
        args.zip.parent.mkdir(parents=True, exist_ok=True)
        args.zip.write_bytes(data)
    if not args.zip.is_file():
        parser.error("Provide the RCDIFO archive with --zip, or use --download.")
    prepare(args.zip)


if __name__ == "__main__":
    main()
