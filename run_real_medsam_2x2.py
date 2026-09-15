#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WYSIWYR real MedSAM 2x2 factorial pipeline for AutoDL.

One-command goal
================
    cd <DATA_ROOT>
    python run_real_medsam_2x2.py --auto

The script can, in one run:
1) bootstrap the official MedSAM code and official MedSAM ViT-B checkpoint;
2) use local TrainDataset.zip + the user's local RAR TestDataset first; network only as fallback;
3) create a fixed 870/580 train/validation manifest from the 1450-image train pool;
4) fine-tune TWO MedSAM models from the SAME initialization:
      - baseline loss = Dice + BCE
      - ABLoss        = Dice + BCE + ambiguity-aware FP/FN boundary penalties
5) run leakage-free automatic test inference (NO GT box at test):
      full-image box -> coarse prediction -> prediction-derived tight box -> refined mask
6) apply USR to both trained models to create the full 2x2 factorial:
      MedSAM / MedSAM+ABLoss / MedSAM+USR / MedSAM+ABLoss+USR
7) save masks, probability maps, uncertainty maps, prompt diagnostics and latency;
8) call run_reviewer2_experiments.py automatically for each test dataset.

Scientific/reproducibility notes
================================
- Training prompts ARE GT-derived boxes because training annotations are supervised data.
  Test prompts NEVER use GT. This distinction is written to protocol.json.
- The original manuscript disclosed "fixed random seed" but not the numeric value, and did
  not disclose the main epoch count / learning rate / batch size. Therefore the defaults
  below are REVISION REPRODUCIBILITY SETTINGS, not claimed recovery of the lost originals.
- Public PraNet protocol has 900 Kvasir + 550 CVC-ClinicDB training images, and test counts
  Kvasir=100, CVC-ClinicDB=62, CVC-ColonDB=380, ETIS=196, CVC-300=60. If the manuscript
  currently states different counts, do not hide the mismatch; reconcile it in revision.
- ABLoss and baseline are trained with identical optimizer/schedule/data/prompt protocol.
- The pipeline is restartable: rerunning --auto reuses downloads and completed best models,
  and resumes interrupted training from last.pt.

This script does not fabricate any experimental result.
"""


from __future__ import annotations

import os
DATA_ROOT = os.environ.get("WYSIWYR_DATA_ROOT", ".")

import argparse
import contextlib
import csv
import hashlib
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
import urllib.request
import zipfile
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# AutoDL images can carry malformed OpenMP variables.
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    _v = os.environ.get(_k)
    if _v is not None:
        try:
            if int(_v) <= 0:
                raise ValueError
        except Exception:
            os.environ[_k] = "1"

import numpy as np

try:
    import cv2
except Exception as e:
    raise SystemExit("opencv-python is required. The bundled core normally already needs it.") from e

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
except Exception as e:
    raise SystemExit("PyTorch is required in the AutoDL image.") from e

try:
    import wysiwyr_autodl_allinone as core
except Exception as e:
    raise SystemExit(
        "Put run_real_medsam_2x2.py and wysiwyr_autodl_allinone.py in the SAME directory.\n"
        f"Import error: {e}"
    ) from e


# ---------------------------------------------------------------------------
# Constants / public sources
# ---------------------------------------------------------------------------
MEDSAM_GIT = "https://github.com/bowang-lab/MedSAM.git"
MEDSAM_CKPT_URL = "https://zenodo.org/records/10689643/files/medsam_vit_b.pth?download=1"
# Official MedSAM Google Drive file (linked from the official MedSAM project).
MEDSAM_CKPT_GDRIVE_ID = "1UAmWL88roYR7wKlnApw5Bcuzf2iQgk6_"
# Public fallback mirror of the same 375 MB MedSAM ViT-B checkpoint.
MEDSAM_CKPT_HF_URL = "https://huggingface.co/spaces/KalbeDigitalLab/medsam-segment-anything/resolve/main/checkpoint/medsam_vit_b.pth?download=true"
MEDSAM_CKPT_MD5 = "3bb6db55bd0c9ca30b61248bca72f8d6"

# Official PraNet README Google Drive IDs (common polyp benchmark protocol)
PRANET_TRAIN_GDRIVE_ID = "1YiGHLw4iTvKdvbT6MgwO9zcCv8zJ_Bnb"
PRANET_TEST_GDRIVE_ID = "1Y2z7FD5p5y31vkZwQQomXFRB0HutHyao"

EXPECTED_PUBLIC_COUNTS = {
    "CVC-300": 60,
    "CVC-ClinicDB": 62,
    "Kvasir": 100,
    "CVC-ColonDB": 380,
    "ETIS-LaribPolypDB": 196,
}
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def log(msg: str = "") -> None:
    print(msg, flush=True)


def run_cmd(cmd: Sequence[str], cwd: Optional[Path] = None, check: bool = True) -> subprocess.CompletedProcess:
    log("[cmd] " + " ".join(str(x) for x in cmd))
    return subprocess.run(list(map(str, cmd)), cwd=str(cwd) if cwd else None, check=check)


def md5sum(path: Path, chunk: int = 4 * 1024 * 1024) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def atomic_torch_save(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def append_csv(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            w.writeheader()
        w.writerow(row)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Determinism where practical; warn_only avoids unsupported-op crashes.
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


# ---------------------------------------------------------------------------
# Setup: MedSAM repo / checkpoint / public datasets
# ---------------------------------------------------------------------------
def ensure_medsam_repo(repo_dir: Path) -> Path:
    try:
        import segment_anything  # noqa: F401
        return Path(segment_anything.__file__).resolve().parents[1]
    except Exception:
        pass

    if not repo_dir.exists():
        repo_dir.parent.mkdir(parents=True, exist_ok=True)
        run_cmd(["git", "clone", "--depth", "1", MEDSAM_GIT, str(repo_dir)])
    if str(repo_dir) not in sys.path:
        sys.path.insert(0, str(repo_dir))
    try:
        import segment_anything  # noqa: F401
    except Exception as e:
        raise RuntimeError(f"MedSAM repo exists but segment_anything import failed: {e}") from e
    return repo_dir


def _download_wget(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if shutil.which("wget"):
        # Do not hang for minutes on a blocked host. Continue with the next mirror.
        run_cmd([
            "wget", "-c", "--show-progress", "--timeout=20", "--tries=2",
            "-O", str(dest), url
        ])
        return
    # urllib fallback (no resume)
    log(f"Downloading {url} -> {dest}")
    urllib.request.urlretrieve(url, dest)


def _checkpoint_is_valid(path: Path) -> bool:
    if not path.exists() or path.stat().st_size <= 300_000_000:
        return False
    digest = md5sum(path)
    if digest.lower() != MEDSAM_CKPT_MD5.lower():
        log(f"[WARN] checkpoint MD5 differs: {digest}")
        return False
    log(f"[OK] MedSAM checkpoint verified: {path}")
    return True


def ensure_medsam_checkpoint(path: Path) -> Path:
    if _checkpoint_is_valid(path):
        return path

    # Auto-detect a manually uploaded checkpoint in the AutoDL project root.
    # This lets the user upload medsam_vit_b.pth through Jupyter without any mv/cp command.
    local_candidates = [Path.cwd() / "medsam_vit_b.pth", Path(DATA_ROOT + "/medsam_vit_b.pth")]
    for candidate in local_candidates:
        try:
            if candidate.resolve() == path.resolve():
                continue
        except Exception:
            pass
        if _checkpoint_is_valid(candidate):
            import shutil
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(candidate, path)
            log(f"[OK] Reused manually uploaded MedSAM checkpoint: {candidate}")
            return path

    if path.exists():
        log(f"[WARN] removing incomplete/invalid checkpoint: {path}")
        try:
            path.unlink()
        except FileNotFoundError:
            pass

    path.parent.mkdir(parents=True, exist_ok=True)
    errors = []

    # 1) Google Drive first: this is the checkpoint source linked by the official MedSAM repo.
    try:
        log("[1/3] Downloading MedSAM checkpoint from official Google Drive ...")
        gdown_file(MEDSAM_CKPT_GDRIVE_ID, path)
        if _checkpoint_is_valid(path):
            return path
        raise RuntimeError("Google Drive download completed but MD5 verification failed")
    except Exception as e:
        errors.append(f"Google Drive: {e}")
        log(f"[WARN] Google Drive failed: {e}")
        if path.exists():
            path.unlink()

    # 2) Hugging Face public mirror. The file size is 375 MB; MD5 is still verified against Zenodo.
    try:
        log("[2/3] Downloading MedSAM checkpoint from Hugging Face mirror ...")
        _download_wget(MEDSAM_CKPT_HF_URL, path)
        if _checkpoint_is_valid(path):
            return path
        raise RuntimeError("Hugging Face download completed but MD5 verification failed")
    except Exception as e:
        errors.append(f"Hugging Face: {e}")
        log(f"[WARN] Hugging Face failed: {e}")
        if path.exists():
            path.unlink()

    # 3) Zenodo last, because some AutoDL routes refuse Zenodo connections.
    try:
        log("[3/3] Downloading MedSAM checkpoint from Zenodo ...")
        _download_wget(MEDSAM_CKPT_URL, path)
        if _checkpoint_is_valid(path):
            return path
        raise RuntimeError("Zenodo download completed but MD5 verification failed")
    except Exception as e:
        errors.append(f"Zenodo: {e}")
        if path.exists():
            path.unlink()

    raise RuntimeError(
        "All automatic MedSAM checkpoint sources failed.\n"
        + "\n".join("  - " + x for x in errors)
        + f"\nYou can also upload medsam_vit_b.pth manually to: {path}"
    )


def ensure_gdown() -> None:
    try:
        import gdown  # noqa: F401
        return
    except Exception:
        run_cmd([sys.executable, "-m", "pip", "install", "-q", "gdown"])


def gdown_file(file_id: str, dest: Path) -> None:
    """Download a public Google Drive file across old/new gdown versions."""
    ensure_gdown()
    import gdown
    import inspect

    dest.parent.mkdir(parents=True, exist_ok=True)
    log(f"Downloading public Google Drive file -> {dest}")

    # AutoDL may have an older gdown where download() has no `fuzzy` argument.
    # Inspect the installed API and pass only kwargs it actually supports.
    params = inspect.signature(gdown.download).parameters
    kwargs = {"output": str(dest), "quiet": False}
    if "id" in params:
        kwargs["id"] = file_id
    else:
        kwargs["url"] = f"https://drive.google.com/uc?id={file_id}"
    if "fuzzy" in params:
        kwargs["fuzzy"] = True
    if "resume" in params:
        kwargs["resume"] = True
    if "use_cookies" in params:
        kwargs["use_cookies"] = True

    try:
        out = gdown.download(**kwargs)
    except TypeError as e:
        log(f"[WARN] gdown API compatibility retry: {e}")
        out = gdown.download(
            url=f"https://drive.google.com/uc?id={file_id}",
            output=str(dest),
            quiet=False,
        )

    if not out or not dest.exists() or dest.stat().st_size == 0:
        raise RuntimeError(f"gdown failed for file id {file_id}")


def _find_named_dir(root: Path, name: str) -> Optional[Path]:
    cands = [p for p in root.rglob("*") if p.is_dir() and p.name.lower() == name.lower()]
    if not cands:
        return None
    cands.sort(key=lambda p: (len(p.parts), str(p)))
    return cands[0]


def _pick_latest(patterns: Sequence[str]) -> Optional[Path]:
    hits: List[Path] = []
    for pat in patterns:
        hits.extend(Path(DATA_ROOT + "").glob(pat))
    hits = [p for p in hits if p.is_file() and p.stat().st_size > 0]
    if not hits:
        return None
    return max(hits, key=lambda p: p.stat().st_mtime)


def _ensure_rar_extractor() -> Optional[str]:
    """Return an available RAR extractor; on AutoDL try installing unar once."""
    for name in ("unar", "7z", "7za", "unrar"):
        tool = shutil.which(name)
        if tool:
            return tool
    if os.geteuid() == 0 and shutil.which("apt-get"):
        log("[INFO] No RAR extractor found. Trying apt-get install -y unar ...")
        run_cmd(["apt-get", "update", "-qq"], check=False)
        run_cmd(["apt-get", "install", "-y", "unar"], check=False)
        tool = shutil.which("unar")
        if tool:
            return tool
    return None


def _extract_rar(archive: Path, out_dir: Path) -> None:
    tool = _ensure_rar_extractor()
    if not tool:
        raise RuntimeError(
            "A local .rar bundle was found, but no RAR extractor is available. "
            "Install 'unar' (apt-get install -y unar) or extract the archive locally and upload the TestDataset folder."
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    name = Path(tool).name
    log(f"Extracting local RAR bundle with {name}: {archive}")
    if name == "unar":
        run_cmd([tool, "-f", "-o", str(out_dir), str(archive)])
    elif name.startswith("7z"):
        run_cmd([tool, "x", "-y", f"-o{out_dir}", str(archive)])
    else:
        run_cmd([tool, "x", "-o+", str(archive), str(out_dir)])


def ensure_public_data(data_root: Path, downloads: Path, test_rar: Optional[Path] = None) -> Tuple[Path, Path]:
    """Return TrainDataset/TestDataset, preferring LOCAL files before any network access.

    Local-first protocol for this revision:
      - training: <DATA_ROOT>/TrainDataset*.zip (manual upload) or downloads/TrainDataset.zip
      - testing:  the user's local tumor_30.rar bundle if it contains TestDataset/
      - network is used only as a last fallback.
    """
    data_root.mkdir(parents=True, exist_ok=True)
    downloads.mkdir(parents=True, exist_ok=True)
    train_dir = _find_named_dir(data_root, "TrainDataset")
    test_dir = _find_named_dir(data_root, "TestDataset")

    if train_dir is None:
        z = downloads / "TrainDataset.zip"
        if not z.exists() or z.stat().st_size == 0:
            local_z = _pick_latest(["TrainDataset*.zip", "trainDataset*.zip", "traindataset*.zip"])
            if local_z:
                log(f"[LOCAL] Using uploaded training archive: {local_z}")
                shutil.copy2(local_z, z)
            else:
                log("[WARN] No local TrainDataset.zip found; trying public Google Drive as fallback ...")
                gdown_file(PRANET_TRAIN_GDRIVE_ID, z)
        log("Extracting TrainDataset.zip ...")
        with zipfile.ZipFile(z) as f:
            f.extractall(data_root)
        train_dir = _find_named_dir(data_root, "TrainDataset")

    if test_dir is None:
        rar = test_rar if test_rar and test_rar.exists() else None
        if rar is None:
            cand = _pick_latest(["tumor_30*.rar", "Tumor_30*.rar", "tumor30*.rar"])
            rar = cand
        if rar is not None:
            log(f"[LOCAL] Using test datasets already present in RAR bundle: {rar}")
            _extract_rar(rar, data_root)
            test_dir = _find_named_dir(data_root, "TestDataset")

    if test_dir is None:
        z = downloads / "TestDataset.zip"
        if not z.exists() or z.stat().st_size == 0:
            local_z = _pick_latest(["TestDataset*.zip", "testDataset*.zip", "testdataset*.zip"])
            if local_z:
                log(f"[LOCAL] Using uploaded test archive: {local_z}")
                shutil.copy2(local_z, z)
            else:
                log("[WARN] Local RAR did not provide TestDataset; trying public Google Drive fallback ...")
                gdown_file(PRANET_TEST_GDRIVE_ID, z)
        log("Extracting TestDataset.zip ...")
        with zipfile.ZipFile(z) as f:
            f.extractall(data_root)
        test_dir = _find_named_dir(data_root, "TestDataset")

    if train_dir is None or test_dir is None:
        raise RuntimeError(f"Could not locate TrainDataset/TestDataset after local preparation under {data_root}")
    log(f"[OK] TrainDataset: {train_dir}")
    log(f"[OK] TestDataset:  {test_dir}")
    return train_dir, test_dir


# ---------------------------------------------------------------------------
# Dataset discovery / pairing / split manifest
# ---------------------------------------------------------------------------
def _image_files(folder: Path) -> List[Path]:
    if not folder.exists():
        return []
    return sorted([p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS])


def _folder_child(root: Path, names: Sequence[str]) -> Optional[Path]:
    lowered = {n.lower() for n in names}
    for p in root.iterdir() if root.exists() else []:
        if p.is_dir() and p.name.lower() in lowered:
            return p
    return None


def find_image_mask_dirs(root: Path) -> Tuple[Path, Path]:
    """Find image/mask pair folders in common polyp-dataset layouts."""
    img = _folder_child(root, ["images", "image", "imgs", "original", "NewTRimage"])
    msk = _folder_child(root, ["masks", "mask", "gts", "gt", "ground truth", "ground_truth", "labels", "NewTRmask"])
    if img and msk:
        return img, msk

    # Sometimes TrainDataset itself directly contains oddly named folders.
    dirs = [p for p in root.rglob("*") if p.is_dir()]
    img_cands = [p for p in dirs if p.name.lower() in {"images", "image", "imgs", "original", "newtrimage"}]
    msk_cands = [p for p in dirs if p.name.lower() in {"masks", "mask", "gts", "gt", "ground truth", "ground_truth", "labels", "newtrmask"}]
    for i in img_cands:
        for m in msk_cands:
            # prefer siblings / close common parent
            if i.parent == m.parent:
                return i, m
    raise FileNotFoundError(f"Cannot find image/mask directories under {root}")


def pair_by_stem(images_dir: Path, masks_dir: Path) -> List[Tuple[Path, Path]]:
    imgs = _image_files(images_dir)
    msks = _image_files(masks_dir)
    md = {p.stem: p for p in msks}
    pairs = [(p, md[p.stem]) for p in imgs if p.stem in md]
    if not pairs:
        # fallback: basename including extension
        md2 = {p.name: p for p in msks}
        pairs = [(p, md2[p.name]) for p in imgs if p.name in md2]
    if not pairs:
        raise RuntimeError(f"No paired images/masks between {images_dir} and {masks_dir}")
    return pairs


def make_split_manifest(train_dataset: Path, manifest_dir: Path, seed: int) -> Tuple[List[Tuple[Path, Path]], List[Tuple[Path, Path]]]:
    images_dir, masks_dir = find_image_mask_dirs(train_dataset)
    pairs = pair_by_stem(images_dir, masks_dir)
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest = manifest_dir / f"split_seed_{seed}.csv"

    if manifest.exists():
        rows = list(csv.DictReader(manifest.open("r", encoding="utf-8")))
        train, val = [], []
        for r in rows:
            pair = (Path(r["image"]), Path(r["mask"]))
            (train if r["split"] == "train" else val).append(pair)
        if train and val and all(a.exists() and b.exists() for a, b in train + val):
            log(f"[OK] Reusing split manifest: {manifest} ({len(train)} train / {len(val)} val)")
            return train, val

    n = len(pairs)
    idx = np.arange(n)
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    # Manuscript states 870/580 from a 1450-image pool. Preserve exact counts when possible.
    n_train = 870 if n == 1450 else int(round(0.60 * n))
    tr_idx, va_idx = idx[:n_train], idx[n_train:]
    train = [pairs[int(i)] for i in tr_idx]
    val = [pairs[int(i)] for i in va_idx]
    with manifest.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["split", "case", "image", "mask"])
        w.writeheader()
        for split, seq in (("train", train), ("val", val)):
            for a, b in seq:
                w.writerow({"split": split, "case": a.stem, "image": str(a), "mask": str(b)})
    log(f"[OK] Created split manifest: {manifest} ({len(train)} train / {len(val)} val)")
    return train, val


def discover_test_sets(test_root: Path) -> Dict[str, Tuple[Path, Path]]:
    result: Dict[str, Tuple[Path, Path]] = {}
    # Standard: TestDataset/<dataset>/images,masks
    for d in sorted([p for p in test_root.iterdir() if p.is_dir()]):
        try:
            img, msk = find_image_mask_dirs(d)
            pairs = pair_by_stem(img, msk)
            if pairs:
                result[d.name] = (img, msk)
        except Exception:
            continue
    return result


def audit_counts(train_pairs_n: int, test_sets: Dict[str, Tuple[Path, Path]], out: Path) -> Dict[str, Any]:
    audit: Dict[str, Any] = {"train_pool": train_pairs_n, "test": {}, "warnings": []}
    if train_pairs_n != 1450:
        audit["warnings"].append(f"Train pool count is {train_pairs_n}, manuscript/common protocol target is 1450.")
    for name, (img, msk) in test_sets.items():
        n = len(pair_by_stem(img, msk))
        audit["test"][name] = n
        expected = EXPECTED_PUBLIC_COUNTS.get(name)
        if expected is not None and n != expected:
            audit["warnings"].append(f"{name}: found {n}, common PraNet protocol expects {expected}.")
    write_json(out, audit)
    return audit


# ---------------------------------------------------------------------------
# Image / box utilities
# ---------------------------------------------------------------------------
def read_rgb(path: Path) -> np.ndarray:
    x = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if x is None:
        raise ValueError(f"Failed to read image: {path}")
    return cv2.cvtColor(x, cv2.COLOR_BGR2RGB)


def read_mask(path: Path) -> np.ndarray:
    x = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if x is None:
        raise ValueError(f"Failed to read mask: {path}")
    if x.ndim == 3:
        x = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
    return (x > 127).astype(np.uint8)


def resize_norm_image(img_rgb: np.ndarray, size: int = 1024) -> np.ndarray:
    x = cv2.resize(img_rgb, (size, size), interpolation=cv2.INTER_CUBIC).astype(np.float32)
    mn, mx = float(x.min()), float(x.max())
    x = (x - mn) / max(mx - mn, 1e-8)
    return x


def mask_box(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    ys, xs = np.nonzero(mask > 0)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def jitter_gt_box(mask: np.ndarray, shift: int, rng: np.random.Generator) -> np.ndarray:
    h, w = mask.shape
    b = mask_box(mask)
    if b is None:
        return np.array([0, 0, w - 1, h - 1], dtype=np.float32)
    x0, y0, x1, y1 = b
    # Same idea as official MedSAM training: expand GT box by random margins.
    x0 = max(0, x0 - int(rng.integers(0, shift + 1)))
    y0 = max(0, y0 - int(rng.integers(0, shift + 1)))
    x1 = min(w - 1, x1 + int(rng.integers(0, shift + 1)))
    y1 = min(h - 1, y1 + int(rng.integers(0, shift + 1)))
    return np.array([x0, y0, x1, y1], dtype=np.float32)


# ---------------------------------------------------------------------------
# Training dataset
# ---------------------------------------------------------------------------
class PolypTrainDataset(Dataset):
    def __init__(self, pairs: Sequence[Tuple[Path, Path]], seed: int, bbox_shift: int = 20, size: int = 1024):
        self.pairs = list(pairs)
        self.seed = int(seed)
        self.bbox_shift = int(bbox_shift)
        self.size = int(size)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        ip, mp = self.pairs[idx]
        img = resize_norm_image(read_rgb(ip), self.size)
        gt0 = read_mask(mp)
        gt = cv2.resize(gt0, (self.size, self.size), interpolation=cv2.INTER_NEAREST).astype(np.uint8)
        rng = np.random.default_rng(self.seed + self.epoch * max(1, len(self.pairs)) + idx)
        box = jitter_gt_box(gt, self.bbox_shift, rng)
        img_t = torch.from_numpy(img).permute(2, 0, 1).float()
        gt_t = torch.from_numpy(gt[None]).float()
        return img_t, gt_t, torch.from_numpy(box).float(), ip.stem


class PolypValDataset(Dataset):
    def __init__(self, pairs: Sequence[Tuple[Path, Path]], size: int = 1024):
        self.pairs = list(pairs)
        self.size = int(size)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        ip, mp = self.pairs[idx]
        img = resize_norm_image(read_rgb(ip), self.size)
        gt0 = read_mask(mp)
        gt = cv2.resize(gt0, (self.size, self.size), interpolation=cv2.INTER_NEAREST).astype(np.uint8)
        b = mask_box(gt)
        box = np.array(b if b is not None else [0, 0, self.size - 1, self.size - 1], dtype=np.float32)
        return torch.from_numpy(img).permute(2, 0, 1).float(), torch.from_numpy(gt[None]).float(), torch.from_numpy(box).float(), ip.stem


# ---------------------------------------------------------------------------
# MedSAM model forward
# ---------------------------------------------------------------------------
class MedSAMTrainWrapper(nn.Module):
    def __init__(self, sam_model: nn.Module, freeze_image_encoder: bool = False):
        super().__init__()
        self.image_encoder = sam_model.image_encoder
        self.mask_decoder = sam_model.mask_decoder
        self.prompt_encoder = sam_model.prompt_encoder
        for p in self.prompt_encoder.parameters():
            p.requires_grad = False
        if freeze_image_encoder:
            for p in self.image_encoder.parameters():
                p.requires_grad = False
        self.freeze_image_encoder = bool(freeze_image_encoder)

    def forward(self, image: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
        if self.freeze_image_encoder:
            with torch.no_grad():
                emb = self.image_encoder(image)
        else:
            emb = self.image_encoder(image)
        with torch.no_grad():
            b = boxes.to(device=image.device, dtype=torch.float32)
            if b.ndim == 2:
                b = b[:, None, :]
            sparse, dense = self.prompt_encoder(points=None, boxes=b, masks=None)
        low, _ = self.mask_decoder(
            image_embeddings=emb,
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=dense,
            multimask_output=False,
        )
        return F.interpolate(low, size=image.shape[-2:], mode="bilinear", align_corners=False)


def dice_bce_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    p = torch.sigmoid(logits)
    inter = (p * target).sum(dim=(1, 2, 3))
    denom = p.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice = 1.0 - ((2 * inter + eps) / (denom + eps)).mean()
    bce = F.binary_cross_entropy_with_logits(logits, target)
    return dice + bce


def load_sam_from_checkpoint(base_ckpt: Path, medsam_repo: Path, device: torch.device):
    if str(medsam_repo) not in sys.path:
        sys.path.insert(0, str(medsam_repo))
    from segment_anything import sam_model_registry
    sam = sam_model_registry["vit_b"](checkpoint=str(base_ckpt))
    return sam.to(device)


def load_sam_state(base_ckpt: Path, trained_state_path: Path, medsam_repo: Path, device: torch.device):
    sam = load_sam_from_checkpoint(base_ckpt, medsam_repo, device)
    obj = torch.load(trained_state_path, map_location="cpu")
    state = obj.get("model", obj) if isinstance(obj, dict) else obj
    # train wrapper prefixes match sam components, so manually load components if wrapper state.
    if any(k.startswith("image_encoder.") for k in state.keys()):
        missing, unexpected = sam.load_state_dict(state, strict=False)
        # Prompt encoder is present in wrapper state as well; strict=False is just robust to metadata variants.
        if unexpected:
            log(f"[WARN] unexpected trained-state keys: {unexpected[:5]}")
    else:
        sam.load_state_dict(state, strict=True)
    return sam.to(device).eval()


# ---------------------------------------------------------------------------
# Train/validate one arm
# ---------------------------------------------------------------------------
def _autocast(device: torch.device, enabled: bool):
    if not enabled or device.type != "cuda":
        return contextlib.nullcontext()
    try:
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    except Exception:
        return torch.cuda.amp.autocast(dtype=torch.float16)


def _make_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except Exception:
        return torch.cuda.amp.GradScaler(enabled=enabled)


@torch.no_grad()
def validate(model: nn.Module, loader: DataLoader, criterion, device: torch.device, amp: bool) -> Dict[str, float]:
    model.eval()
    losses, dices = [], []
    for image, gt, box, _ in loader:
        image, gt, box = image.to(device), gt.to(device), box.to(device)
        with _autocast(device, amp):
            logits = model(image, box)
            loss = criterion(logits, gt)
        p = (torch.sigmoid(logits) > 0.5).float()
        inter = (p * gt).sum(dim=(1, 2, 3))
        denom = p.sum(dim=(1, 2, 3)) + gt.sum(dim=(1, 2, 3))
        d = ((2 * inter + 1e-6) / (denom + 1e-6)).detach().cpu().numpy()
        dices.extend([float(x) for x in d])
        losses.append(float(loss.detach().cpu()))
    model.train()
    return {"val_loss": float(np.mean(losses)), "val_dice": float(np.mean(dices))}


def train_arm(
    arm: str,
    base_ckpt: Path,
    medsam_repo: Path,
    train_pairs: Sequence[Tuple[Path, Path]],
    val_pairs: Sequence[Tuple[Path, Path]],
    work_dir: Path,
    device: torch.device,
    seed: int,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    num_workers: int,
    bbox_shift: int,
    val_every: int,
    patience: int,
    amp: bool,
    freeze_image_encoder: bool,
) -> Path:
    arm_dir = work_dir / "checkpoints" / arm
    best_path = arm_dir / "best.pt"
    last_path = arm_dir / "last.pt"
    hist = arm_dir / "history.csv"
    arm_dir.mkdir(parents=True, exist_ok=True)

    if best_path.exists():
        done_meta = arm_dir / "TRAINING_COMPLETE.json"
        if done_meta.exists():
            log(f"[SKIP] {arm}: completed checkpoint exists: {best_path}")
            return best_path

    seed_everything(seed)
    sam = load_sam_from_checkpoint(base_ckpt, medsam_repo, device)
    model = MedSAMTrainWrapper(sam, freeze_image_encoder=freeze_image_encoder).to(device)
    criterion = core.ABLoss() if arm == "abloss" else dice_bce_loss
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    scaler = _make_scaler(amp and device.type == "cuda")

    train_ds = PolypTrainDataset(train_pairs, seed=seed, bbox_shift=bbox_shift, size=1024)
    val_ds = PolypValDataset(val_pairs, size=1024)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=True)

    start_epoch = 0
    best_val = float("inf")
    best_dice = -1.0
    stale = 0
    if last_path.exists():
        log(f"[RESUME] {arm}: {last_path}")
        state = torch.load(last_path, map_location="cpu")
        model.load_state_dict(state["model"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        if state.get("scaler") is not None:
            try:
                scaler.load_state_dict(state["scaler"])
            except Exception:
                pass
        start_epoch = int(state["epoch"]) + 1
        best_val = float(state.get("best_val", best_val))
        best_dice = float(state.get("best_dice", best_dice))
        stale = int(state.get("stale", 0))

    log(f"\n===== TRAIN {arm.upper()} | epochs={epochs} batch={batch_size} lr={lr} wd={weight_decay} =====")
    log(f"train={len(train_ds)} val={len(val_ds)} trainable_params={sum(p.numel() for p in params):,}")

    for epoch in range(start_epoch, epochs):
        model.train()
        train_ds.set_epoch(epoch)
        g = torch.Generator()
        g.manual_seed(seed + epoch)
        loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True, generator=g,
            num_workers=num_workers, pin_memory=True, drop_last=False,
        )
        total, nstep = 0.0, 0
        t0 = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        for step, (image, gt, box, _) in enumerate(loader, 1):
            image, gt, box = image.to(device, non_blocking=True), gt.to(device, non_blocking=True), box.to(device, non_blocking=True)
            with _autocast(device, amp):
                logits = model(image, box)
                loss = criterion(logits, gt)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            total += float(loss.detach().cpu())
            nstep += 1
            if step == 1 or step % 100 == 0 or step == len(loader):
                log(f"{arm} epoch {epoch+1:03d}/{epochs} step {step:04d}/{len(loader)} loss={float(loss):.5f}")
        scheduler.step()
        train_loss = total / max(1, nstep)
        elapsed = time.perf_counter() - t0

        val_stats = {"val_loss": float("nan"), "val_dice": float("nan")}
        do_val = ((epoch + 1) % val_every == 0) or epoch == 0 or (epoch + 1 == epochs)
        if do_val:
            val_stats = validate(model, val_loader, criterion, device, amp)
            improved = val_stats["val_loss"] < best_val
            if improved:
                best_val = val_stats["val_loss"]
                best_dice = val_stats["val_dice"]
                stale = 0
                atomic_torch_save({
                    "model": model.state_dict(), "epoch": epoch,
                    "best_val": best_val, "best_dice": best_dice,
                    "arm": arm,
                }, best_path)
                log(f"[BEST] {arm}: val_loss={best_val:.6f} val_dice={best_dice:.6f}")
            else:
                stale += 1
        row = {
            "epoch": epoch + 1, "train_loss": train_loss,
            "val_loss": val_stats["val_loss"], "val_dice": val_stats["val_dice"],
            "lr": optimizer.param_groups[0]["lr"], "seconds": elapsed,
        }
        append_csv(hist, row)
        atomic_torch_save({
            "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
            "epoch": epoch, "best_val": best_val, "best_dice": best_dice, "stale": stale,
            "arm": arm,
        }, last_path)

        if do_val and patience > 0 and stale >= patience:
            log(f"[EARLY STOP] {arm}: no val-loss improvement for {stale} validation checks")
            break

    if not best_path.exists():
        # Should only happen if val data is empty (guarded earlier), but keep robust.
        atomic_torch_save({"model": model.state_dict(), "epoch": epochs - 1, "arm": arm}, best_path)
    write_json(arm_dir / "TRAINING_COMPLETE.json", {
        "arm": arm, "best_path": str(best_path), "best_val_loss": best_val,
        "best_val_dice": best_dice, "seed": seed, "epochs_requested": epochs,
        "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    })
    # Remove giant optimizer checkpoint after successful completion; best model is enough for inference.
    if last_path.exists():
        try:
            last_path.unlink()
        except Exception:
            pass
    del model, sam, optimizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return best_path


# ---------------------------------------------------------------------------
# Inference: no GT prompt at test
# ---------------------------------------------------------------------------
@torch.no_grad()
def encode_image(sam, img_rgb: np.ndarray, device: torch.device) -> torch.Tensor:
    x = resize_norm_image(img_rgb, 1024)
    t = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).float().to(device)
    return sam.image_encoder(t)


@torch.no_grad()
def decode_box_probability(sam, embedding: torch.Tensor, box_original: Tuple[int, int, int, int], h: int, w: int) -> np.ndarray:
    box = np.asarray(box_original, dtype=np.float32)[None]
    box1024 = box / np.asarray([w, h, w, h], dtype=np.float32)[None] * 1024.0
    bt = torch.as_tensor(box1024, dtype=torch.float32, device=embedding.device)
    if bt.ndim == 2:
        bt = bt[:, None, :]
    sparse, dense = sam.prompt_encoder(points=None, boxes=bt, masks=None)
    low, _ = sam.mask_decoder(
        image_embeddings=embedding,
        image_pe=sam.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse,
        dense_prompt_embeddings=dense,
        multimask_output=False,
    )
    logits = F.interpolate(low, size=(h, w), mode="bilinear", align_corners=False)
    return torch.sigmoid(logits).squeeze().detach().cpu().numpy().astype(np.float32)


def automatic_coarse_then_refine(sam, img_rgb: np.ndarray, device: torch.device) -> Tuple[np.ndarray, np.ndarray, Optional[Tuple[int, int, int, int]], torch.Tensor, Dict[str, float]]:
    h, w = img_rgb.shape[:2]
    t0 = time.perf_counter()
    emb = encode_image(sam, img_rgb, device)
    t_enc = time.perf_counter() - t0
    full = (0, 0, w - 1, h - 1)
    t1 = time.perf_counter()
    coarse_prob = decode_box_probability(sam, emb, full, h, w)
    coarse_mask = (coarse_prob > 0.5).astype(np.uint8)
    b = mask_box(coarse_mask)
    if b is None:
        b = full
    refined_prob = decode_box_probability(sam, emb, b, h, w)
    t_decode = time.perf_counter() - t1
    return coarse_mask, refined_prob, b, emb, {"encode_s": t_enc, "coarse_plus_refine_decode_s": t_decode}


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_mask(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), (mask.astype(np.uint8) * 255))


def save_map(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, arr.astype(np.float32))


def infer_dataset(
    dataset_name: str,
    image_dir: Path,
    mask_dir: Path,
    sam_baseline,
    sam_abloss,
    output_root: Path,
    device: torch.device,
    usr_cfg: core.USRConfig,
) -> None:
    pairs = pair_by_stem(image_dir, mask_dir)
    out_ds = output_root / "predictions" / dataset_name
    dirs = {}
    for v in ("baseline", "abloss", "usr", "both"):
        dirs[(v, "mask")] = ensure_dir(out_ds / v / "masks")
        dirs[(v, "prob")] = ensure_dir(out_ds / v / "prob")
        dirs[(v, "unc")] = ensure_dir(out_ds / v / "unc")
    latency_csv = out_ds / "latency.csv"
    prompt_csv = out_ds / "prompt_protocol.csv"

    # Resume case-wise: masks are completion markers.
    log(f"\n===== INFER {dataset_name}: {len(pairs)} cases =====")
    usr_engine = core.USR(usr_cfg)

    for i, (ip, mp) in enumerate(pairs, 1):
        stem = ip.stem
        done = all((dirs[(v, "mask")] / f"{stem}.png").exists() for v in ("baseline", "abloss", "usr", "both"))
        if done:
            if i == 1 or i % 50 == 0 or i == len(pairs):
                log(f"{dataset_name} {i}/{len(pairs)} [skip existing] {stem}")
            continue

        img = read_rgb(ip)
        h, w = img.shape[:2]
        case_t0 = time.perf_counter()

        # Baseline-loss model: automatic full-image prompt -> prediction-derived box -> refined prediction.
        c0, pb, box_b, emb_b, tb = automatic_coarse_then_refine(sam_baseline, img, device)
        mb = (pb > 0.5).astype(np.uint8)

        def pred_b(_img, box):
            return decode_box_probability(sam_baseline, emb_b, tuple(map(int, box)), h, w)

        t_usr0 = time.perf_counter()
        rb = usr_engine.run_prompt(img, pred_b, coarse_mask=c0)
        usr_b_s = time.perf_counter() - t_usr0

        # ABLoss model, exact same automatic prompt protocol.
        c1, pa, box_a, emb_a, ta = automatic_coarse_then_refine(sam_abloss, img, device)
        ma = (pa > 0.5).astype(np.uint8)

        def pred_a(_img, box):
            return decode_box_probability(sam_abloss, emb_a, tuple(map(int, box)), h, w)

        t_usr1 = time.perf_counter()
        ra = usr_engine.run_prompt(img, pred_a, coarse_mask=c1)
        usr_a_s = time.perf_counter() - t_usr1

        # Save final masks.
        save_mask(dirs[("baseline", "mask")] / f"{stem}.png", mb)
        save_mask(dirs[("abloss", "mask")] / f"{stem}.png", ma)
        save_mask(dirs[("usr", "mask")] / f"{stem}.png", rb.rectified_mask)
        save_mask(dirs[("both", "mask")] / f"{stem}.png", ra.rectified_mask)
        # Save probabilities. Baseline/ABLoss are refined single-pass probabilities; USR are aggregated pbar.
        save_map(dirs[("baseline", "prob")] / f"{stem}.npy", pb)
        save_map(dirs[("abloss", "prob")] / f"{stem}.npy", pa)
        save_map(dirs[("usr", "prob")] / f"{stem}.npy", rb.aggregated_probability)
        save_map(dirs[("both", "prob")] / f"{stem}.npy", ra.aggregated_probability)
        # Single-pass entropy is saved for completeness; reviewer-critical uncertainty is USR T-pass entropy.
        save_map(dirs[("baseline", "unc")] / f"{stem}.npy", core.USR.bernoulli_entropy(pb))
        save_map(dirs[("abloss", "unc")] / f"{stem}.npy", core.USR.bernoulli_entropy(pa))
        save_map(dirs[("usr", "unc")] / f"{stem}.npy", rb.uncertainty)
        save_map(dirs[("both", "unc")] / f"{stem}.npy", ra.uncertainty)

        append_csv(latency_csv, {
            "case": stem,
            "baseline_encode_s": tb["encode_s"],
            "baseline_coarse_refine_decode_s": tb["coarse_plus_refine_decode_s"],
            "baseline_usr_s": usr_b_s,
            "abloss_encode_s": ta["encode_s"],
            "abloss_coarse_refine_decode_s": ta["coarse_plus_refine_decode_s"],
            "abloss_usr_s": usr_a_s,
            "total_case_s": time.perf_counter() - case_t0,
        })
        append_csv(prompt_csv, {
            "case": stem,
            "test_gt_prompt_used": 0,
            "baseline_init_prompt": "full_image_box",
            "baseline_prediction_box": str(box_b),
            "abloss_init_prompt": "full_image_box",
            "abloss_prediction_box": str(box_a),
            "usr_T": usr_cfg.num_passes,
            "usr_aggregation": usr_cfg.aggregation_mode,
            "usr_normalization": usr_cfg.normalization_mode,
            "usr_scale_spatial_params": int(usr_cfg.scale_spatial_params),
            "usr_baseline_prune_accepted": int(rb.prune_accepted),
            "usr_baseline_repair_accepted": int(rb.repair_accepted),
            "usr_abloss_prune_accepted": int(ra.prune_accepted),
            "usr_abloss_repair_accepted": int(ra.repair_accepted),
        })
        if i == 1 or i % 10 == 0 or i == len(pairs):
            log(f"{dataset_name} {i}/{len(pairs)} {stem} total={time.perf_counter()-case_t0:.2f}s")

        # Do not retain embeddings across cases.
        del emb_b, emb_a
        if device.type == "cuda" and i % 20 == 0:
            torch.cuda.empty_cache()


def run_stage2_analysis(dataset_name: str, image_dir: Path, mask_dir: Path, real_root: Path, script_dir: Path, seed: int) -> None:
    pred = real_root / "predictions" / dataset_name
    out = real_root / "reviewer2_results" / dataset_name
    cmd = [
        sys.executable, str(script_dir / "run_reviewer2_experiments.py"),
        "--gt", str(mask_dir),
        "--baseline", str(pred / "baseline" / "masks"),
        "--abloss", str(pred / "abloss" / "masks"),
        "--usr", str(pred / "usr" / "masks"),
        "--both", str(pred / "both" / "masks"),
        "--prob-baseline", str(pred / "baseline" / "prob"),
        "--unc-baseline", str(pred / "baseline" / "unc"),
        "--prob-abloss", str(pred / "abloss" / "prob"),
        "--unc-abloss", str(pred / "abloss" / "unc"),
        "--prob-usr", str(pred / "usr" / "prob"),
        "--unc-usr", str(pred / "usr" / "unc"),
        "--prob-both", str(pred / "both" / "prob"),
        "--unc-both", str(pred / "both" / "unc"),
        "--output", str(out),
        "--seed", str(seed),
    ]
    run_cmd(cmd)


# ---------------------------------------------------------------------------
# Optional Tumor30 extraction/discovery
# ---------------------------------------------------------------------------
def try_extract_tumor30(search_root: Path, data_root: Path) -> Optional[Path]:
    rars = [p for p in search_root.glob("*.rar") if "tumor" in p.name.lower()]
    if not rars:
        return None
    out = data_root / "Tumor30"
    if out.exists() and any(out.rglob("*")):
        return out
    tool = shutil.which("7z") or shutil.which("7za") or shutil.which("unrar")
    if not tool:
        log("[WARN] Tumor30 .rar found but 7z/unrar is unavailable; public experiments will continue.")
        return None
    out.mkdir(parents=True, exist_ok=True)
    if Path(tool).name.startswith("7z"):
        run_cmd([tool, "x", "-y", f"-o{out}", str(rars[0])], check=False)
    else:
        run_cmd([tool, "x", "-o+", str(rars[0]), str(out)], check=False)
    return out if any(out.rglob("*")) else None


def discover_optional_single_dataset(root: Path, name: str) -> Optional[Tuple[str, Path, Path]]:
    if not root or not root.exists():
        return None
    try:
        img, msk = find_image_mask_dirs(root)
        pairs = pair_by_stem(img, msk)
        if pairs:
            return name, img, msk
    except Exception:
        pass
    # Search child dirs if root contains an extra wrapper.
    for d in [p for p in root.rglob("*") if p.is_dir()]:
        try:
            img, msk = find_image_mask_dirs(d)
            pairs = pair_by_stem(img, msk)
            if pairs:
                return name, img, msk
        except Exception:
            continue
    return None


# ---------------------------------------------------------------------------
# Protocol record
# ---------------------------------------------------------------------------
def protocol_record(args, train_dir: Path, test_dir: Path, train_n: int, val_n: int, usr_cfg: core.USRConfig) -> Dict[str, Any]:
    return {
        "purpose": "Reviewer #2 full 2x2 factorial and USR reliability experiments",
        "data": {
            "train_pool_source": "PraNet common split if auto-downloaded",
            "train_dataset_dir": str(train_dir),
            "test_dataset_dir": str(test_dir),
            "train_n": train_n,
            "val_n": val_n,
            "split_seed": args.seed,
        },
        "training_prompt_protocol": {
            "source": "GT-derived bounding box with random outward jitter",
            "gt_allowed": True,
            "reason": "supervised training annotations",
            "bbox_shift_at_1024": args.bbox_shift,
        },
        "test_prompt_protocol": {
            "gt_box_used": False,
            "stage_1": "full-image box prompt -> coarse probability/mask",
            "stage_2": "derive tight box from coarse predicted mask -> refined prediction",
            "usr": "T prediction-derived box perturbations around coarse-mask box",
        },
        "optimizer": "AdamW",
        "scheduler": "CosineAnnealingLR",
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "epochs_requested": args.epochs,
        "batch_size": args.batch_size,
        "amp": bool(args.amp),
        "freeze_image_encoder": bool(args.freeze_image_encoder),
        "note_on_recovered_hyperparameters": (
            "The lost original code and undisclosed numeric main-training hyperparameters cannot be recovered from the manuscript. "
            "These are revision reproducibility settings and must be reported as such if used for new revision experiments."
        ),
        "abloss": asdict(core.ABLossConfig()),
        "usr": asdict(usr_cfg),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--auto", action="store_true", help="Setup/download + train + infer + analyze in one run")
    p.add_argument("--prepare-only", action="store_true", help="Only setup/download/audit; do not train")
    p.add_argument("--root", default=DATA_ROOT + "/wysiwyr_real", help="Persistent work/output directory")
    p.add_argument("--data-root", default="", help="Existing data root; default <root>/data")
    p.add_argument("--test-rar", default=DATA_ROOT + "/tumor_30.rar", help="Local RAR bundle that already contains TestDataset/*")
    p.add_argument("--medsam-checkpoint", default="", help="Existing medsam_vit_b.pth; otherwise auto-download with Google Drive/Hugging Face/Zenodo fallbacks")
    p.add_argument("--seed", type=int, default=2023, help="Revision reproducibility seed (numeric original was not disclosed)")
    p.add_argument("--epochs", type=int, default=50, help="Revision training cap; early stopping is enabled")
    p.add_argument("--batch-size", type=int, default=1, help="Safe for 24GB 4090-class GPU at 1024x1024")
    p.add_argument("--lr", type=float, default=1e-4, help="Official MedSAM one-GPU default; original paper main LR was undisclosed")
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--bbox-shift", type=int, default=20, help="GT-box outward jitter at 1024 during training")
    p.add_argument("--val-every", type=int, default=5, help="Validate every N epochs")
    p.add_argument("--patience", type=int, default=4, help="Early-stop after N non-improving validation checks; 0 disables")
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--freeze-image-encoder", action="store_true", help="Memory fallback; changes fine-tuning protocol, keep OFF for main run")
    p.add_argument("--skip-download-data", action="store_true", help="Require local TrainDataset/TestDataset instead of auto-download")
    p.add_argument("--skip-tumor30", action="store_true", default=True, help="Keep true by default: the archive labeled/ subset has 435 pairs and must NOT be mislabeled as manuscript Tumor30=162 without a verified mapping")
    p.add_argument("--datasets", default="", help="Comma-separated test-set names; empty = all discovered")
    return p


def main() -> None:
    args = build_parser().parse_args()
    if not (args.auto or args.prepare_only):
        args.auto = True

    script_dir = Path(__file__).resolve().parent
    root = Path(args.root).resolve()
    data_root = Path(args.data_root).resolve() if args.data_root else root / "data"
    downloads = root / "downloads"
    medsam_repo_dir = root / "third_party" / "MedSAM"
    medsam_ckpt = Path(args.medsam_checkpoint).resolve() if args.medsam_checkpoint else root / "weights" / "medsam_vit_b.pth"
    root.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)

    log("\n" + "=" * 78)
    log("WYSIWYR REAL MEDSAM 2x2 PIPELINE")
    log("=" * 78)
    log(f"root:   {root}")
    log(f"device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    if torch.cuda.is_available():
        log(f"gpu:    {torch.cuda.get_device_name(0)}")

    # Required bundle companion files.
    if not (script_dir / "run_reviewer2_experiments.py").exists():
        raise SystemExit("Missing bundled run_reviewer2_experiments.py in same folder.")

    medsam_repo = ensure_medsam_repo(medsam_repo_dir)
    ensure_medsam_checkpoint(medsam_ckpt)

    if args.skip_download_data:
        train_dir = _find_named_dir(data_root, "TrainDataset")
        test_dir = _find_named_dir(data_root, "TestDataset")
        if train_dir is None or test_dir is None:
            raise SystemExit(f"--skip-download-data was set but TrainDataset/TestDataset were not found under {data_root}")
    else:
        train_dir, test_dir = ensure_public_data(data_root, downloads, Path(args.test_rar).resolve() if args.test_rar else None)

    train_pairs, val_pairs = make_split_manifest(train_dir, root / "manifests", args.seed)
    test_sets = discover_test_sets(test_dir)
    audit = audit_counts(len(train_pairs) + len(val_pairs), test_sets, root / "DATASET_COUNT_AUDIT.json")
    if audit["warnings"]:
        log("\n[DATASET AUDIT WARNINGS]")
        for w in audit["warnings"]:
            log(" - " + w)

    # IMPORTANT: do not silently treat the archive's generic labeled/ subset as Tumor30.
    # The inspected archive has 435 labeled pairs, while the manuscript reports Tumor30=162.
    # Include Tumor30 only when a clearly named Tumor30/Tumor_30 directory is supplied later.
    if not args.skip_tumor30:
        named = None
        for pth in data_root.rglob("*"):
            if pth.is_dir() and pth.name.lower() in {"tumor30", "tumor_30"}:
                named = pth
                break
        opt = discover_optional_single_dataset(named, "Tumor30") if named else None
        if opt:
            name, img, msk = opt
            n_t = len(pair_by_stem(img, msk))
            if n_t == 162:
                test_sets[name] = (img, msk)
            else:
                log(f"[WARN] Found named Tumor30 directory with {n_t} pairs, not manuscript count 162; excluded pending verification.")
    else:
        log("[INFO] Tumor30 internal subset is intentionally excluded for now: local archive labeled/ has 435 pairs, not the manuscript's 162.")

    usr_cfg = core.USRConfig(
        num_passes=8,
        aggregation_mode="soft",              # reviewer-oriented continuous probability aggregation
        normalization_mode="none",            # avoid per-image min-max threshold reinterpretation
        scale_spatial_params=True,             # normalize morphology/area params by resolution
        seed=args.seed,
    )
    write_json(root / "protocol.json", protocol_record(args, train_dir, test_dir, len(train_pairs), len(val_pairs), usr_cfg))

    if args.prepare_only:
        log("\nPREPARATION COMPLETE")
        log(f"Protocol: {root / 'protocol.json'}")
        log(f"Audit:    {root / 'DATASET_COUNT_AUDIT.json'}")
        return

    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU is required for real MedSAM training/inference.")
    device = torch.device("cuda:0")
    if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
        torch.backends.cuda.matmul.allow_tf32 = True

    # Train the two model cells from identical MedSAM initialization.
    baseline_ckpt = train_arm(
        "baseline", medsam_ckpt, medsam_repo, train_pairs, val_pairs, root, device,
        args.seed, args.epochs, args.batch_size, args.lr, args.weight_decay,
        args.num_workers, args.bbox_shift, args.val_every, args.patience, args.amp,
        args.freeze_image_encoder,
    )
    abloss_ckpt = train_arm(
        "abloss", medsam_ckpt, medsam_repo, train_pairs, val_pairs, root, device,
        args.seed, args.epochs, args.batch_size, args.lr, args.weight_decay,
        args.num_workers, args.bbox_shift, args.val_every, args.patience, args.amp,
        args.freeze_image_encoder,
    )

    log("\nLoading best trained models for test inference ...")
    sam_b = load_sam_state(medsam_ckpt, baseline_ckpt, medsam_repo, device)
    sam_a = load_sam_state(medsam_ckpt, abloss_ckpt, medsam_repo, device)

    selected = set(x.strip() for x in args.datasets.split(",") if x.strip())
    if selected:
        missing = selected.difference(test_sets.keys())
        if missing:
            log(f"[WARN] requested datasets not discovered: {sorted(missing)}")
        test_sets = {k: v for k, v in test_sets.items() if k in selected}

    for name, (img_dir, msk_dir) in test_sets.items():
        infer_dataset(name, img_dir, msk_dir, sam_b, sam_a, root, device, usr_cfg)
        run_stage2_analysis(name, img_dir, msk_dir, root, script_dir, args.seed)

    write_json(root / "RUN_COMPLETE.json", {
        "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "datasets": list(test_sets.keys()),
        "baseline_checkpoint": str(baseline_ckpt),
        "abloss_checkpoint": str(abloss_ckpt),
        "results_root": str(root / "reviewer2_results"),
    })
    log("\n" + "=" * 78)
    log("REAL 2x2 EXPERIMENT PIPELINE COMPLETE")
    log(f"Predictions: {root / 'predictions'}")
    log(f"Analyses:    {root / 'reviewer2_results'}")
    log(f"Protocol:    {root / 'protocol.json'}")
    log("=" * 78)


if __name__ == "__main__":
    main()
