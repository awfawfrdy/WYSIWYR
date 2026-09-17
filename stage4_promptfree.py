#!/usr/bin/env python3
"""
WYSIWYR Stage 4: reviewer-grade prompt repair for MedSAM.

Purpose
-------
The Stage-3 full-image-box bootstrap is retained only as a failed diagnostic.
This Stage-4 pipeline:
  A) runs a GT-box ORACLE DIAGNOSTIC on held-out test sets (never used as main result),
  B) trains one prompt-free U-Net coarse proposer on the SAME fixed 870/580 train/val split,
  C) calibrates proposal threshold + normalized box margin ONLY on the 580-image validation split,
  D) uses the SAME prediction-derived proposal box for both trained MedSAM arms,
  E) runs the full 2x2 test factorial:
       MedSAM / MedSAM+ABLoss / MedSAM+USR / MedSAM+ABLoss+USR,
  F) runs Reviewer-2 segmentation, factorial, uncertainty, and USR audits.

Main-test leakage rule
----------------------
GT masks are never used to construct a prompt in the main Stage-4 test inference.
GT is read only AFTER prediction for metric computation. Oracle outputs are written to a
separate diagnostic-only directory and are never mixed with main results.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    _v = os.environ.get(_k)
    if _v is not None:
        try:
            if int(_v) <= 0:
                raise ValueError
        except Exception:
            os.environ[_k] = "1"

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

try:
    import run_real_medsam_2x2 as s3
    import wysiwyr_autodl_allinone as core
except Exception as e:
    raise SystemExit(
        "Stage4 companion files are missing. Keep stage4_promptfree.py, "
        "run_real_medsam_2x2.py, run_reviewer2_experiments.py and "
        "wysiwyr_autodl_allinone.py in the same folder.\n"
        f"Original import error: {e}"
    ) from e


# ------------------------------- basics ------------------------------------
def log(msg: str = "") -> None:
    print(msg, flush=True)


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: List[str] = []
    seen = set()
    for r in rows:
        for k in r:
            if k not in seen:
                keys.append(k); seen.add(k)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def append_csv(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            w.writeheader()
        w.writerow(row)


def mean_finite(xs: Sequence[float]) -> float:
    a = np.asarray(xs, dtype=float)
    a = a[np.isfinite(a)]
    return float(a.mean()) if len(a) else float("nan")


def require_stage3_assets(root: Path) -> Tuple[Path, Path, Path, Path, Path]:
    data_root = root / "data"
    train_dir = s3._find_named_dir(data_root, "TrainDataset")
    test_dir = s3._find_named_dir(data_root, "TestDataset")
    base = root / "weights" / "medsam_vit_b.pth"
    b = root / "checkpoints" / "baseline" / "best.pt"
    a = root / "checkpoints" / "abloss" / "best.pt"
    missing = []
    for name, p in (("TrainDataset", train_dir), ("TestDataset", test_dir),
                    ("medsam_vit_b.pth", base), ("baseline/best.pt", b), ("abloss/best.pt", a)):
        if p is None or not Path(p).exists():
            missing.append(name)
    if missing:
        raise SystemExit("Missing Stage-3 assets: " + ", ".join(missing))
    return Path(train_dir), Path(test_dir), base, b, a


# ------------------------- ORACLE diagnostic -------------------------------
@torch.no_grad()
def run_oracle(
    test_sets: Dict[str, Tuple[Path, Path]],
    sam_b,
    sam_a,
    device: torch.device,
    out_root: Path,
) -> Dict[str, Any]:
    case_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []
    log("\n" + "=" * 78)
    log("PHASE A - GT-BOX ORACLE DIAGNOSTIC (DIAGNOSTIC ONLY; NOT MAIN RESULT)")
    log("=" * 78)
    for ds, (img_dir, mask_dir) in test_sets.items():
        pairs = s3.pair_by_stem(img_dir, mask_dir)
        vals = {"baseline": [], "abloss": []}
        for i, (ip, mp) in enumerate(pairs, 1):
            img = s3.read_rgb(ip)
            gt = s3.read_mask(mp)
            h, w = gt.shape
            box = s3.mask_box(gt)
            if box is None:
                box = (0, 0, w - 1, h - 1)
            for arm, sam in (("baseline", sam_b), ("abloss", sam_a)):
                emb = s3.encode_image(sam, img, device)
                prob = s3.decode_box_probability(sam, emb, box, h, w)
                pred = (prob > 0.5).astype(np.uint8)
                m = core.segmentation_metrics(pred, gt)
                vals[arm].append(float(m["dice"]))
                case_rows.append({"dataset": ds, "case": ip.stem, "arm": arm,
                                  "prompt": "GT_tight_box_DIAGNOSTIC_ONLY", **m})
                del emb
            if i == 1 or i % 50 == 0 or i == len(pairs):
                log(f"oracle {ds}: {i}/{len(pairs)}")
        for arm in ("baseline", "abloss"):
            rows = [r for r in case_rows if r["dataset"] == ds and r["arm"] == arm]
            sr = {"dataset": ds, "arm": arm, "n": len(rows)}
            for k in ("dice", "iou", "precision", "recall", "boundary_dice", "hd95", "assd"):
                sr[k] = mean_finite([float(r[k]) for r in rows])
            summary_rows.append(sr)
            log(f"  {ds:22s} {arm:8s} oracle Dice={sr['dice']:.4f}")
    write_csv(out_root / "oracle" / "oracle_case_metrics.csv", case_rows)
    write_csv(out_root / "oracle" / "oracle_summary.csv", summary_rows)
    macro_b = mean_finite([r["dice"] for r in summary_rows if r["arm"] == "baseline"])
    macro_a = mean_finite([r["dice"] for r in summary_rows if r["arm"] == "abloss"])
    result = {"macro_dice_baseline": macro_b, "macro_dice_abloss": macro_a,
              "diagnostic_only": True, "gt_prompt_used": True}
    write_json(out_root / "oracle" / "ORACLE_DIAGNOSTIC.json", result)
    log(f"Oracle macro Dice: baseline={macro_b:.4f}, ABLoss={macro_a:.4f}")
    return result


# -------------------------- prompt-free U-Net ------------------------------
class ConvBlock(nn.Module):
    def __init__(self, c1: int, c2: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c1, c2, 3, padding=1, bias=False), nn.BatchNorm2d(c2), nn.ReLU(inplace=True),
            nn.Conv2d(c2, c2, 3, padding=1, bias=False), nn.BatchNorm2d(c2), nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.net(x)


class SmallUNet(nn.Module):
    def __init__(self, base: int = 24):
        super().__init__()
        self.e1 = ConvBlock(3, base)
        self.e2 = ConvBlock(base, base * 2)
        self.e3 = ConvBlock(base * 2, base * 4)
        self.e4 = ConvBlock(base * 4, base * 8)
        self.mid = ConvBlock(base * 8, base * 16)
        self.u4 = nn.ConvTranspose2d(base * 16, base * 8, 2, 2); self.d4 = ConvBlock(base * 16, base * 8)
        self.u3 = nn.ConvTranspose2d(base * 8, base * 4, 2, 2); self.d3 = ConvBlock(base * 8, base * 4)
        self.u2 = nn.ConvTranspose2d(base * 4, base * 2, 2, 2); self.d2 = ConvBlock(base * 4, base * 2)
        self.u1 = nn.ConvTranspose2d(base * 2, base, 2, 2); self.d1 = ConvBlock(base * 2, base)
        self.out = nn.Conv2d(base, 1, 1)

    @staticmethod
    def cat(up, skip):
        if up.shape[-2:] != skip.shape[-2:]:
            up = F.interpolate(up, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return torch.cat([up, skip], dim=1)

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(F.max_pool2d(e1, 2))
        e3 = self.e3(F.max_pool2d(e2, 2))
        e4 = self.e4(F.max_pool2d(e3, 2))
        m = self.mid(F.max_pool2d(e4, 2))
        d4 = self.d4(self.cat(self.u4(m), e4))
        d3 = self.d3(self.cat(self.u3(d4), e3))
        d2 = self.d2(self.cat(self.u2(d3), e2))
        d1 = self.d1(self.cat(self.u1(d2), e1))
        return self.out(d1)


class ProposalDataset(Dataset):
    def __init__(self, pairs: Sequence[Tuple[Path, Path]], size: int, train: bool, seed: int):
        self.pairs = list(pairs); self.size = int(size); self.train = bool(train); self.seed = int(seed); self.epoch = 0
    def __len__(self): return len(self.pairs)
    def set_epoch(self, e: int): self.epoch = int(e)
    def __getitem__(self, idx: int):
        ip, mp = self.pairs[idx]
        img = s3.read_rgb(ip)
        gt = s3.read_mask(mp)
        img = cv2.resize(img, (self.size, self.size), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        gt = cv2.resize(gt, (self.size, self.size), interpolation=cv2.INTER_NEAREST).astype(np.float32)
        if self.train:
            rng = np.random.default_rng(self.seed + self.epoch * max(1, len(self.pairs)) + idx)
            if rng.random() < 0.5:
                img = np.flip(img, 1).copy(); gt = np.flip(gt, 1).copy()
            if rng.random() < 0.2:
                img = np.flip(img, 0).copy(); gt = np.flip(gt, 0).copy()
            k = int(rng.integers(0, 4))
            if k:
                img = np.rot90(img, k).copy(); gt = np.rot90(gt, k).copy()
            # Mild intensity jitter; segmentation geometry remains unchanged.
            gain = float(rng.uniform(0.90, 1.10)); bias = float(rng.uniform(-0.04, 0.04))
            img = np.clip(img * gain + bias, 0.0, 1.0)
        return torch.from_numpy(img).permute(2, 0, 1).float(), torch.from_numpy(gt[None]).float(), ip.stem


def proposal_loss(logits: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, gt)
    p = torch.sigmoid(logits)
    inter = (p * gt).sum((1, 2, 3)); den = p.sum((1, 2, 3)) + gt.sum((1, 2, 3))
    dice = 1.0 - ((2.0 * inter + 1e-6) / (den + 1e-6)).mean()
    return bce + dice


def make_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except Exception:
        return torch.cuda.amp.GradScaler(enabled=enabled)


def autocast_ctx(enabled: bool):
    if not enabled:
        import contextlib
        return contextlib.nullcontext()
    try:
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    except Exception:
        return torch.cuda.amp.autocast(dtype=torch.float16)


@torch.no_grad()
def proposal_val(model: nn.Module, loader: DataLoader, device: torch.device, amp: bool) -> Dict[str, float]:
    model.eval(); losses = []; dices = []
    for x, y, _ in loader:
        x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
        with autocast_ctx(amp):
            z = model(x); loss = proposal_loss(z, y)
        p = (torch.sigmoid(z) > 0.5).float()
        inter = (p*y).sum((1,2,3)); den = p.sum((1,2,3))+y.sum((1,2,3))
        d = (2*inter+1e-6)/(den+1e-6)
        losses.extend([float(loss.detach().cpu())] * len(x)); dices.extend([float(v) for v in d.detach().cpu().numpy()])
    return {"loss": float(np.mean(losses)), "dice": float(np.mean(dices))}


def train_proposer(
    train_pairs: Sequence[Tuple[Path, Path]], val_pairs: Sequence[Tuple[Path, Path]], out_root: Path,
    device: torch.device, seed: int, size: int, epochs: int, batch_size: int, workers: int,
    lr: float, patience: int, amp: bool,
) -> Path:
    ckpt_dir = out_root / "proposal" / "checkpoints"; ckpt_dir.mkdir(parents=True, exist_ok=True)
    best = ckpt_dir / "best.pt"; done = ckpt_dir / "TRAINING_COMPLETE.json"
    if best.exists() and done.exists():
        log(f"[SKIP] prompt-free proposer already trained: {best}")
        return best
    seed_all(seed)
    model = SmallUNet().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))
    scaler = make_scaler(amp)
    tr = ProposalDataset(train_pairs, size, True, seed); va = ProposalDataset(val_pairs, size, False, seed)
    gen = torch.Generator(); gen.manual_seed(seed)
    train_loader = DataLoader(tr, batch_size=batch_size, shuffle=True, generator=gen, num_workers=workers,
                              pin_memory=True, drop_last=False)
    val_loader = DataLoader(va, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)
    best_d = -1.0; stale = 0; hist = []
    log("\n" + "=" * 78); log("PHASE B - TRAIN PROMPT-FREE COARSE PROPOSER (TRAIN SPLIT ONLY)"); log("=" * 78)
    for ep in range(epochs):
        tr.set_epoch(ep); model.train(); total = 0.0; n = 0; t0 = time.time()
        for x, y, _ in train_loader:
            x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with autocast_ctx(amp):
                z = model(x); loss = proposal_loss(z, y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            total += float(loss.detach().cpu()) * len(x); n += len(x)
        sch.step(); vs = proposal_val(model, val_loader, device, amp)
        row = {"epoch": ep+1, "train_loss": total/max(n,1), "val_loss": vs["loss"], "val_dice": vs["dice"],
               "lr": opt.param_groups[0]["lr"], "seconds": time.time()-t0}
        hist.append(row); write_csv(out_root / "proposal" / "training_history.csv", hist)
        log(f"proposal epoch {ep+1:03d}/{epochs} train_loss={row['train_loss']:.4f} val_loss={vs['loss']:.4f} val_dice={vs['dice']:.4f}")
        if vs["dice"] > best_d + 1e-5:
            best_d = vs["dice"]; stale = 0
            torch.save({"model": model.state_dict(), "val_dice": best_d, "epoch": ep, "size": size}, best)
            log(f"[BEST] proposal val_dice={best_d:.4f}")
        else:
            stale += 1
        if patience > 0 and stale >= patience:
            log(f"[EARLY STOP] proposal after {stale} non-improving epochs")
            break
    write_json(done, {"best_checkpoint": str(best), "best_val_dice": best_d, "seed": seed, "input_size": size})
    del model; torch.cuda.empty_cache()
    return best


def load_proposer(path: Path, device: torch.device) -> Tuple[nn.Module, int]:
    obj = torch.load(path, map_location="cpu")
    model = SmallUNet(); model.load_state_dict(obj["model"]); model.to(device).eval()
    return model, int(obj.get("size", 320))


@torch.no_grad()
def proposer_prob(model: nn.Module, img_rgb: np.ndarray, size: int, device: torch.device, amp: bool) -> np.ndarray:
    x = cv2.resize(img_rgb, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    t = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).float().to(device)
    with autocast_ctx(amp):
        p = torch.sigmoid(model(t))[0,0].float().cpu().numpy()
    return np.clip(p.astype(np.float32), 0, 1)


def largest_component(mask: np.ndarray) -> np.ndarray:
    m = (mask > 0).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    if n <= 1:
        return np.zeros_like(m)
    j = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (lab == j).astype(np.uint8)


def box_from_prob(prob: np.ndarray, threshold: float, margin: float) -> Tuple[np.ndarray, Tuple[int,int,int,int], str]:
    h, w = prob.shape
    m = largest_component(prob >= threshold)
    fallback = "none"
    if int(m.sum()) == 0:
        adaptive = max(0.05, min(float(threshold), 0.60 * float(prob.max())))
        m = largest_component(prob >= adaptive)
        fallback = "adaptive_threshold"
    if int(m.sum()) == 0:
        yy, xx = np.unravel_index(int(np.argmax(prob)), prob.shape)
        rw = max(8, int(round(0.20*w))); rh = max(8, int(round(0.20*h)))
        x0=max(0,xx-rw//2); x1=min(w-1,xx+rw//2); y0=max(0,yy-rh//2); y1=min(h-1,yy+rh//2)
        m[y0:y1+1, x0:x1+1] = 1; fallback = "argmax_window"
    b = s3.mask_box(m)
    assert b is not None
    x0,y0,x1,y1 = b
    mx = int(round(margin * w)); my = int(round(margin * h))
    x0=max(0,x0-mx); y0=max(0,y0-my); x1=min(w-1,x1+mx); y1=min(h-1,y1+my)
    return m, (x0,y0,x1,y1), fallback


def box_metrics(box: Tuple[int,int,int,int], gt: np.ndarray) -> Dict[str,float]:
    h,w = gt.shape; gb = s3.mask_box(gt)
    if gb is None:
        return {"box_iou": float("nan"), "lesion_coverage": float("nan"), "box_area_ratio": float("nan")}
    x0,y0,x1,y1 = box; gx0,gy0,gx1,gy1 = gb
    ix0=max(x0,gx0); iy0=max(y0,gy0); ix1=min(x1,gx1); iy1=min(y1,gy1)
    inter=max(0,ix1-ix0+1)*max(0,iy1-iy0+1)
    a1=(x1-x0+1)*(y1-y0+1); a2=(gx1-gx0+1)*(gy1-gy0+1)
    biou=inter/max(a1+a2-inter,1)
    inside = gt[y0:y1+1,x0:x1+1].sum(); cov=float(inside/max(int(gt.sum()),1))
    return {"box_iou": float(biou), "lesion_coverage": cov, "box_area_ratio": float(a1/(h*w))}


def calibrate_proposal(
    model: nn.Module, val_pairs: Sequence[Tuple[Path, Path]], size: int, device: torch.device,
    out_root: Path, amp: bool,
) -> Dict[str,float]:
    selected_path = out_root / "proposal" / "selected_prompt_params.json"
    if selected_path.exists():
        cfg = json.loads(selected_path.read_text(encoding="utf-8")); log(f"[SKIP] reuse proposal calibration: {cfg}"); return cfg
    log("\n" + "=" * 78); log("PHASE C - VALIDATION-ONLY PROMPT CALIBRATION"); log("=" * 78)
    thresholds = [0.25,0.35,0.45,0.55,0.65]
    margins = [0.02,0.05,0.08,0.10,0.15]
    probs: List[np.ndarray] = []; gts: List[np.ndarray] = []
    for i,(ip,mp) in enumerate(val_pairs,1):
        probs.append(proposer_prob(model,s3.read_rgb(ip),size,device,amp));
        gts.append(cv2.resize(s3.read_mask(mp),(size,size),interpolation=cv2.INTER_NEAREST).astype(np.uint8))
        if i==1 or i%100==0 or i==len(val_pairs): log(f"proposal val predictions {i}/{len(val_pairs)}")
    rows=[]
    for th in thresholds:
        for mg in margins:
            ious=[]; covs=[]; areas=[]; fb=0
            for p,g in zip(probs,gts):
                _,b,f=box_from_prob(p,th,mg); z=box_metrics(b,g)
                ious.append(z["box_iou"]); covs.append(z["lesion_coverage"]); areas.append(z["box_area_ratio"]); fb += int(f!="none")
            rows.append({"threshold":th,"margin_fraction":mg,"mean_box_iou":mean_finite(ious),
                         "mean_lesion_coverage":mean_finite(covs),"mean_box_area_ratio":mean_finite(areas),
                         "fallback_rate":fb/len(gts)})
    write_csv(out_root/"proposal"/"validation_prompt_calibration_grid.csv",rows)
    feasible=[r for r in rows if r["mean_lesion_coverage"]>=0.95]
    if feasible:
        best=max(feasible,key=lambda r:(r["mean_box_iou"],-r["mean_box_area_ratio"]))
        rule="max validation box IoU among settings with mean lesion coverage >= 0.95"
    else:
        best=max(rows,key=lambda r:(0.70*r["mean_lesion_coverage"]+0.30*r["mean_box_iou"]))
        rule="fallback: maximize 0.70*coverage + 0.30*box IoU because no grid cell reached 0.95 coverage"
    cfg={"threshold":float(best["threshold"]),"margin_fraction":float(best["margin_fraction"]),
         "selection_rule":rule,"validation_mean_box_iou":float(best["mean_box_iou"]),
         "validation_mean_lesion_coverage":float(best["mean_lesion_coverage"]),
         "validation_mean_box_area_ratio":float(best["mean_box_area_ratio"]),
         "validation_fallback_rate":float(best["fallback_rate"])}
    write_json(selected_path,cfg); log(f"Selected proposal prompt params: {cfg}")
    return cfg


# --------------------------- main 2x2 inference -----------------------------
def save_mask(path: Path, m: np.ndarray) -> None:
    path.parent.mkdir(parents=True,exist_ok=True); cv2.imwrite(str(path),(m.astype(np.uint8)*255))

def save_map(path: Path, x: np.ndarray) -> None:
    path.parent.mkdir(parents=True,exist_ok=True); np.save(path,x.astype(np.float32))


def usr_from_fixed_box(engine: core.USR, image: np.ndarray, predictor, box: Tuple[int,int,int,int]) -> core.USRResult:
    h,w=image.shape[:2]; rng=np.random.default_rng(engine.cfg.seed); boxes=[]; preds=[]
    for _ in range(engine.cfg.num_passes):
        jb=engine.jitter_box(box,h,w,rng); boxes.append(jb); preds.append(predictor(image,jb))
    pbar,m0,u=engine.aggregate(preds); r=engine.rectify(pbar,u,m0)
    r.prediction_box=box; r.perturbed_boxes=boxes
    return r


@torch.no_grad()
def infer_promptfree_dataset(
    ds: str, img_dir: Path, mask_dir: Path, proposer: nn.Module, proposal_size: int,
    prompt_cfg: Dict[str,float], sam_b, sam_a, device: torch.device, out_root: Path,
    usr_cfg: core.USRConfig, amp: bool,
) -> None:
    pairs=s3.pair_by_stem(img_dir,mask_dir); out=out_root/"predictions"/ds
    dirs={}
    for v in ("baseline","abloss","usr","both"):
        for kind in ("masks","prob","unc"):
            dirs[(v,kind)]=out/v/kind; dirs[(v,kind)].mkdir(parents=True,exist_ok=True)
    (out/"proposal"/"masks").mkdir(parents=True,exist_ok=True)
    prompt_csv=out/"prompt_protocol.csv"; latency_csv=out/"latency.csv"; diag_csv=out/"proposal_test_diagnostics.csv"
    engine=core.USR(usr_cfg); log(f"\n===== STAGE4 PROMPT-FREE INFER {ds}: {len(pairs)} cases =====")
    def _csv_cases(path: Path) -> set:
        if not path.exists() or path.stat().st_size == 0:
            return set()
        try:
            return {r.get("case", "") for r in csv.DictReader(path.open("r", encoding="utf-8-sig"))}
        except Exception:
            return set()
    diag_done=_csv_cases(diag_csv); prompt_done=_csv_cases(prompt_csv); latency_done=_csv_cases(latency_csv)
    for i,(ip,mp) in enumerate(pairs,1):
        stem=ip.stem
        outputs_done=all((dirs[(v,"masks")]/f"{stem}.png").exists() for v in ("baseline","abloss","usr","both"))
        metadata_done=(stem in diag_done and stem in prompt_done and stem in latency_done)
        done=outputs_done and metadata_done
        if done:
            if i==1 or i%50==0 or i==len(pairs): log(f"{ds} {i}/{len(pairs)} [skip existing] {stem}")
            continue
        img=s3.read_rgb(ip); h,w=img.shape[:2]; t0=time.perf_counter()
        # IMPORTANT: derive proposal BEFORE reading GT. Main prompt is image-only.
        pp=proposer_prob(proposer,img,proposal_size,device,amp)
        pm_small,b_small,fallback=box_from_prob(pp,float(prompt_cfg["threshold"]),float(prompt_cfg["margin_fraction"]))
        sx=w/proposal_size; sy=h/proposal_size
        bx0,by0,bx1,by1=b_small
        box=(max(0,int(round(bx0*sx))),max(0,int(round(by0*sy))),min(w-1,int(round((bx1+1)*sx)-1)),min(h-1,int(round((by1+1)*sy)-1)))
        pm=cv2.resize(pm_small,(w,h),interpolation=cv2.INTER_NEAREST).astype(np.uint8)
        save_mask(out/"proposal"/"masks"/f"{stem}.png",pm)
        t_prop=time.perf_counter()-t0

        eb=s3.encode_image(sam_b,img,device); ea=s3.encode_image(sam_a,img,device)
        pb=s3.decode_box_probability(sam_b,eb,box,h,w); pa=s3.decode_box_probability(sam_a,ea,box,h,w)
        mb=(pb>0.5).astype(np.uint8); ma=(pa>0.5).astype(np.uint8)
        def pred_b(_img,b): return s3.decode_box_probability(sam_b,eb,tuple(map(int,b)),h,w)
        def pred_a(_img,b): return s3.decode_box_probability(sam_a,ea,tuple(map(int,b)),h,w)
        tu0=time.perf_counter(); rb=usr_from_fixed_box(engine,img,pred_b,box); usr_b=time.perf_counter()-tu0
        tu1=time.perf_counter(); ra=usr_from_fixed_box(engine,img,pred_a,box); usr_a=time.perf_counter()-tu1
        save_mask(dirs[("baseline","masks")]/f"{stem}.png",mb); save_mask(dirs[("abloss","masks")]/f"{stem}.png",ma)
        save_mask(dirs[("usr","masks")]/f"{stem}.png",rb.rectified_mask); save_mask(dirs[("both","masks")]/f"{stem}.png",ra.rectified_mask)
        save_map(dirs[("baseline","prob")]/f"{stem}.npy",pb); save_map(dirs[("abloss","prob")]/f"{stem}.npy",pa)
        save_map(dirs[("usr","prob")]/f"{stem}.npy",rb.aggregated_probability); save_map(dirs[("both","prob")]/f"{stem}.npy",ra.aggregated_probability)
        save_map(dirs[("baseline","unc")]/f"{stem}.npy",core.USR.bernoulli_entropy(pb)); save_map(dirs[("abloss","unc")]/f"{stem}.npy",core.USR.bernoulli_entropy(pa))
        save_map(dirs[("usr","unc")]/f"{stem}.npy",rb.uncertainty); save_map(dirs[("both","unc")]/f"{stem}.npy",ra.uncertainty)
        # GT enters only here, after all four predictions have been constructed.
        gt=s3.read_mask(mp); bm=box_metrics(box,gt); prop_dice=core.dice_score(pm,gt)
        append_csv(diag_csv,{"case":stem,"proposal_threshold":prompt_cfg["threshold"],"margin_fraction":prompt_cfg["margin_fraction"],
                             "fallback":fallback,"proposal_dice":prop_dice,**bm})
        append_csv(prompt_csv,{"case":stem,"test_gt_prompt_used":0,"proposal_source":"prompt_free_UNet_image_only",
                               "shared_prompt_box":str(box),"same_box_for_baseline_and_abloss":1,
                               "threshold_selected_on":"580-image_validation_only","threshold":prompt_cfg["threshold"],
                               "margin_fraction":prompt_cfg["margin_fraction"],"usr_T":usr_cfg.num_passes,
                               "usr_aggregation":usr_cfg.aggregation_mode,"usr_normalization":usr_cfg.normalization_mode,
                               "usr_scale_spatial_params":int(usr_cfg.scale_spatial_params)})
        append_csv(latency_csv,{"case":stem,"proposal_s":t_prop,"usr_baseline_s":usr_b,"usr_abloss_s":usr_a,"total_case_s":time.perf_counter()-t0})
        if i==1 or i%10==0 or i==len(pairs): log(f"{ds} {i}/{len(pairs)} box={box} fallback={fallback} total={time.perf_counter()-t0:.2f}s")
        del eb,ea
        if i%20==0: torch.cuda.empty_cache()


def summarize_proposal_test(ds: str, out_root: Path) -> None:
    p=out_root/"predictions"/ds/"proposal_test_diagnostics.csv"
    if not p.exists(): return
    rows=list(csv.DictReader(p.open("r",encoding="utf-8-sig")))
    if not rows: return
    out={"dataset":ds,"n":len(rows)}
    for k in ("proposal_dice","box_iou","lesion_coverage","box_area_ratio"):
        out[k]=mean_finite([float(r[k]) for r in rows])
    out["fallback_rate"]=float(np.mean([r["fallback"]!="none" for r in rows]))
    write_csv(out_root/"proposal"/f"test_summary_{ds}.csv",[out])


# -------------------------------- main --------------------------------------
def parser() -> argparse.ArgumentParser:
    p=argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--auto",action="store_true")
    p.add_argument("--root",default=str(Path(os.environ.get("WYSIWYR_DATA_ROOT","."))/"wysiwyr_real"),
                   help="Data/artefact root; defaults to $WYSIWYR_DATA_ROOT/wysiwyr_real")
    p.add_argument("--seed",type=int,default=2023)
    p.add_argument("--proposal-size",type=int,default=320)
    p.add_argument("--proposal-epochs",type=int,default=30)
    p.add_argument("--proposal-batch-size",type=int,default=8)
    p.add_argument("--proposal-lr",type=float,default=3e-4)
    p.add_argument("--proposal-patience",type=int,default=6)
    p.add_argument("--num-workers",type=int,default=0)
    p.add_argument("--amp",action=argparse.BooleanOptionalAction,default=True)
    p.add_argument("--datasets",default="")
    p.add_argument("--oracle-only",action="store_true")
    p.add_argument("--skip-oracle",action="store_true")
    p.add_argument("--oracle-fail-threshold",type=float,default=0.55,
                   help="Abort automatic main run only if BOTH oracle macro Dice scores are below this diagnostic threshold")
    return p


def main() -> None:
    args=parser().parse_args(); seed_all(args.seed)
    if not torch.cuda.is_available(): raise SystemExit("CUDA GPU is required.")
    device=torch.device("cuda:0"); root=Path(args.root).resolve(); script_dir=Path(__file__).resolve().parent
    out_root=root/"stage4_promptfree"; out_root.mkdir(parents=True,exist_ok=True)
    train_dir,test_dir,base_ckpt,b_ckpt,a_ckpt=require_stage3_assets(root)
    medsam_repo=root/"third_party"/"MedSAM"
    train_pairs,val_pairs=s3.make_split_manifest(train_dir,root/"manifests",args.seed)
    test_sets=s3.discover_test_sets(test_dir)
    selected={x.strip() for x in args.datasets.split(",") if x.strip()}
    if selected: test_sets={k:v for k,v in test_sets.items() if k in selected}
    log("\n"+"="*78); log("WYSIWYR STAGE4 - REVIEWER-GRADE PROMPT-FREE REPAIR"); log("="*78)
    log(f"GPU: {torch.cuda.get_device_name(0)}"); log(f"split: {len(train_pairs)} train / {len(val_pairs)} val")
    log(f"test sets: {[(k,len(s3.pair_by_stem(*v))) for k,v in test_sets.items()]}")
    log("Loading trained Stage-3 MedSAM checkpoints (NO RETRAINING) ...")
    sam_b=s3.load_sam_state(base_ckpt,b_ckpt,medsam_repo,device); sam_a=s3.load_sam_state(base_ckpt,a_ckpt,medsam_repo,device)
    oracle={"skipped":True}
    if not args.skip_oracle:
        oracle=run_oracle(test_sets,sam_b,sam_a,device,out_root)
        if args.oracle_only:
            log("\nORACLE-ONLY COMPLETE"); return
        if max(float(oracle["macro_dice_baseline"]),float(oracle["macro_dice_abloss"])) < args.oracle_fail_threshold:
            raise SystemExit("Oracle diagnostic is too low. Stop before changing prompt logic; inspect checkpoint/data protocol first.")
    prop_ckpt=train_proposer(train_pairs,val_pairs,out_root,device,args.seed,args.proposal_size,args.proposal_epochs,
                             args.proposal_batch_size,args.num_workers,args.proposal_lr,args.proposal_patience,args.amp)
    proposer,prop_size=load_proposer(prop_ckpt,device)
    prompt_cfg=calibrate_proposal(proposer,val_pairs,prop_size,device,out_root,args.amp)
    usr_cfg=core.USRConfig(num_passes=8,aggregation_mode="soft",normalization_mode="none",scale_spatial_params=True,seed=args.seed)
    protocol={
        "stage":"Stage4 prompt-free repair","seed":args.seed,"train_n":len(train_pairs),"val_n":len(val_pairs),
        "oracle":{"diagnostic_only":True,"used_for_main_prompt":False,**oracle},
        "proposal":{"architecture":"SmallUNet","input_size":prop_size,"training_data":"fixed 870-image train split only",
                    "selection_data":"fixed 580-image validation split only","checkpoint":str(prop_ckpt),**prompt_cfg},
        "main_test_prompt":{"gt_used":False,"algorithm":"image -> prompt-free U-Net probability -> largest component -> validation-calibrated normalized-margin box -> MedSAM",
                            "same_box_for_baseline_and_abloss":True},
        "usr":{"T":8,"aggregation":"soft probability mean","normalization":"none","spatial_parameters_scaled_by_resolution":True,
               "perturbation_center":"same prediction-derived proposal box"},
        "test_sets":{k:len(s3.pair_by_stem(*v)) for k,v in test_sets.items()},
    }
    write_json(out_root/"protocol_stage4.json",protocol)
    log("\n"+"="*78); log("PHASE D/E - MAIN LEAKAGE-FREE 2x2 TEST + REVIEWER-2 ANALYSIS"); log("="*78)
    for ds,(img_dir,mask_dir) in test_sets.items():
        infer_promptfree_dataset(ds,img_dir,mask_dir,proposer,prop_size,prompt_cfg,sam_b,sam_a,device,out_root,usr_cfg,args.amp)
        summarize_proposal_test(ds,out_root)
        s3.run_stage2_analysis(ds,img_dir,mask_dir,out_root,script_dir,args.seed)
    write_json(out_root/"RUN_COMPLETE.json",{"completed_at":time.strftime("%Y-%m-%d %H:%M:%S"),
                                               "datasets":list(test_sets),"main_gt_prompt_used":False,
                                               "results_root":str(out_root/"reviewer2_results")})
    log("\n"+"="*78); log("STAGE4 COMPLETE"); log(f"Oracle diagnostic: {out_root/'oracle'/'oracle_summary.csv'}")
    log(f"Proposal calibration: {out_root/'proposal'/'validation_prompt_calibration_grid.csv'}")
    log(f"Main predictions: {out_root/'predictions'}"); log(f"Reviewer2 results: {out_root/'reviewer2_results'}")
    log(f"Protocol: {out_root/'protocol_stage4.json'}"); log("="*78)

if __name__ == "__main__":
    main()
