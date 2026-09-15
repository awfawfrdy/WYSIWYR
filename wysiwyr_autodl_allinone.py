"""
WYSIWYR all-in-one reconstruction for AutoDL.

Contains:
  1) ABLoss: Ambiguity-aware Boundary Loss
  2) USR: Uncertainty-Driven Structured Rectifier
  3) SEIG: Structured Evidence Inference Graph
  4) Reviewer-oriented segmentation / uncertainty / USR audit metrics

Design notes for the major revision:
- ABLoss follows the manuscript equations and reported hyperparameters.
- USR supports BOTH prompt-driven and non-prompt backbones explicitly.
- For prompt-driven backbones, the default first coarse prompt is a full-image box,
  avoiding ground-truth prompt leakage and circular "mask -> box -> mask" initialization.
- For non-prompt backbones, box prompts are NOT fabricated. Instead, uncertainty is
  estimated by test-time transforms (TTA). This should be disclosed as a clarified /
  revised implementation in the manuscript and its experiments re-run.
- USR can aggregate soft probabilities (recommended for revision) or binary votes
  (legacy manuscript reproduction). Soft aggregation avoids the T=8 nine-value issue.
- USR spatial hyperparameters can be scaled with image resolution using 224 as the
  reference resolution.
- SEIG rules, lexicons, claim decomposition, negation handling, rule priority, and
  rewrite behavior are made explicit for reproducibility.

This file intentionally does NOT embed MedSAM weights or a specific dataset loader.
Pass lightweight predictor callables from your existing training/inference code.
"""


from __future__ import annotations

import os
DATA_ROOT = os.environ.get("WYSIWYR_DATA_ROOT", ".")

import argparse
import json
import math
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    import cv2
except Exception as e:  # pragma: no cover
    cv2 = None

try:
    from scipy import ndimage as ndi
    from scipy.spatial.distance import cdist
except Exception:  # pragma: no cover
    ndi = None
    cdist = None

try:
    from sklearn.metrics import roc_auc_score
except Exception:  # pragma: no cover
    roc_auc_score = None

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception:  # pragma: no cover
    torch = None
    nn = object
    F = None

Array = np.ndarray
Box = Tuple[int, int, int, int]  # xmin, ymin, xmax, ymax inclusive


# ============================================================================
# 1) ABLoss
# ============================================================================

@dataclass(frozen=True)
class ABLossConfig:
    boundary_kernel_size: int = 7
    lambda_fp: float = 0.8
    lambda_fn: float = 0.3
    alpha: float = 2.0
    beta: float = 1.0
    gamma: float = 1.0
    delta: float = 0.3
    eps: float = 1e-6
    dice_smooth: float = 1e-6


class ABLoss(nn.Module if torch is not None else object):
    """Ambiguity-aware Boundary Loss.

    L = Dice + BCE + lambda_fp * L_fp + lambda_fn * L_fn

    Expected logits/target: [B,1,H,W] or [B,H,W].
    """

    def __init__(self, config: Optional[ABLossConfig] = None):
        if torch is None:
            raise ImportError("PyTorch is required for ABLoss")
        super().__init__()
        self.cfg = config or ABLossConfig()
        k = self.cfg.boundary_kernel_size
        if k <= 0 or k % 2 == 0:
            raise ValueError("boundary_kernel_size must be a positive odd integer")

    @staticmethod
    def _as_4d(x: "torch.Tensor") -> "torch.Tensor":
        if x.ndim == 3:
            x = x.unsqueeze(1)
        if x.ndim != 4 or x.shape[1] != 1:
            raise ValueError(f"Expected [B,1,H,W] or [B,H,W], got {tuple(x.shape)}")
        return x

    @staticmethod
    def _dilate(x: "torch.Tensor", k: int) -> "torch.Tensor":
        return F.max_pool2d(x, kernel_size=k, stride=1, padding=k // 2)

    @staticmethod
    def _erode(x: "torch.Tensor", k: int) -> "torch.Tensor":
        return 1.0 - F.max_pool2d(1.0 - x, kernel_size=k, stride=1, padding=k // 2)

    def boundary_bands(self, target: "torch.Tensor"):
        y = self._as_4d(target.float()).clamp(0.0, 1.0)
        k = self.cfg.boundary_kernel_size
        by = (self._dilate(y, k) - self._erode(y, k)).clamp(0.0, 1.0)
        b_out = by * (1.0 - y)
        b_in = by * y
        return by, b_out, b_in

    def dice_loss(self, p: "torch.Tensor", y: "torch.Tensor") -> "torch.Tensor":
        dims = (1, 2, 3)
        inter = (p * y).sum(dims)
        denom = p.sum(dims) + y.sum(dims)
        dice = (2.0 * inter + self.cfg.dice_smooth) / (denom + self.cfg.dice_smooth)
        return 1.0 - dice.mean()

    def forward(self, logits, target, return_components: bool = False):
        z = self._as_4d(logits)
        y = self._as_4d(target.float()).clamp(0.0, 1.0)
        if z.shape != y.shape:
            raise ValueError(f"Shape mismatch logits={tuple(z.shape)} target={tuple(y.shape)}")

        p = torch.sigmoid(z)
        _, b_out, b_in = self.boundary_bands(y)

        # manuscript Eq. (6), detached when used as a weight
        up = (4.0 * p * (1.0 - p)).detach()
        w_fp = 1.0 + self.cfg.alpha * b_out + self.cfg.beta * up
        w_fn = 1.0 + self.cfg.gamma * b_in + self.cfg.delta * up

        l_fp = ((1.0 - y) * p * w_fp).mean()
        l_fn = (y * (1.0 - p) * w_fn).mean()
        l_dice = self.dice_loss(p, y)
        l_bce = F.binary_cross_entropy_with_logits(z, y)
        total = l_dice + l_bce + self.cfg.lambda_fp * l_fp + self.cfg.lambda_fn * l_fn

        if not return_components:
            return total
        return total, {
            "total": float(total.detach().cpu()),
            "dice": float(l_dice.detach().cpu()),
            "bce": float(l_bce.detach().cpu()),
            "fp": float(l_fp.detach().cpu()),
            "fn": float(l_fn.detach().cpu()),
            "mean_uncertainty_proxy": float(up.mean().detach().cpu()),
        }


# ============================================================================
# 2) USR
# ============================================================================

@dataclass(frozen=True)
class USRConfig:
    # manuscript defaults
    num_passes: int = 8
    jitter_px_ref: int = 10
    vote_threshold: float = 0.5
    inner_kernel_ref: int = 3
    outer_kernel_ref: int = 9
    prune_prob_threshold: float = 0.50
    prune_uncertainty_threshold: float = 0.35
    local_support_threshold: float = 0.80
    repair_prob_threshold: float = 0.01
    max_area_drop: float = 0.35
    max_area_rise: float = 0.25
    max_component_increase: int = 1
    max_hole_increase: int = 1
    min_lcc_ratio: float = 0.50
    min_object_area_ref: int = 100
    max_hole_area_ref: int = 100
    eps: float = 1e-8
    seed: int = 2026

    # revision-oriented controls
    reference_resolution: int = 224
    scale_spatial_params: bool = True
    aggregation_mode: str = "soft"       # "soft" recommended; "binary_vote" legacy
    normalization_mode: str = "none"     # "none" recommended; "per_image_minmax" legacy
    initial_prompt_mode: str = "full_image_box"  # prompt-driven model initialization
    binary_threshold_per_pass: float = 0.5


@dataclass
class USRResult:
    rectified_mask: Array
    aggregated_probability: Array
    uncertainty: Array
    initial_mask: Array
    coarse_mask: Optional[Array] = None
    prediction_box: Optional[Box] = None
    perturbed_boxes: List[Box] = field(default_factory=list)
    prune_candidates: Optional[Array] = None
    repair_candidates: Optional[Array] = None
    prune_accepted: bool = False
    repair_accepted: bool = False
    diagnostics: Dict[str, Any] = field(default_factory=dict)


class USR:
    """Uncertainty-Driven Structured Rectifier.

    Predictor signatures:
      prompt predictor:     predictor(image, box) -> HxW probability map in [0,1]
      non-prompt predictor: predictor(image)      -> HxW probability map in [0,1]
    """

    def __init__(self, config: Optional[USRConfig] = None):
        self.cfg = config or USRConfig()
        if self.cfg.aggregation_mode not in {"soft", "binary_vote"}:
            raise ValueError("aggregation_mode must be 'soft' or 'binary_vote'")
        if self.cfg.normalization_mode not in {"none", "per_image_minmax"}:
            raise ValueError("normalization_mode must be 'none' or 'per_image_minmax'")

    @staticmethod
    def _require_cv():
        if cv2 is None or ndi is None:
            raise ImportError("USR requires opencv-python and scipy")

    @staticmethod
    def _float2d(x: Array) -> Array:
        a = np.asarray(x, dtype=np.float32)
        a = np.squeeze(a)
        if a.ndim != 2:
            raise ValueError(f"Expected HxW array, got {a.shape}")
        return a

    @staticmethod
    def _binary(x: Array, threshold: float = 0.5) -> Array:
        a = USR._float2d(x)
        return (a > threshold).astype(np.uint8)

    def _scale_len(self, base: int, h: int, w: int, force_odd: bool = False) -> int:
        if not self.cfg.scale_spatial_params:
            v = int(base)
        else:
            scale = min(h, w) / float(self.cfg.reference_resolution)
            v = max(1, int(round(base * scale)))
        if force_odd and v % 2 == 0:
            v += 1
        return max(1, v)

    def _scale_area(self, base: int, h: int, w: int) -> int:
        if not self.cfg.scale_spatial_params:
            return int(base)
        ref = float(self.cfg.reference_resolution ** 2)
        return max(1, int(round(base * (h * w) / ref)))

    @staticmethod
    def box_from_mask(mask: Array) -> Optional[Box]:
        m = USR._binary(mask)
        ys, xs = np.nonzero(m)
        if xs.size == 0:
            return None
        return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())

    @staticmethod
    def _clip_box(box: Box, h: int, w: int) -> Box:
        x0, y0, x1, y1 = box
        x0 = int(np.clip(x0, 0, w - 1)); x1 = int(np.clip(x1, 0, w - 1))
        y0 = int(np.clip(y0, 0, h - 1)); y1 = int(np.clip(y1, 0, h - 1))
        if x1 < x0: x0, x1 = x1, x0
        if y1 < y0: y0, y1 = y1, y0
        return x0, y0, x1, y1

    def jitter_box(self, box: Box, h: int, w: int, rng: np.random.Generator) -> Box:
        j = self._scale_len(self.cfg.jitter_px_ref, h, w, force_odd=False)
        d = rng.integers(-j, j + 1, size=4)
        return self._clip_box((box[0] + int(d[0]), box[1] + int(d[1]),
                               box[2] + int(d[2]), box[3] + int(d[3])), h, w)

    @staticmethod
    def bernoulli_entropy(p: Array, eps: float = 1e-8) -> Array:
        p = np.asarray(p, dtype=np.float32)
        q = np.clip(p, eps, 1.0 - eps)
        u = -(p * np.log2(q) + (1.0 - p) * np.log2(np.clip(1.0 - p, eps, 1.0)))
        return np.clip(u, 0.0, 1.0).astype(np.float32)

    def aggregate(self, predictions: Sequence[Array]) -> Tuple[Array, Array, Array]:
        if len(predictions) == 0:
            raise ValueError("No predictions supplied to USR")
        probs = [np.clip(self._float2d(x), 0.0, 1.0) for x in predictions]
        if self.cfg.aggregation_mode == "binary_vote":
            stack = np.stack([(x > self.cfg.binary_threshold_per_pass).astype(np.float32) for x in probs])
        else:
            stack = np.stack(probs).astype(np.float32)
        p_bar = stack.mean(axis=0).astype(np.float32)
        m0 = (p_bar > self.cfg.vote_threshold).astype(np.uint8)
        u = self.bernoulli_entropy(p_bar, self.cfg.eps)
        return p_bar, m0, u

    def _normalize_if_requested(self, x: Array) -> Array:
        a = self._float2d(x)
        if self.cfg.normalization_mode == "none":
            return a
        mn, mx = float(a.min()), float(a.max())
        return ((a - mn) / (mx - mn + self.cfg.eps)).astype(np.float32)

    def prompt_uncertainty(
        self,
        image: Any,
        predictor: Callable[[Any, Box], Array],
        coarse_mask: Optional[Array] = None,
    ) -> Tuple[Array, Array, Array, Optional[Array], Optional[Box], List[Box]]:
        """Prompt-driven uncertainty.

        If coarse_mask is not supplied, use a full-image box ONCE to obtain M_init.
        Then derive the lesion box from M_init and perturb only that prediction-derived box.
        No ground-truth mask or GT box is used.
        """
        if coarse_mask is None:
            # infer H,W from image
            arr = np.asarray(image)
            if arr.ndim < 2:
                raise ValueError("Cannot infer image spatial size")
            h, w = arr.shape[:2]
            if self.cfg.initial_prompt_mode != "full_image_box":
                raise ValueError("Without coarse_mask, only initial_prompt_mode='full_image_box' is supported")
            full_box = (0, 0, w - 1, h - 1)
            coarse_prob = np.clip(self._float2d(predictor(image, full_box)), 0.0, 1.0)
            coarse_mask = (coarse_prob > self.cfg.vote_threshold).astype(np.uint8)
        else:
            coarse_mask = self._binary(coarse_mask, self.cfg.vote_threshold)
            h, w = coarse_mask.shape

        box = self.box_from_mask(coarse_mask)
        if box is None:
            z = np.zeros((h, w), dtype=np.float32)
            return z, z.astype(np.uint8), z, coarse_mask, None, []

        rng = np.random.default_rng(self.cfg.seed)
        boxes, preds = [], []
        for _ in range(self.cfg.num_passes):
            b = self.jitter_box(box, h, w, rng)
            boxes.append(b)
            preds.append(predictor(image, b))
        p_bar, m0, u = self.aggregate(preds)
        return p_bar, m0, u, coarse_mask, box, boxes

    # ---- non-prompt TTA ----------------------------------------------------

    @staticmethod
    def _apply_tta(image: Array, mode: int) -> Array:
        x = np.asarray(image)
        if mode % 4 == 0:
            return x.copy()
        if mode % 4 == 1:
            return np.flip(x, axis=1).copy()
        if mode % 4 == 2:
            return np.flip(x, axis=0).copy()
        return np.flip(np.flip(x, axis=0), axis=1).copy()

    @staticmethod
    def _invert_tta(prob: Array, mode: int) -> Array:
        p = np.asarray(prob)
        if mode % 4 == 0:
            return p.copy()
        if mode % 4 == 1:
            return np.flip(p, axis=1).copy()
        if mode % 4 == 2:
            return np.flip(p, axis=0).copy()
        return np.flip(np.flip(p, axis=0), axis=1).copy()

    def nonprompt_uncertainty(
        self,
        image: Array,
        predictor: Callable[[Array], Array],
    ) -> Tuple[Array, Array, Array, Array]:
        """Non-prompt backbone uncertainty via TTA.

        We do NOT pass fake bounding boxes to U-Net/DeepLab/etc. The predictor is
        called on image transforms, predictions are inverse-transformed, then aggregated.
        """
        preds = []
        for t in range(self.cfg.num_passes):
            x_t = self._apply_tta(image, t)
            p_t = np.clip(self._float2d(predictor(x_t)), 0.0, 1.0)
            preds.append(self._invert_tta(p_t, t))
        p_bar, m0, u = self.aggregate(preds)
        coarse = (np.clip(self._float2d(predictor(image)), 0.0, 1.0) > self.cfg.vote_threshold).astype(np.uint8)
        return p_bar, m0, u, coarse

    @staticmethod
    def _kernel(k: int) -> Array:
        return np.ones((k, k), dtype=np.uint8)

    def _boundary_bands(self, m0: Array) -> Tuple[Array, Array, Dict[str, int]]:
        self._require_cv()
        m = self._binary(m0)
        h, w = m.shape
        kin = self._scale_len(self.cfg.inner_kernel_ref, h, w, force_odd=True)
        kout = self._scale_len(self.cfg.outer_kernel_ref, h, w, force_odd=True)
        er = cv2.erode(m, self._kernel(kin), iterations=1)
        di = cv2.dilate(m, self._kernel(kout), iterations=1)
        b_in = ((m == 1) & (er == 0)).astype(np.uint8)
        b_out = ((di == 1) & (m == 0)).astype(np.uint8)
        return b_in, b_out, {"inner_kernel": kin, "outer_kernel": kout}

    @staticmethod
    def _local_support(mask: Array) -> Array:
        return cv2.blur(mask.astype(np.float32), (3, 3), borderType=cv2.BORDER_REPLICATE)

    @staticmethod
    def _component_count(mask: Array) -> int:
        n, _ = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
        return max(0, int(n) - 1)

    @staticmethod
    def _hole_count(mask: Array) -> int:
        m = mask.astype(bool)
        labels, n = ndi.label(~m)
        if n == 0:
            return 0
        border = set(np.unique(np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]])).tolist())
        return sum(1 for lab in range(1, n + 1) if lab not in border)

    @staticmethod
    def _largest_component_ratio(mask: Array, eps: float) -> float:
        m = mask.astype(np.uint8)
        area = int(m.sum())
        if area == 0:
            return 0.0
        n, _, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        if n <= 1:
            return 0.0
        return float(stats[1:, cv2.CC_STAT_AREA].max()) / (area + eps)

    def topology_guard(self, old: Array, new: Array) -> Tuple[bool, Dict[str, Any]]:
        old_m, new_m = self._binary(old), self._binary(new)
        old_area, new_area = float(old_m.sum()), float(new_m.sum())
        if old_area <= 0:
            return False, {"reason": "empty_old_mask"}
        drop = max(0.0, (old_area - new_area) / (old_area + self.cfg.eps))
        rise = max(0.0, (new_area - old_area) / (old_area + self.cfg.eps))
        dc = self._component_count(new_m) - self._component_count(old_m)
        dh = self._hole_count(new_m) - self._hole_count(old_m)
        lcc = self._largest_component_ratio(new_m, self.cfg.eps)
        ok = (drop <= self.cfg.max_area_drop and rise <= self.cfg.max_area_rise and
              dc <= self.cfg.max_component_increase and dh <= self.cfg.max_hole_increase and
              lcc >= self.cfg.min_lcc_ratio)
        return ok, {
            "area_drop": drop, "area_rise": rise,
            "component_delta": dc, "hole_delta": dh,
            "largest_component_ratio": lcc,
        }

    def _postprocess(self, mask: Array) -> Tuple[Array, Dict[str, int]]:
        m = self._binary(mask)
        h, w = m.shape
        min_area = self._scale_area(self.cfg.min_object_area_ref, h, w)
        max_hole = self._scale_area(self.cfg.max_hole_area_ref, h, w)

        # remove small foreground components
        n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        out = np.zeros_like(m)
        for lab in range(1, n):
            if int(stats[lab, cv2.CC_STAT_AREA]) >= min_area:
                out[labels == lab] = 1

        # fill small holes only
        bg_labels, nbg = ndi.label(~out.astype(bool))
        border = set(np.unique(np.concatenate([bg_labels[0], bg_labels[-1], bg_labels[:, 0], bg_labels[:, -1]])).tolist())
        for lab in range(1, nbg + 1):
            if lab in border:
                continue
            hole = (bg_labels == lab)
            if int(hole.sum()) <= max_hole:
                out[hole] = 1
        return out.astype(np.uint8), {"min_object_area": min_area, "max_hole_area": max_hole}

    def rectify(self, p_bar: Array, uncertainty: Array, m0: Optional[Array] = None) -> USRResult:
        self._require_cv()
        p = np.clip(self._float2d(p_bar), 0.0, 1.0)
        u = np.clip(self._float2d(uncertainty), 0.0, 1.0)
        m = self._binary(m0 if m0 is not None else p, self.cfg.vote_threshold)
        if not (p.shape == u.shape == m.shape):
            raise ValueError("p_bar, uncertainty and m0 must have identical shapes")

        if int(m.sum()) == 0:
            return USRResult(m.copy(), p, u, m.copy(), diagnostics={"empty_mask": True})

        p_used = self._normalize_if_requested(p)
        u_used = self._normalize_if_requested(u)
        b_in, b_out, scale_diag = self._boundary_bands(m)
        support = self._local_support(m)

        prune = ((m == 1) & (b_in == 1) &
                 (p_used < self.cfg.prune_prob_threshold) &
                 (u_used > self.cfg.prune_uncertainty_threshold) &
                 (support < self.cfg.local_support_threshold))

        repair = ((m == 0) & (b_out == 1) &
                  (p_used > self.cfg.repair_prob_threshold))

        m_minus = m.copy(); m_minus[prune] = 0
        ok_prune, d_prune = self.topology_guard(m, m_minus)
        m_p = m_minus if ok_prune else m.copy()

        m_plus = m_p.copy(); m_plus[repair] = 1
        ok_repair, d_repair = self.topology_guard(m_p, m_plus)
        m_c = m_plus if ok_repair else m_p

        m_r, post_diag = self._postprocess(m_c)
        diag = {
            "normalization_mode": self.cfg.normalization_mode,
            "aggregation_mode": self.cfg.aggregation_mode,
            "n_prune_candidates": int(prune.sum()),
            "n_repair_candidates": int(repair.sum()),
            "prune_guard": d_prune,
            "repair_guard": d_repair,
            **scale_diag,
            **post_diag,
        }
        return USRResult(
            rectified_mask=m_r,
            aggregated_probability=p,
            uncertainty=u,
            initial_mask=m,
            prune_candidates=prune.astype(np.uint8),
            repair_candidates=repair.astype(np.uint8),
            prune_accepted=ok_prune,
            repair_accepted=ok_repair,
            diagnostics=diag,
        )

    def run_prompt(self, image: Any, predictor: Callable[[Any, Box], Array], coarse_mask: Optional[Array] = None) -> USRResult:
        p, m0, u, coarse, box, boxes = self.prompt_uncertainty(image, predictor, coarse_mask)
        r = self.rectify(p, u, m0)
        r.coarse_mask = coarse
        r.prediction_box = box
        r.perturbed_boxes = boxes
        return r

    def run_nonprompt(self, image: Array, predictor: Callable[[Array], Array]) -> USRResult:
        p, m0, u, coarse = self.nonprompt_uncertainty(image, predictor)
        r = self.rectify(p, u, m0)
        r.coarse_mask = coarse
        return r


# ============================================================================
# 3) SEIG
# ============================================================================

@dataclass(frozen=True)
class SEIGConfig:
    # manuscript Table 4 defaults
    area_tiny: float = 0.01
    area_small: float = 0.05
    area_medium: float = 0.15
    confidence_high: float = 0.80
    confidence_moderate: float = 0.60
    uncertainty_low: float = 0.25
    uncertainty_high: float = 0.50
    compactness_regular: float = 1.30
    compactness_irregular: float = 1.80
    eps: float = 1e-8
    include_numeric_evidence: bool = True


@dataclass(frozen=True)
class EvidenceVector:
    area_ratio: float
    centroid_x: float
    centroid_y: float
    compactness: float
    lesion_confidence: float
    lesion_uncertainty: float
    boundary_uncertainty: float
    valid_foreground: bool


@dataclass(frozen=True)
class SymbolicEvidence:
    region: str
    location: str
    area_level: str
    shape: str
    boundary_status: str
    confidence: str
    internal_uncertainty: str
    boundary_uncertainty: str
    quality: str


@dataclass
class ClaimPermissionPlan:
    allowed: List[str] = field(default_factory=list)
    cautious: List[str] = field(default_factory=list)
    prohibited: List[str] = field(default_factory=list)


@dataclass
class VerifiedClaim:
    original: str
    status: str
    rewritten: str
    rule: str


class SEIG:
    """Deterministic, fully inspectable evidence-control layer."""

    # Explicit lexicons for reproducibility. Keep these versioned in the repo.
    PATHOLOGY_TERMS = {
        "malignant", "malignancy", "benign", "cancer", "carcinoma", "adenoma",
        "adenomatous", "dysplasia", "histology", "histological", "pathology",
        "pathological", "neoplasm", "neoplastic"
    }
    TREATMENT_TERMS = {
        "surgery", "surgical", "resection", "chemotherapy", "radiotherapy",
        "radiation", "treatment", "therapy", "biopsy", "polypectomy"
    }
    ANATOMY_TERMS = {
        "cecum", "caecum", "ascending colon", "transverse colon", "descending colon",
        "sigmoid", "rectum", "rectosigmoid", "hepatic flexure", "splenic flexure"
    }
    BOUNDARY_ASSERTIVE = {
        "clear boundary", "clearly defined boundary", "well-defined boundary",
        "sharp boundary", "definite boundary", "irregular boundary",
        "smooth boundary", "regular boundary"
    }
    NEGATION_CUES = {"no", "not", "without", "cannot", "can't", "unable", "unlikely", "neither", "nor"}

    # Priority: safety > anatomy > invalid evidence > boundary calibration > confidence calibration > supported
    RULE_PRIORITY = [
        "prohibited_pathology_treatment",
        "unsupported_anatomy",
        "invalid_lesion_evidence",
        "boundary_uncertainty_calibration",
        "low_confidence_calibration",
        "supported",
    ]

    def __init__(self, config: Optional[SEIGConfig] = None):
        self.cfg = config or SEIGConfig()

    @staticmethod
    def _binary(x: Array) -> Array:
        a = np.squeeze(np.asarray(x))
        if a.ndim != 2:
            raise ValueError("Expected HxW mask")
        return (a > 0.5).astype(np.uint8)

    @staticmethod
    def _float2d(x: Array) -> Array:
        a = np.squeeze(np.asarray(x, dtype=np.float32))
        if a.ndim != 2:
            raise ValueError("Expected HxW map")
        return a

    def extract(self, mr: Array, p_bar: Array, uncertainty: Array) -> EvidenceVector:
        if cv2 is None:
            raise ImportError("SEIG evidence extraction requires opencv-python")
        m = self._binary(mr)
        p = np.clip(self._float2d(p_bar), 0.0, 1.0)
        u = np.clip(self._float2d(uncertainty), 0.0, 1.0)
        if not (m.shape == p.shape == u.shape):
            raise ValueError("Mask/probability/uncertainty shapes differ")
        h, w = m.shape
        area = float(m.sum())
        if area <= 0:
            return EvidenceVector(0.0, 0.5, 0.5, 0.0, 0.0, 1.0, 1.0, False)

        ar = area / float(h * w)
        c = float((p * m).sum() / (area + self.cfg.eps))
        ul = float((u * m).sum() / (area + self.cfg.eps))
        ys, xs = np.nonzero(m)
        cx = float(xs.sum() / ((w - 1) * area + self.cfg.eps)) if w > 1 else 0.5
        cy = float(ys.sum() / ((h - 1) * area + self.cfg.eps)) if h > 1 else 0.5

        er = cv2.erode(m, np.ones((3, 3), np.uint8), iterations=1)
        boundary = ((m == 1) & (er == 0)).astype(np.uint8)
        perimeter = float(boundary.sum())
        comp = (perimeter ** 2) / (4.0 * math.pi * area + self.cfg.eps)
        ub = float((u * boundary).sum() / (perimeter + self.cfg.eps)) if perimeter > 0 else 0.0

        return EvidenceVector(ar, float(np.clip(cx, 0, 1)), float(np.clip(cy, 0, 1)),
                              comp, float(np.clip(c, 0, 1)), float(np.clip(ul, 0, 1)),
                              float(np.clip(ub, 0, 1)), True)

    def _area(self, x: float) -> str:
        if x < self.cfg.area_tiny: return "tiny"
        if x < self.cfg.area_small: return "small"
        if x < self.cfg.area_medium: return "medium"
        return "large"

    def _conf(self, x: float) -> str:
        if x >= self.cfg.confidence_high: return "high"
        if x >= self.cfg.confidence_moderate: return "moderate"
        return "low"

    def _unc(self, x: float) -> str:
        if x < self.cfg.uncertainty_low: return "low"
        if x < self.cfg.uncertainty_high: return "moderate"
        return "high"

    def _shape(self, x: float) -> str:
        if x < self.cfg.compactness_regular: return "regular / smooth"
        if x < self.cfg.compactness_irregular: return "mildly irregular / partially irregular"
        return "irregular"

    @staticmethod
    def _third(v: float, a: str, b: str, c: str) -> str:
        return a if v < 1/3 else b if v < 2/3 else c

    def symbolize(self, v: EvidenceVector) -> SymbolicEvidence:
        if not v.valid_foreground:
            return SymbolicEvidence("no valid lesion evidence", "not available", "not available",
                                    "not available", "not available", "low", "high", "high",
                                    "invalid lesion evidence")
        conf, iu, bu = self._conf(v.lesion_confidence), self._unc(v.lesion_uncertainty), self._unc(v.boundary_uncertainty)
        shape = self._shape(v.compactness)
        loc = f"{self._third(v.centroid_y,'upper','middle','lower')}-{self._third(v.centroid_x,'left','center','right')} field"
        bstatus = "smooth/regular" if shape == "regular / smooth" else "partially irregular" if shape.startswith("mildly") else "irregular"
        if conf == "high" and iu == "low" and bu == "low":
            q = "reliable lesion presence and boundary evidence"
        elif conf == "high" and iu == "low":
            q = "reliable lesion presence but uncertain boundary"
        elif conf == "moderate" and iu != "high":
            q = "limited but usable lesion evidence"
        else:
            q = "low-reliability lesion evidence"
        return SymbolicEvidence("main lesion", loc, self._area(v.area_ratio), shape, bstatus, conf, iu, bu, q)

    def graph(self, s: SymbolicEvidence) -> Dict[str, Any]:
        nodes = {
            "v_loc": s.location,
            "v_area": s.area_level,
            "v_shape": s.shape,
            "v_boundary": s.boundary_status,
            "v_conf": s.confidence,
            "v_int_unc": s.internal_uncertainty,
            "v_bnd_unc": s.boundary_uncertainty,
            "v_quality": s.quality,
        }
        edges = [
            {"id": "E_conf_unc", "source": "v_conf", "target": "v_int_unc", "action": "calibrate lesion-presence certainty"},
            {"id": "E_bnd_unc", "source": "v_boundary", "target": "v_bnd_unc", "action": "calibrate boundary wording"},
            {"id": "E_shape_bnd", "source": "v_shape", "target": "v_boundary", "action": "link morphology and boundary wording"},
            {"id": "E_area_loc", "source": "v_area", "target": "v_loc", "action": "coarse extent/location description"},
            {"id": "E_quality_claim", "source": "v_quality", "target": "claim_plan", "action": "control claim permission"},
        ]
        return {"nodes": nodes, "edges": edges}

    def permissions(self, s: SymbolicEvidence) -> ClaimPermissionPlan:
        p = ClaimPermissionPlan()
        p.prohibited = [
            "histological diagnosis", "malignancy grade", "benign/malignant conclusion",
            "definitive pathology type", "treatment decision",
            "fine-grained anatomical subsite without external metadata",
        ]
        if s.quality == "invalid lesion evidence":
            p.cautious = ["state no valid lesion-specific segmentation evidence"]
            p.prohibited += ["lesion presence", "lesion extent", "lesion morphology", "lesion boundary status"]
            return p
        (p.allowed if s.confidence != "low" else p.cautious).append("lesion presence")
        p.allowed += ["field-of-view location", "approximate lesion extent", "morphology", "conservative safety note"]
        (p.cautious if s.boundary_uncertainty in {"moderate", "high"} else p.allowed).append("boundary status")
        (p.cautious if (s.internal_uncertainty in {"moderate", "high"} or s.confidence != "high") else p.allowed).append("confidence wording")
        return p

    def prompt(self, v: EvidenceVector, s: SymbolicEvidence, g: Dict[str, Any], p: ClaimPermissionPlan) -> str:
        numeric = ""
        if self.cfg.include_numeric_evidence:
            numeric = (
                "\nNUMERIC EVIDENCE (framework thresholds; not diagnostic thresholds):\n"
                f"- area_ratio: {v.area_ratio:.6f}\n"
                f"- normalized_centroid: ({v.centroid_x:.4f}, {v.centroid_y:.4f})\n"
                f"- compactness: {v.compactness:.4f}\n"
                f"- lesion_confidence: {v.lesion_confidence:.4f}\n"
                f"- lesion_uncertainty: {v.lesion_uncertainty:.4f}\n"
                f"- boundary_uncertainty: {v.boundary_uncertainty:.4f}\n"
            )
        return f"""You are generating a conservative observational report for a colorectal endoscopic image.
Use the image together with the segmentation-derived evidence below.
Do not infer unsupported pathology, malignancy, treatment decisions, or fine-grained anatomical subsites.

SYMBOLIC EVIDENCE TUPLE:
{json.dumps(asdict(s), ensure_ascii=False)}
{numeric}
EVIDENCE INTERACTION GRAPH:
{json.dumps(g, ensure_ascii=False)}

CLAIM PERMISSION PLAN:
Allowed: {', '.join(p.allowed) if p.allowed else 'none'}
Cautious: {', '.join(p.cautious) if p.cautious else 'none'}
Prohibited: {', '.join(p.prohibited)}

WORDING RULES:
1. For cautious claims use terms such as appears, may, partially, suggests, or should be interpreted cautiously.
2. If boundary uncertainty is moderate/high, do not make deterministic boundary claims.
3. Use only field-of-view location; do not name a specific colorectal subsite without external metadata.
4. Do not make histological, malignant/benign, or treatment claims from the image alone.
5. Output exactly five sections: Visual finding; Location and approximate extent; Morphology and boundary; Confidence and uncertainty; Safety note.
"""

    @staticmethod
    def decompose_claims(report: str) -> List[str]:
        """Deterministic claim decomposition.

        Split on line breaks and sentence punctuation, but keep decimal numbers intact.
        """
        text = re.sub(r"\r\n?", "\n", report.strip())
        units = []
        for line in text.split("\n"):
            line = re.sub(r"^\s*[-*#]+\s*", "", line).strip()
            if not line:
                continue
            # punctuation split not between digits
            parts = re.split(r"(?<!\d)[.!?;]+(?!\d)", line)
            units.extend([p.strip(" :-\t") for p in parts if p.strip(" :-\t")])
        return units

    @classmethod
    def _has_term(cls, claim: str, terms: Iterable[str]) -> Optional[str]:
        low = claim.lower()
        for t in sorted(terms, key=len, reverse=True):
            if re.search(r"\b" + re.escape(t) + r"\b", low):
                return t
        return None

    @classmethod
    def _negated_near(cls, claim: str, term: str, window: int = 5) -> bool:
        """Simple transparent negation handling: cue within 5 tokens before term."""
        toks = re.findall(r"[a-zA-Z']+", claim.lower())
        tt = re.findall(r"[a-zA-Z']+", term.lower())
        if not tt:
            return False
        for i in range(len(toks) - len(tt) + 1):
            if toks[i:i+len(tt)] == tt:
                pre = toks[max(0, i-window):i]
                if any(cue in pre for cue in cls.NEGATION_CUES):
                    return True
        return False

    def verify_claim(self, claim: str, s: SymbolicEvidence, external_anatomy_metadata: Optional[str] = None) -> VerifiedClaim:
        c = claim.strip()
        if not c:
            return VerifiedClaim(claim, "unsupported", "", "empty")

        # Rule 1: prohibited pathology/treatment. Negated safety statements are retained.
        t = self._has_term(c, self.PATHOLOGY_TERMS | self.TREATMENT_TERMS)
        if t is not None:
            if self._negated_near(c, t) or "cannot be inferred" in c.lower() or "cannot infer" in c.lower():
                return VerifiedClaim(c, "supported", c, "negated_safety_statement")
            return VerifiedClaim(c, "prohibited", "", "prohibited_pathology_treatment")

        # Rule 2: unsupported anatomy.
        a = self._has_term(c, self.ANATOMY_TERMS)
        if a is not None and not external_anatomy_metadata:
            rewritten = "The lesion-like region is located within the visible endoscopic field."
            return VerifiedClaim(c, "calibrated", rewritten, "unsupported_anatomy")

        # Rule 3: invalid lesion evidence.
        lesion_words = re.search(r"\b(lesion|polyp|mass|region|boundary|shape|morpholog|size|extent)\w*\b", c.lower())
        if s.quality == "invalid lesion evidence" and lesion_words:
            return VerifiedClaim(c, "unsupported", "", "invalid_lesion_evidence")

        # Rule 4: boundary uncertainty calibration.
        if re.search(r"\bboundar\w*|margin\w*|edge\w*\b", c.lower()) and s.boundary_uncertainty in {"moderate", "high"}:
            low = c.lower()
            if not any(x in low for x in ["appears", "may", "partially", "suggests", "cautious", "uncertain"]):
                rewritten = c.rstrip(".") + "; the boundary should be interpreted cautiously because segmentation-derived boundary uncertainty is " + s.boundary_uncertainty + "."
                return VerifiedClaim(c, "calibrated", rewritten, "boundary_uncertainty_calibration")

        # Rule 5: low confidence calibration.
        if re.search(r"\b(lesion|polyp|region)\b", c.lower()) and s.confidence == "low":
            low = c.lower()
            if not any(x in low for x in ["appears", "possible", "may", "suggests"]):
                return VerifiedClaim(c, "calibrated", "A lesion-like region appears to be present, but confidence is low.", "low_confidence_calibration")

        return VerifiedClaim(c, "supported", c, "supported")

    def verify_report(self, report: str, s: SymbolicEvidence, external_anatomy_metadata: Optional[str] = None) -> Tuple[str, List[VerifiedClaim]]:
        claims = self.decompose_claims(report)
        checked = [self.verify_claim(c, s, external_anatomy_metadata) for c in claims]
        kept = [x.rewritten.strip() for x in checked if x.rewritten.strip()]
        # one conservative safety note, if none present
        joined_low = " ".join(kept).lower()
        if not any(k in joined_low for k in ["histological diagnosis cannot", "pathology cannot", "cannot be inferred from the image"]):
            kept.append("Histological diagnosis and treatment decisions cannot be inferred from the image alone.")
        return "\n".join(kept), checked

    def build(self, mr: Array, p_bar: Array, uncertainty: Array) -> Dict[str, Any]:
        v = self.extract(mr, p_bar, uncertainty)
        s = self.symbolize(v)
        g = self.graph(s)
        p = self.permissions(s)
        return {"vector": v, "symbolic": s, "graph": g, "permissions": p, "prompt": self.prompt(v, s, g, p)}


# ============================================================================
# 4) Metrics for Reviewer #2 experiments
# ============================================================================

def _bin(x: Array) -> Array:
    a = np.squeeze(np.asarray(x))
    if a.ndim != 2:
        raise ValueError("Expected HxW mask")
    return (a > 0.5).astype(np.uint8)


def confusion(pred: Array, gt: Array) -> Tuple[int, int, int, int]:
    p, y = _bin(pred), _bin(gt)
    tp = int(((p == 1) & (y == 1)).sum())
    fp = int(((p == 1) & (y == 0)).sum())
    fn = int(((p == 0) & (y == 1)).sum())
    tn = int(((p == 0) & (y == 0)).sum())
    return tp, fp, fn, tn


def dice_score(pred: Array, gt: Array, eps: float = 1e-8) -> float:
    tp, fp, fn, _ = confusion(pred, gt)
    return (2 * tp + eps) / (2 * tp + fp + fn + eps)


def iou_score(pred: Array, gt: Array, eps: float = 1e-8) -> float:
    tp, fp, fn, _ = confusion(pred, gt)
    return (tp + eps) / (tp + fp + fn + eps)


def precision_score_seg(pred: Array, gt: Array, eps: float = 1e-8) -> float:
    tp, fp, _, _ = confusion(pred, gt)
    return (tp + eps) / (tp + fp + eps)


def recall_score_seg(pred: Array, gt: Array, eps: float = 1e-8) -> float:
    tp, _, fn, _ = confusion(pred, gt)
    return (tp + eps) / (tp + fn + eps)


def boundary_map(mask: Array, kernel: int = 3) -> Array:
    if cv2 is None:
        raise ImportError("opencv-python required")
    m = _bin(mask)
    k = max(1, int(kernel)); k += (k % 2 == 0)
    er = cv2.erode(m, np.ones((k, k), np.uint8), iterations=1)
    return ((m == 1) & (er == 0)).astype(np.uint8)


def boundary_dice(pred: Array, gt: Array, tolerance: int = 2, eps: float = 1e-8) -> float:
    if cv2 is None:
        raise ImportError("opencv-python required")
    bp, bg = boundary_map(pred), boundary_map(gt)
    k = max(1, 2 * int(tolerance) + 1)
    dp = cv2.dilate(bp, np.ones((k, k), np.uint8), iterations=1)
    dg = cv2.dilate(bg, np.ones((k, k), np.uint8), iterations=1)
    matched_p = int((bp & dg).sum())
    matched_g = int((bg & dp).sum())
    denom = int(bp.sum() + bg.sum())
    return (matched_p + matched_g + eps) / (denom + eps)


def _boundary_points(mask: Array) -> Array:
    ys, xs = np.nonzero(boundary_map(mask))
    return np.stack([ys, xs], axis=1).astype(np.float32) if len(xs) else np.empty((0, 2), np.float32)


def hd95(pred: Array, gt: Array) -> float:
    if cdist is None:
        raise ImportError("scipy required")
    a, b = _boundary_points(pred), _boundary_points(gt)
    if len(a) == 0 and len(b) == 0: return 0.0
    if len(a) == 0 or len(b) == 0: return float("inf")
    d = cdist(a, b)
    da, db = d.min(axis=1), d.min(axis=0)
    return float(max(np.percentile(da, 95), np.percentile(db, 95)))


def assd(pred: Array, gt: Array) -> float:
    if cdist is None:
        raise ImportError("scipy required")
    a, b = _boundary_points(pred), _boundary_points(gt)
    if len(a) == 0 and len(b) == 0: return 0.0
    if len(a) == 0 or len(b) == 0: return float("inf")
    d = cdist(a, b)
    return float((d.min(axis=1).sum() + d.min(axis=0).sum()) / (len(a) + len(b)))


def segmentation_metrics(pred: Array, gt: Array) -> Dict[str, float]:
    return {
        "dice": dice_score(pred, gt),
        "iou": iou_score(pred, gt),
        "precision": precision_score_seg(pred, gt),
        "recall": recall_score_seg(pred, gt),
        "boundary_dice": boundary_dice(pred, gt),
        "hd95": hd95(pred, gt),
        "assd": assd(pred, gt),
    }


def expected_calibration_error(prob: Array, gt: Array, n_bins: int = 15) -> float:
    p = np.clip(np.asarray(prob, np.float32).ravel(), 0, 1)
    y = _bin(gt).ravel().astype(np.float32)
    edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i+1]
        sel = (p >= lo) & (p < hi if i < n_bins - 1 else p <= hi)
        if not np.any(sel): continue
        conf = float(p[sel].mean())
        acc = float(y[sel].mean())
        ece += float(sel.mean()) * abs(acc - conf)
    return float(ece)


def brier_score(prob: Array, gt: Array) -> float:
    p = np.clip(np.asarray(prob, np.float32), 0, 1)
    y = _bin(gt).astype(np.float32)
    return float(np.mean((p - y) ** 2))


def _roc_auc_numpy(y_true: Array, score: Array) -> float:
    """Pure-NumPy ROC-AUC with average ranks for ties.

    Equivalent to the Mann-Whitney U interpretation of ROC-AUC.
    This keeps the experiment runner usable on minimal AutoDL images
    where scikit-learn may not be installed.
    """
    y = np.asarray(y_true).astype(np.uint8).ravel()
    s = np.asarray(score, np.float64).ravel()
    valid = np.isfinite(s)
    y, s = y[valid], s[valid]
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(s, kind="mergesort")
    sorted_s = s[order]
    ranks = np.empty(len(s), dtype=np.float64)
    i = 0
    while i < len(s):
        j = i + 1
        while j < len(s) and sorted_s[j] == sorted_s[i]:
            j += 1
        # Ranks are 1-based. Average rank handles tied scores.
        avg_rank = ((i + 1) + j) / 2.0
        ranks[order[i:j]] = avg_rank
        i = j

    rank_sum_pos = float(ranks[y == 1].sum())
    u_stat = rank_sum_pos - n_pos * (n_pos + 1) / 2.0
    return float(u_stat / (n_pos * n_neg))


def error_detection_auroc(uncertainty: Array, pred_mask: Array, gt: Array) -> float:
    u = np.asarray(uncertainty, np.float32).ravel()
    err = (_bin(pred_mask) != _bin(gt)).astype(np.uint8).ravel()
    if len(np.unique(err)) < 2:
        return float("nan")
    if roc_auc_score is not None:
        return float(roc_auc_score(err, u))
    return _roc_auc_numpy(err, u)


def risk_coverage_curve(prob: Array, uncertainty: Array, gt: Array, points: int = 20) -> Dict[str, Any]:
    p = np.asarray(prob, np.float32).ravel()
    u = np.asarray(uncertainty, np.float32).ravel()
    y = _bin(gt).ravel().astype(np.uint8)
    pred = (p > 0.5).astype(np.uint8)
    order = np.argsort(u)  # retain low-uncertainty pixels first
    coverages = np.linspace(0.05, 1.0, points)
    risks = []
    n = len(order)
    for c in coverages:
        k = max(1, int(round(c * n)))
        idx = order[:k]
        risks.append(float(np.mean(pred[idx] != y[idx])))
    
    if hasattr(np, "trapezoid"):
        aurc = float(np.trapezoid(risks, coverages))
    else:
        aurc = float(np.trapz(risks, coverages))
    return {"coverage": coverages.tolist(), "risk": risks, "aurc": aurc}


def uncertainty_metrics(prob: Array, uncertainty: Array, gt: Array, pred_mask: Optional[Array] = None) -> Dict[str, Any]:
    pm = _bin(prob) if pred_mask is None else _bin(pred_mask)
    rc = risk_coverage_curve(prob, uncertainty, gt)
    return {
        "error_detection_auroc": error_detection_auroc(uncertainty, pm, gt),
        "brier": brier_score(prob, gt),
        "ece_15bin": expected_calibration_error(prob, gt, 15),
        "aurc": rc["aurc"],
        "risk_coverage": rc,
    }


def usr_correction_audit(before: Array, after: Array, gt: Array) -> Dict[str, float]:
    """Reviewer-requested semantic pixel accounting.

    added = after=1,before=0
      - correctly recovered lesion pixels: added & GT=1
      - incorrectly added pixels:          added & GT=0
    removed = after=0,before=1
      - correctly removed FP pixels:       removed & GT=0
      - incorrectly removed lesion pixels: removed & GT=1
    """
    b, a, y = _bin(before), _bin(after), _bin(gt)
    added = (a == 1) & (b == 0)
    removed = (a == 0) & (b == 1)
    rec = int((added & (y == 1)).sum())
    wrong_add = int((added & (y == 0)).sum())
    good_rm = int((removed & (y == 0)).sum())
    wrong_rm = int((removed & (y == 1)).sum())
    lesion_pixels = max(1, int((y == 1).sum()))
    bg_pixels = max(1, int((y == 0).sum()))
    return {
        "correctly_recovered_lesion_pixels": rec,
        "incorrectly_added_pixels": wrong_add,
        "correctly_removed_false_positive_pixels": good_rm,
        "incorrectly_removed_lesion_pixels": wrong_rm,
        "correctly_recovered_rate_vs_gt_lesion": rec / lesion_pixels,
        "incorrectly_removed_rate_vs_gt_lesion": wrong_rm / lesion_pixels,
        "incorrectly_added_rate_vs_gt_background": wrong_add / bg_pixels,
        "correctly_removed_fp_rate_vs_gt_background": good_rm / bg_pixels,
        "dice_before": dice_score(b, y),
        "dice_after": dice_score(a, y),
        "dice_worsened": float(dice_score(a, y) < dice_score(b, y)),
        "boundary_dice_before": boundary_dice(b, y),
        "boundary_dice_after": boundary_dice(a, y),
        "boundary_dice_worsened": float(boundary_dice(a, y) < boundary_dice(b, y)),
    }


def factorial_2x2_effects(metric_baseline: Sequence[float], metric_abloss: Sequence[float],
                          metric_usr: Sequence[float], metric_both: Sequence[float]) -> Dict[str, float]:
    """Descriptive 2x2 main and interaction effects for a metric where larger is better.

    A main effect = mean(A on) - mean(A off), averaging over B states.
    B main effect = mean(B on) - mean(B off), averaging over A states.
    interaction  = (both - A) - (B - baseline), using group means.
    """
    b = np.asarray(metric_baseline, float); a = np.asarray(metric_abloss, float)
    u = np.asarray(metric_usr, float); ab = np.asarray(metric_both, float)
    m = lambda x: float(np.nanmean(x))
    return {
        "mean_baseline": m(b), "mean_abloss": m(a), "mean_usr": m(u), "mean_abloss_usr": m(ab),
        "main_effect_abloss": 0.5 * ((m(a) - m(b)) + (m(ab) - m(u))),
        "main_effect_usr": 0.5 * ((m(u) - m(b)) + (m(ab) - m(a))),
        "interaction": (m(ab) - m(a)) - (m(u) - m(b)),
    }


# ============================================================================
# 5) Smoke test
# ============================================================================

def _smoke_test():
    print("[1/4] ABLoss")
    if torch is None:
        print("  skipped: PyTorch not installed")
    else:
        torch.manual_seed(1)
        logits = torch.randn(2, 1, 64, 64)
        gt = torch.zeros(2, 1, 64, 64)
        gt[:, :, 16:48, 18:46] = 1
        loss, parts = ABLoss()(logits, gt, return_components=True)
        print("  loss=", float(loss), parts)

    print("[2/4] USR")
    h = w = 128
    yy, xx = np.mgrid[:h, :w]
    gt_np = (((xx - 64) / 30) ** 2 + ((yy - 64) / 24) ** 2 <= 1).astype(np.uint8)
    p = np.exp(-((((xx - 64) / 34) ** 2 + ((yy - 64) / 28) ** 2))).astype(np.float32)
    p = np.clip(p, 0, 1)
    u = USR.bernoulli_entropy(p)
    r = USR().rectify(p, u, p > 0.5)
    print("  initial pixels=", int(r.initial_mask.sum()), "rectified=", int(r.rectified_mask.sum()))
    print("  diag=", r.diagnostics)

    print("[3/4] SEIG")
    seig = SEIG()
    pack = seig.build(r.rectified_mask, p, u)
    print("  symbolic=", asdict(pack["symbolic"]))
    demo_report = "The lesion is malignant. It has an irregular boundary. It is located in the sigmoid colon."
    final, checked = seig.verify_report(demo_report, pack["symbolic"])
    print("  checked report:\n", final)
    print("  rules=", [asdict(x) for x in checked])

    print("[4/4] Reviewer metrics")
    print("  seg=", segmentation_metrics(r.rectified_mask, gt_np))
    try:
        print("  unc=", {k: v for k, v in uncertainty_metrics(p, u, gt_np).items() if k != "risk_coverage"})
    except Exception as e:
        print("  uncertainty metrics skipped:", e)
    print("  audit=", usr_correction_audit(p > 0.5, r.rectified_mask, gt_np))
    print("SMOKE TEST PASSED")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke-test", action="store_true", help="run a dependency and function smoke test")
    ap.add_argument("--print-seig-rules", action="store_true", help="print deterministic SEIG rules/lexicons")
    args = ap.parse_args()

    if args.print_seig_rules:
        payload = {
            "rule_priority": SEIG.RULE_PRIORITY,
            "pathology_terms": sorted(SEIG.PATHOLOGY_TERMS),
            "treatment_terms": sorted(SEIG.TREATMENT_TERMS),
            "anatomy_terms": sorted(SEIG.ANATOMY_TERMS),
            "negation_cues": sorted(SEIG.NEGATION_CUES),
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    if args.smoke_test:
        _smoke_test()
    if not args.smoke_test and not args.print_seig_rules:
        ap.print_help()


if __name__ == "__main__":
    main()
