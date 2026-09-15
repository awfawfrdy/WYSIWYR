from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import math
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence

import cv2
import numpy as np

Array = np.ndarray


@dataclass(frozen=True)
class SEIGConfig:
    """Fixed discretization thresholds reported in Table 4.

    Note: the manuscript does not fully specify a numeric rule for the final
    evidence-quality state q nor an independent boundary-status threshold. This
    reconstruction makes those two mappings explicit and configurable.
    """

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

    # Reconstruction-only policy choices; easy to revise after reviewer feedback.
    allow_moderate_confidence_presence: bool = True
    cautious_boundary_at: str = "moderate"  # moderate/high => cautious
    include_numeric_evidence_in_prompt: bool = True


@dataclass(frozen=True)
class EvidenceVector:
    area_ratio: float
    centroid_x: float
    centroid_y: float
    compactness: float
    lesion_confidence: float
    lesion_uncertainty: float
    boundary_uncertainty: float
    valid_foreground: bool = True


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
    allowed: list[str] = field(default_factory=list)
    cautious: list[str] = field(default_factory=list)
    prohibited: list[str] = field(default_factory=list)


@dataclass
class SEIGResult:
    vector: EvidenceVector
    symbolic: SymbolicEvidence
    graph: Dict[str, Any]
    claim_plan: ClaimPermissionPlan
    prompt: str

    def as_dict(self) -> Dict[str, Any]:
        return {
            "vector": asdict(self.vector),
            "symbolic": asdict(self.symbolic),
            "graph": self.graph,
            "claim_plan": asdict(self.claim_plan),
            "prompt": self.prompt,
        }


class SEIG:
    """Structured Evidence Inference Graph as a deterministic expert rule engine."""

    # ---------------- Reviewer-grade checker lexicons ----------------
    # These patterns are deliberately explicit and deterministic so the full
    # rulebook can be reported and independently validated.
    _DIAGNOSIS_TERMS = re.compile(
        r"\b(malignan(?:t|cy)|benign|dysplasia|cancer(?:ous)?|carcinoma|adenoma|polyp(?:oid)?|"
        r"growth|inflammat\w*|histolog\w*|histopatholog\w*|patholog\w*|neoplasm\w*|"
        r"tumou?r\s+type|precancer\w*)\b",
        re.IGNORECASE,
    )
    _PROCEDURE_TERMS = re.compile(
        r"\b(biopsy|resection|endoscopic\s+resection|surgery|surgical\s+resection|"
        r"chemotherapy|radiotherapy|radiation\s+therapy|treatment(?:\s+plan|\s+decision|\s+planning)?|"
        r"therap\w*|clinical\s+management|follow[- ]?up(?:\s+(?:imaging|endoscopy|examination))?|"
        r"additional\s+imaging|imaging\s+(?:study|studies|technique|techniques|test|tests)|"
        r"diagnostic\s+(?:test|tests|procedure|procedures|workup)|histolog\w*\s+analysis|"
        r"histopatholog\w*\s+analysis|evaluation|assessment|investigation|"
        r"consult(?:ation)?(?:\s+with\s+(?:a\s+)?)?specialist|monitor(?:ing)?|surveillance)\b",
        re.IGNORECASE,
    )
    _RECOMMENDATION_CUES = re.compile(
        r"\b(recommend(?:ed|ation)?|advise(?:d|able)?|advisable|should|must|need(?:s|ed)?|necessary|"
        r"proceed\s+with|consider|warrant(?:s|ed)?|requires?|consult|further\s+(?:evaluat\w*|assess\w*|investigat\w*)|"
        r"additional\s+(?:evaluat\w*|imaging|diagnostic\s+(?:test|tests|procedure|procedures))|"
        r"may\s+be\s+necessary|might\s+be\s+necessary|would\s+be\s+necessary|"
        r"important\s+to\s+(?:continue\s+)?(?:monitor\w*|follow\w*|consult\w*|evaluat\w*|assess\w*|investigat\w*))\b",
        re.IGNORECASE,
    )
    _ANATOMY_TERMS = re.compile(
        r"\b(rectum|rectal|sigmoid(?:\s+colon)?|cecum|caecum|ascending\s+colon|descending\s+colon|"
        r"transverse\s+colon|hepatic\s+flexure|splenic\s+flexure|ileocecal\w*)\b",
        re.IGNORECASE,
    )

    # Safe epistemic disclaimers are not treated as positive diagnostic or
    # treatment claims. Importantly, statements such as "no indication of
    # malignancy" are NOT exempted, because they still make a diagnostic claim.
    _SAFE_EPISTEMIC_DISCLAIMER = re.compile(
        r"(?:"
        r"\b(?:diagnos\w*|histolog\w*|histopatholog\w*|patholog\w*|malignan\w*|benign|"
        r"treatment(?:\s+plan|\s+decision|\s+planning)?|clinical\s+management)\b"
        r".{0,100}\b(?:cannot|can\s+not|should\s+not|must\s+not|is\s+not\s+possible\s+to|"
        r"cannot\s+be|insufficient\s+to|unable\s+to)\b.{0,80}\b"
        r"(?:infer\w*|determin\w*|establish\w*|confirm\w*|conclud\w*|assess\w*|diagnos\w*|"
        r"classif\w*|recommend\w*|decid\w*)\b"
        r"|\b(?:cannot|can\s+not|should\s+not|must\s+not|is\s+not\s+possible\s+to|"
        r"insufficient\s+to|unable\s+to)\b.{0,80}\b"
        r"(?:infer\w*|determin\w*|establish\w*|confirm\w*|conclud\w*|assess\w*|diagnos\w*|"
        r"classif\w*|recommend\w*|decid\w*)\b.{0,100}\b"
        r"(?:diagnos\w*|histolog\w*|histopatholog\w*|patholog\w*|malignan\w*|benign|"
        r"treatment(?:\s+plan|\s+decision|\s+planning)?|clinical\s+management)\b"
        r"|\b(?:histological|histopathological|pathological)?\s*diagnos\w*\s+(?:should|must)\s+be\s+deferred\b"
        r"|\bthis\s+(?:report|description|observation)\s+is\s+not\s+(?:a\s+)?(?:pathological|histological)?\s*diagnos\w*\b"
        r")",
        re.IGNORECASE,
    )
    _SAFE_TREATMENT_DISCLAIMER = re.compile(
        r"(?:"
        r"\b(?:treatment(?:\s+plan|\s+decision|\s+planning)?|clinical\s+management|biopsy|resection|surgery)\b"
        r".{0,100}\b(?:cannot|can\s+not|should\s+not|must\s+not|is\s+not\s+possible\s+to|"
        r"cannot\s+be|insufficient\s+to|unable\s+to)\b.{0,80}\b(?:infer\w*|determin\w*|"
        r"establish\w*|recommend\w*|decid\w*|select\w*)\b"
        r"|\b(?:cannot|can\s+not|should\s+not|must\s+not|is\s+not\s+possible\s+to|"
        r"insufficient\s+to|unable\s+to)\b.{0,80}\b(?:infer\w*|determin\w*|establish\w*|"
        r"recommend\w*|decid\w*|select\w*)\b.{0,100}\b(?:treatment|management|biopsy|resection|surgery)\b"
        r")",
        re.IGNORECASE,
    )
    _SAFE_CLINICAL_ACTION_DISCLAIMER = re.compile(
        r"(?:"
        r"\b(?:biopsy|resection|surgery|treatment|management|evaluation|assessment|investigation|"
        r"imaging|follow[- ]?up|monitoring|surveillance|consultation)\b.{0,100}"
        r"\b(?:cannot|can\s+not|should\s+not|must\s+not|is\s+not\s+possible\s+to|"
        r"insufficient\s+to|unable\s+to)\b.{0,80}\b(?:recommend\w*|determin\w*|decid\w*|select\w*)\b"
        r"|\b(?:cannot|can\s+not|should\s+not|must\s+not|is\s+not\s+possible\s+to|"
        r"insufficient\s+to|unable\s+to)\b.{0,80}\b(?:recommend\w*|determin\w*|decid\w*|select\w*)\b"
        r".{0,100}\b(?:biopsy|resection|surgery|treatment|management|evaluation|assessment|investigation|"
        r"imaging|follow[- ]?up|monitoring|surveillance|consultation)\b"
        r")",
        re.IGNORECASE,
    )
    _SAFE_ANATOMY_DISCLAIMER = re.compile(
        r"\b(?:anatomical\s+subsite|rectum|rectal|sigmoid|cecum|caecum|ascending\s+colon|"
        r"descending\s+colon|transverse\s+colon|hepatic\s+flexure|splenic\s+flexure|ileocecal\w*)\b"
        r".{0,100}\b(?:cannot|can\s+not|not\s+possible|uncertain|unknown|unable)\b.{0,80}\b"
        r"(?:infer\w*|determin\w*|localiz\w*|confirm\w*|assign\w*)\b",
        re.IGNORECASE,
    )

    _BOUNDARY_TERMS = re.compile(r"\b(boundar(?:y|ies)|margin|contour|edge(?:s)?)\b", re.IGNORECASE)
    _BOUNDARY_DESCRIPTOR = re.compile(
        r"\b(irregular|regular|smooth|lobulated|spiculated|sharp|distinct|well[- ]defined|"
        r"ill[- ]defined|poorly[- ]defined|indistinct|clear(?:ly)?\s+defined)\b",
        re.IGNORECASE,
    )
    _CAUTION_TERMS = re.compile(
        r"\b(appears?|seems?|may|might|possibly|possible|partially|suggests?|suggesting|"
        r"cautious(?:ly)?|uncertain|uncertainty|approximately|approximate|not\s+clearly|"
        r"not\s+well[- ]defined|poorly[- ]defined|ill[- ]defined|indistinct|limited\s+confidence)\b",
        re.IGNORECASE,
    )
    _STRONG_CERTAINTY = re.compile(
        r"\b(clearly|definitely|certainly|unambiguously|well[- ]defined|sharp|distinct)\b",
        re.IGNORECASE,
    )
    _NEGATED_CERTAINTY = re.compile(
        r"\b(not|isn['’]?t|is\s+not|cannot\s+be)\s+(?:clearly|definitely|certainly|well[- ]defined|sharp|distinct)\b",
        re.IGNORECASE,
    )
    _DETERMINISTIC_COPULA = re.compile(
        r"\b(?:is|are|has|have|shows?|demonstrates?|exhibits?|with)\b",
        re.IGNORECASE,
    )

    def __init__(self, config: SEIGConfig | None = None) -> None:
        self.config = config or SEIGConfig()

    @staticmethod
    def _binary(mask: Array) -> Array:
        m = np.asarray(mask)
        if m.ndim > 2:
            m = np.squeeze(m)
        if m.ndim != 2:
            raise ValueError(f"Expected 2-D mask, got {m.shape}")
        return (m > 0).astype(np.uint8)

    @staticmethod
    def _float2d(x: Array) -> Array:
        a = np.asarray(x, dtype=np.float32)
        if a.ndim > 2:
            a = np.squeeze(a)
        if a.ndim != 2:
            raise ValueError(f"Expected 2-D map, got {a.shape}")
        return a

    def extract_evidence(self, mr: Array, p_bar: Array, uncertainty: Array) -> EvidenceVector:
        """Implement manuscript Eqs. (40)-(48)."""
        cfg = self.config
        m = self._binary(mr)
        p = self._float2d(p_bar)
        u = self._float2d(uncertainty)
        if not (m.shape == p.shape == u.shape):
            raise ValueError("mr, p_bar and uncertainty must have identical HxW shapes")
        h, w = m.shape
        area = float(m.sum())
        n = float(h * w)
        if area <= 0:
            return EvidenceVector(0.0, 0.5, 0.5, 0.0, 0.0, 1.0, 1.0, False)

        area_ratio = area / n
        lesion_conf = float((p * m).sum() / (area + cfg.eps))
        lesion_unc = float((u * m).sum() / (area + cfg.eps))

        ys, xs = np.nonzero(m)
        cx = float(xs.sum() / ((w - 1) * area + cfg.eps)) if w > 1 else 0.5
        cy = float(ys.sum() / ((h - 1) * area + cfg.eps)) if h > 1 else 0.5

        eroded = cv2.erode(m, np.ones((3, 3), np.uint8), iterations=1)
        boundary = ((m == 1) & (eroded == 0)).astype(np.uint8)
        boundary_len = float(boundary.sum())
        compactness = (boundary_len ** 2) / (4.0 * math.pi * area + cfg.eps)
        boundary_unc = float((u * boundary).sum() / (boundary_len + cfg.eps)) if boundary_len else 0.0

        return EvidenceVector(
            area_ratio=float(area_ratio),
            centroid_x=float(np.clip(cx, 0.0, 1.0)),
            centroid_y=float(np.clip(cy, 0.0, 1.0)),
            compactness=float(compactness),
            lesion_confidence=float(np.clip(lesion_conf, 0.0, 1.0)),
            lesion_uncertainty=float(np.clip(lesion_unc, 0.0, 1.0)),
            boundary_uncertainty=float(np.clip(boundary_unc, 0.0, 1.0)),
            valid_foreground=True,
        )

    def _area_label(self, x: float) -> str:
        c = self.config
        if x < c.area_tiny:
            return "tiny"
        if x < c.area_small:
            return "small"
        if x < c.area_medium:
            return "medium"
        return "large"

    def _confidence_label(self, x: float) -> str:
        c = self.config
        if x >= c.confidence_high:
            return "high"
        if x >= c.confidence_moderate:
            return "moderate"
        return "low"

    def _uncertainty_label(self, x: float) -> str:
        c = self.config
        if x < c.uncertainty_low:
            return "low"
        if x < c.uncertainty_high:
            return "moderate"
        return "high"

    def _shape_label(self, x: float) -> str:
        c = self.config
        if x < c.compactness_regular:
            return "regular / smooth"
        if x < c.compactness_irregular:
            return "mildly irregular / partially irregular"
        return "irregular"

    @staticmethod
    def _third(v: float, low: str, mid: str, high: str) -> str:
        if v < 1.0 / 3.0:
            return low
        if v < 2.0 / 3.0:
            return mid
        return high

    def _location_label(self, cx: float, cy: float) -> str:
        vert = self._third(cy, "upper", "middle", "lower")
        horiz = self._third(cx, "left", "center", "right")
        return f"{vert}-{horiz} field"

    def _quality_label(self, confidence: str, internal_unc: str, boundary_unc: str) -> str:
        """Explicit reconstruction of q, which is qualitative in the manuscript."""
        if confidence == "high" and internal_unc == "low":
            if boundary_unc == "low":
                return "reliable lesion presence and boundary evidence"
            return "reliable lesion presence but uncertain boundary"
        if confidence == "moderate" and internal_unc != "high":
            return "limited but usable lesion evidence"
        return "low-reliability lesion evidence"

    def symbolize(self, vector: EvidenceVector) -> SymbolicEvidence:
        if not vector.valid_foreground:
            return SymbolicEvidence(
                region="no valid lesion evidence",
                location="not available",
                area_level="not available",
                shape="not available",
                boundary_status="not available",
                confidence="low",
                internal_uncertainty="high",
                boundary_uncertainty="high",
                quality="invalid lesion evidence",
            )
        conf = self._confidence_label(vector.lesion_confidence)
        iu = self._uncertainty_label(vector.lesion_uncertainty)
        bu = self._uncertainty_label(vector.boundary_uncertainty)
        shape = self._shape_label(vector.compactness)
        # The paper's tuple contains a separate boundary-status field but does
        # not define independent thresholds. We use the compactness-derived
        # morphology label as the boundary-shape status, then calibrate wording
        # with boundary uncertainty.
        boundary_status = (
            "smooth/regular" if shape == "regular / smooth"
            else "partially irregular" if shape.startswith("mildly")
            else "irregular"
        )
        return SymbolicEvidence(
            region="main lesion",
            location=self._location_label(vector.centroid_x, vector.centroid_y),
            area_level=self._area_label(vector.area_ratio),
            shape=shape,
            boundary_status=boundary_status,
            confidence=conf,
            internal_uncertainty=iu,
            boundary_uncertainty=bu,
            quality=self._quality_label(conf, iu, bu),
        )

    def build_graph(self, symbolic: SymbolicEvidence) -> Dict[str, Any]:
        nodes = {
            "location": symbolic.location,
            "area": symbolic.area_level,
            "shape": symbolic.shape,
            "boundary": symbolic.boundary_status,
            "confidence": symbolic.confidence,
            "internal_uncertainty": symbolic.internal_uncertainty,
            "boundary_uncertainty": symbolic.boundary_uncertainty,
            "quality": symbolic.quality,
        }
        edges = [
            {"type": "confidence-uncertainty", "from": "confidence", "to": "internal_uncertainty"},
            {"type": "boundary-uncertainty", "from": "boundary", "to": "boundary_uncertainty"},
            {"type": "shape-boundary", "from": "shape", "to": "boundary"},
            {"type": "area-location", "from": "area", "to": "location"},
            {"type": "quality-claim", "from": "quality", "to": "claim_permissions"},
        ]
        return {"nodes": nodes, "edges": edges}

    def claim_permissions(self, symbolic: SymbolicEvidence) -> ClaimPermissionPlan:
        p = ClaimPermissionPlan()
        p.prohibited.extend([
            "histological diagnosis",
            "malignancy grade",
            "benign/malignant conclusion",
            "definitive pathology type",
            "treatment decision",
            "fine-grained anatomical subsite without metadata",
        ])
        if symbolic.quality == "invalid lesion evidence":
            p.cautious.append("state that no valid lesion-specific segmentation evidence is available")
            p.prohibited.extend(["lesion presence", "lesion size", "lesion morphology", "lesion boundary status"])
            return p

        if symbolic.confidence == "low":
            p.cautious.append("lesion presence")
        else:
            p.allowed.append("lesion presence")
        p.allowed.extend(["field-of-view location", "approximate lesion extent", "morphology"])

        if symbolic.boundary_uncertainty in {"moderate", "high"}:
            p.cautious.append("boundary status")
        else:
            p.allowed.append("boundary status")
        if symbolic.internal_uncertainty in {"moderate", "high"} or symbolic.confidence != "high":
            p.cautious.append("confidence/uncertainty wording")
        else:
            p.allowed.append("confidence wording")
        p.allowed.append("conservative safety note")
        return p

    def render_prompt(
        self,
        vector: EvidenceVector,
        symbolic: SymbolicEvidence,
        graph: Dict[str, Any],
        plan: ClaimPermissionPlan,
    ) -> str:
        """Deterministic structured evidence prompt corresponding to Table 5."""
        numeric = ""
        if self.config.include_numeric_evidence_in_prompt:
            numeric = (
                "\nNUMERIC EVIDENCE (framework-internal, not clinical thresholds):\n"
                f"- lesion area ratio: {vector.area_ratio:.6f}\n"
                f"- normalized centroid: ({vector.centroid_x:.4f}, {vector.centroid_y:.4f})\n"
                f"- compactness: {vector.compactness:.4f}\n"
                f"- mean lesion confidence: {vector.lesion_confidence:.4f}\n"
                f"- mean lesion uncertainty: {vector.lesion_uncertainty:.4f}\n"
                f"- mean boundary uncertainty: {vector.boundary_uncertainty:.4f}\n"
            )

        tuple_text = json.dumps(asdict(symbolic), ensure_ascii=False)
        graph_text = json.dumps(graph, ensure_ascii=False)
        return f"""You are generating a conservative observational report for a colorectal endoscopic image.
Use the image together with the segmentation-derived evidence below. Do not infer unsupported pathology, malignancy, treatment decisions, or fine-grained anatomical subsites.

SYMBOLIC EVIDENCE TUPLE:
{tuple_text}
{numeric}
EVIDENCE INTERACTION GRAPH:
{graph_text}

CLAIM PERMISSION PLAN:
Allowed claims: {', '.join(plan.allowed) if plan.allowed else 'none'}.
Cautious claims: {', '.join(plan.cautious) if plan.cautious else 'none'}.
Prohibited claims: {', '.join(plan.prohibited)}.

WORDING RULES:
- When a claim is cautious, use language such as "appears", "may", "partially", "suggests", or "should be interpreted cautiously".
- When boundary uncertainty is moderate or high, do not use a deterministic boundary statement.
- Use field-of-view location only; do not name a specific colorectal anatomical subsite unless external metadata is explicitly provided.
- The output is an observational description, not a pathological diagnosis or treatment recommendation.

OUTPUT FORMAT — exactly five sections:
1. Visual finding
2. Location and approximate size
3. Morphology and boundary
4. Confidence and uncertainty
5. Evidence-supported safety note
""".strip()

    def build(self, mr: Array, p_bar: Array, uncertainty: Array) -> SEIGResult:
        vector = self.extract_evidence(mr, p_bar, uncertainty)
        symbolic = self.symbolize(vector)
        graph = self.build_graph(symbolic)
        plan = self.claim_permissions(symbolic)
        prompt = self.render_prompt(vector, symbolic, graph, plan)
        return SEIGResult(vector, symbolic, graph, plan, prompt)

    @staticmethod
    def _normalize_heading_text(s: str) -> str:
        x = re.sub(r"^\s{0,3}#{1,6}\s*", "", s.strip())
        x = re.sub(r"^\s*(?:[-*+]\s+)", "", x)
        x = x.strip().strip("*_` ")
        return x

    @classmethod
    def _split_claims(cls, text: str) -> List[str]:
        """Deterministic, Markdown-aware sentence decomposition.

        Each non-empty line is first checked as a section/report heading. Other
        lines are split at sentence-final punctuation. This avoids fragmenting
        headings such as ``#### 3. Morphology and Boundary`` into pseudo-claims.
        """
        out: list[str] = []
        for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
            line = line.strip()
            if not line:
                continue
            # Ignore standalone list/section markers such as "1." or "4.".
            # They are formatting artifacts, not semantic claims.
            if re.fullmatch(r"\d+[.)]?", line):
                continue
            if cls._is_section_heading(line):
                out.append(line)
                continue
            parts = re.split(r"(?<=[.!?])\s+", line)
            for p in parts:
                p = p.strip()
                if not p or re.fullmatch(r"\d+[.)]?", p):
                    continue
                out.append(p)
        return out

    @classmethod
    def _is_section_heading(cls, s: str) -> bool:
        x = cls._normalize_heading_text(s)
        if re.match(r"^observational\s+report\s*:??$", x, re.I):
            return True
        # Accept both numbered and unnumbered versions, with optional Markdown.
        return bool(re.match(
            r"^(?:\d+[.)]\s*)?(visual finding|location and approximate size|morphology and boundary|"
            r"confidence and uncertainty|evidence-supported safety note)\s*:??$", x, re.I
        ))

    @classmethod
    def _is_safe_diagnostic_disclaimer(cls, claim: str) -> bool:
        return bool(cls._SAFE_EPISTEMIC_DISCLAIMER.search(claim))

    @classmethod
    def _is_safe_treatment_disclaimer(cls, claim: str) -> bool:
        return bool(
            cls._SAFE_TREATMENT_DISCLAIMER.search(claim)
            or cls._SAFE_CLINICAL_ACTION_DISCLAIMER.search(claim)
            or cls._SAFE_EPISTEMIC_DISCLAIMER.search(claim)
        )

    @classmethod
    def _is_safe_anatomy_disclaimer(cls, claim: str) -> bool:
        return bool(cls._SAFE_ANATOMY_DISCLAIMER.search(claim))

    @classmethod
    def _boundary_needs_calibration(cls, claim: str) -> bool:
        """Return True only for an assertive boundary claim lacking caution.

        This is intentionally idempotent: a checker-generated cautious sentence
        must not trigger the checker again.
        """
        if not cls._BOUNDARY_TERMS.search(claim):
            return False
        # Pure uncertainty statements are evidence statements, not deterministic
        # boundary-shape claims.
        if not cls._BOUNDARY_DESCRIPTOR.search(claim):
            return False
        negated_certainty = bool(cls._NEGATED_CERTAINTY.search(claim))
        cautious = bool(cls._CAUTION_TERMS.search(claim)) or negated_certainty
        strong = bool(cls._STRONG_CERTAINTY.search(claim)) and not negated_certainty
        deterministic = bool(cls._DETERMINISTIC_COPULA.search(claim))
        return strong or (deterministic and not cautious)

    def _rewrite_boundary(self, claim: str, symbolic: SymbolicEvidence) -> str:
        return (
            f"The lesion morphology is described as {symbolic.shape}; the boundary appears "
            f"{symbolic.boundary_status} and should be interpreted cautiously because boundary "
            f"uncertainty is {symbolic.boundary_uncertainty}."
        )

    @staticmethod
    def checker_rulebook() -> Dict[str, Any]:
        """Machine-readable priority and action policy for reproducibility."""
        return {
            "version": "3.0-frozen-reviewer-grade",
            "claim_decomposition": "Markdown-aware line parsing, numeric-only marker removal, then sentence-final punctuation splitting",
            "priority": [
                "R1 clinical/treatment recommendation",
                "R2 diagnostic/pathology claim",
                "R3 unsupported fine-grained anatomy",
                "R4 lesion-specific claim without valid lesion evidence",
                "R5 boundary calibration under moderate/high boundary uncertainty",
                "R0 supported/keep",
            ],
            "negation_policy": {
                "safe": [
                    "epistemic disclaimers such as 'malignancy cannot be inferred'",
                    "treatment-decision disclaimers such as 'treatment decisions cannot be inferred'",
                    "anatomy disclaimers that explicitly state a subsite cannot be determined",
                ],
                "not_safe": [
                    "negative diagnostic conclusions such as 'no indication of malignancy'",
                    "positive recommendations hedged with may/might/possibly",
                ],
            },
            "rewrite_templates": {
                "unsupported_anatomy": "replace fine-grained subsite with 'the observed endoscopic field'",
                "boundary_calibration": "The lesion morphology is described as <shape>; the boundary appears <status> and should be interpreted cautiously because boundary uncertainty is <level>.",
                "prohibited_diagnosis_or_treatment": "remove claim; append conservative safety note once",
            },
            "idempotence_requirement": "verify(R*) must introduce zero new rule violations and leave text unchanged",
            "freeze_policy": "v3 rules are frozen before independent human validation; the prior v2 400-claim audit sample is treated as development-only and excluded from the final validation sample",
        }

    def verify_report(
        self,
        report: str,
        result: SEIGResult,
        *,
        has_anatomical_metadata: bool = False,
        has_pathology_evidence: bool = False,
        has_treatment_evidence: bool = False,
        append_safety_note: bool = True,
    ) -> tuple[str, list[Dict[str, str]]]:
        """Deterministic claim-level verification with explicit priority.

        Audit status is one of {supported, calibrated, unsupported, prohibited}.
        Each non-heading claim receives exactly one highest-priority action and a
        stable rule_id. The policy is designed to be idempotent.
        """
        s = result.symbolic
        claims = self._split_claims(report)
        output: list[str] = []
        audit: list[Dict[str, str]] = []

        for claim in claims:
            original = claim
            if self._is_section_heading(claim):
                output.append(claim)
                audit.append({
                    "claim": original, "status": "supported", "action": "keep",
                    "output": claim, "rule_id": "H0", "category": "heading",
                    "reason": "recognized report/section heading",
                })
                continue

            status = "supported"
            action = "keep"
            rewritten = claim
            rule_id = "R0"
            category = "supported"
            reason = "no higher-priority rule triggered"

            # R1: explicit clinical/treatment recommendation. A pure epistemic
            # disclaimer is protected, but a hedged recommendation is not.
            proc = bool(self._PROCEDURE_TERMS.search(claim))
            recommendation = bool(self._RECOMMENDATION_CUES.search(claim))
            if (
                not has_treatment_evidence
                and proc and recommendation
                and not self._is_safe_treatment_disclaimer(claim)
            ):
                status, action, rewritten = "prohibited", "remove", ""
                rule_id, category = "R1", "clinical_recommendation"
                reason = "clinical/treatment recommendation is not supported by SEIG evidence"

            # R2: diagnostic/pathology claim. Safe epistemic disclaimers are
            # allowed; 'no indication of malignancy' remains a diagnostic claim.
            elif (
                not has_pathology_evidence
                and self._DIAGNOSIS_TERMS.search(claim)
                and not self._is_safe_diagnostic_disclaimer(claim)
            ):
                status, action, rewritten = "prohibited", "remove", ""
                rule_id, category = "R2", "diagnosis_pathology"
                reason = "diagnostic/pathology conclusion is not supported by image+SEIG evidence"

            # R3: fine-grained anatomy without metadata.
            elif (
                not has_anatomical_metadata
                and self._ANATOMY_TERMS.search(claim)
                and not self._is_safe_anatomy_disclaimer(claim)
            ):
                status, action = "unsupported", "generalize"
                rewritten = self._ANATOMY_TERMS.sub("the observed endoscopic field", claim)
                rule_id, category = "R3", "fine_grained_anatomy"
                reason = "fine-grained anatomical subsite requires external metadata"

            # R4: lesion-specific claim when no valid lesion evidence exists.
            elif not s.region.startswith("main lesion") and re.search(r"\blesion\b", claim, re.I):
                status, action, rewritten = "unsupported", "remove", ""
                rule_id, category = "R4", "lesion_without_evidence"
                reason = "no valid lesion-specific segmentation evidence"

            # R5: boundary wording calibration. Crucially, already-cautious or
            # negated wording is kept, making the checker idempotent.
            elif (
                s.boundary_uncertainty in {"moderate", "high"}
                and self._boundary_needs_calibration(claim)
            ):
                status, action = "calibrated", "rewrite_cautiously"
                rewritten = self._rewrite_boundary(claim, s)
                rule_id, category = "R5", "boundary_calibration"
                reason = f"boundary uncertainty is {s.boundary_uncertainty}; assertive boundary wording requires calibration"

            if rewritten:
                output.append(rewritten)
            audit.append({
                "claim": original,
                "status": status,
                "action": action,
                "output": rewritten,
                "rule_id": rule_id,
                "category": category,
                "reason": reason,
            })

        safety = (
            "Histological diagnosis, malignancy status, and treatment decisions cannot be inferred "
            "from this image-only, segmentation-grounded evidence."
        )
        # Detect the semantic safety concept rather than an exact case-sensitive
        # string so a second checker pass does not append duplicates.
        has_safety = any(
            self._is_safe_diagnostic_disclaimer(x) and ("treatment" in x.lower() or "management" in x.lower())
            for x in output
        )
        if append_safety_note and not has_safety:
            output.append(safety)
            audit.append({
                "claim": "<safety-note>", "status": "supported", "action": "append",
                "output": safety, "rule_id": "S0", "category": "safety_note",
                "reason": "append conservative non-diagnostic/non-treatment safety statement",
            })

        return "\n".join(output), audit

