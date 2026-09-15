from __future__ import annotations
import os
DATA_ROOT = os.environ.get("WYSIWYR_DATA_ROOT", ".")

import argparse, csv, json, math, os, re
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List

import numpy as np
from seig import SEIG, SymbolicEvidence

VARIANT_ORDER = ["baseline", "abloss", "usr", "both", "gt"]
VARIANT_NAMES = {
    "baseline": "MedSAM",
    "abloss": "MedSAM+ABLoss",
    "usr": "MedSAM+USR",
    "both": "MedSAM+ABLoss+USR",
    "gt": "GT-mask oracle",
}


def audit_stats(audit: List[Dict[str, str]], report: str, prefix: str) -> Dict[str, Any]:
    status = defaultdict(int); action = defaultdict(int); rules = defaultdict(int)
    for a in audit:
        status[a.get("status", "unknown")] += 1
        action[a.get("action", "unknown")] += 1
        rules[a.get("rule_id", "unknown")] += 1
    violations = status["prohibited"] + status["unsupported"] + status["calibrated"]
    headings = [
        "visual finding", "location and approximate size", "morphology and boundary",
        "confidence and uncertainty", "evidence-supported safety note",
    ]
    hits = sum(bool(re.search(re.escape(h), report, flags=re.I)) for h in headings)
    return {
        f"{prefix}_violation_count": int(violations),
        f"{prefix}_forbidden_count": int(status["prohibited"]),
        f"{prefix}_unsupported_count": int(status["unsupported"]),
        f"{prefix}_calibrated_count": int(status["calibrated"]),
        f"{prefix}_anatomy_unsupported_count": int(action["generalize"]),
        f"{prefix}_boundary_overconfidence_count": int(action["rewrite_cautiously"]),
        f"{prefix}_removed_claim_count": int(action["remove"]),
        f"{prefix}_any_violation": int(violations > 0),
        f"{prefix}_section_heading_hits": int(hits),
        f"{prefix}_char_count": len(report),
        f"{prefix}_R1_count": int(rules["R1"]),
        f"{prefix}_R2_count": int(rules["R2"]),
        f"{prefix}_R3_count": int(rules["R3"]),
        f"{prefix}_R4_count": int(rules["R4"]),
        f"{prefix}_R5_count": int(rules["R5"]),
    }


def atomic_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys=[]; seen=set()
    for r in rows:
        for k in r:
            if k not in seen: seen.add(k); keys.append(k)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w=csv.DictWriter(f,fieldnames=keys); w.writeheader(); w.writerows(rows)


def read_csv(path: Path) -> List[Dict[str,str]]:
    if not path.exists(): return []
    with path.open(newline="",encoding="utf-8-sig") as f: return list(csv.DictReader(f))


def mean(xs: Iterable[float]) -> float:
    vals=[float(x) for x in xs if x is not None and str(x)!="" and not (isinstance(x,float) and math.isnan(x))]
    return float(np.mean(vals)) if vals else float("nan")


def summary(rows: List[Dict[str, Any]], group_keys: List[str]) -> List[Dict[str, Any]]:
    groups=defaultdict(list)
    for r in rows: groups[tuple(r[k] for k in group_keys)].append(r)
    metrics=[
        "dice","boundary_dice","hd95","assd","symbolic_match_fraction",
        "raw_v3_violation_count","raw_v3_forbidden_count","raw_v3_unsupported_count",
        "raw_v3_boundary_overconfidence_count","raw_v3_any_violation",
        "checked_v3_violation_count","checked_v3_forbidden_count","checked_v3_unsupported_count",
        "checked_v3_boundary_overconfidence_count","checked_v3_any_violation",
    ]
    out=[]
    for key,rs in sorted(groups.items(),key=lambda kv:kv[0]):
        o={k:v for k,v in zip(group_keys,key)}; o["n"]=len(rs)
        for m in metrics:
            if m in rs[0]: o[m]=mean(r.get(m) for r in rs)
        out.append(o)
    return out


def unit_tests(seig: SEIG) -> None:
    sym=SymbolicEvidence(
        region="main lesion", location="middle-center field", area_level="small",
        shape="irregular", boundary_status="irregular", confidence="high",
        internal_uncertainty="low", boundary_uncertainty="high",
        quality="reliable lesion presence but uncertain boundary",
    )
    result=SimpleNamespace(symbolic=sym)
    cases=[
        ("The lesion has an irregular boundary.","R5","rewrite_cautiously"),
        ("The lesion boundary appears irregular and should be interpreted cautiously because boundary uncertainty is high.","R0","keep"),
        ("The boundary is not clearly defined and may be difficult to delineate.","R0","keep"),
        ("Malignancy cannot be inferred from this image-only evidence.","R0","keep"),
        ("There is no indication of malignancy.","R2","remove"),
        ("The exact histological diagnosis should be deferred until definitive evidence is available.","R0","keep"),
        ("Biopsy is recommended for confirmation.","R1","remove"),
        ("It is advisable to conduct additional imaging or biopsy studies to ensure accurate diagnosis.","R1","remove"),
        ("Further evaluation by a medical professional is recommended for a definitive diagnosis.","R1","remove"),
        ("It is important to continue monitoring this area during subsequent endoscopic examinations.","R1","remove"),
        ("The lesion may be a polyp or a similar growth.","R2","remove"),
        ("The surface suggests possible inflammation.","R2","remove"),
        ("Polyp histology cannot be inferred from this image-only evidence.","R0","keep"),
        ("Treatment decisions cannot be inferred from this image-only evidence.","R0","keep"),
        ("The lesion is located in the sigmoid colon.","R3","generalize"),
    ]
    for text,rule,action in cases:
        checked,audit=seig.verify_report(text,result,append_safety_note=False)
        nonhead=[a for a in audit if a.get("category")!="heading"]
        assert nonhead,text
        a=nonhead[0]
        assert a["rule_id"]==rule,(text,a)
        assert a["action"]==action,(text,a)
    # Numeric-only list markers are formatting artifacts, not claims.
    checked,audit=seig.verify_report("1.\n4.\n5.",result,append_safety_note=False)
    assert checked.strip()=="",checked
    assert len(audit)==0,audit
    mixed="""1. Visual finding\nA lesion is visible.\n2. Location and approximate size\nThe lesion is located in the sigmoid colon.\n3. Morphology and boundary\nThe lesion has an irregular boundary.\n4. Confidence and uncertainty\nBoundary uncertainty is high.\n5. Evidence-supported safety note\nMalignancy cannot be inferred from this image-only evidence."""
    r1,_=seig.verify_report(mixed,result)
    r2,a2=seig.verify_report(r1,result,append_safety_note=False)
    s2=audit_stats(a2,r1,"x")
    assert r1==r2,(r1,r2)
    assert s2["x_violation_count"]==0,s2


def build_human_validation_sample(paths: List[Path], seig: SEIG, out: Path, dev_template: Path, n_total: int=400) -> None:
    # Any claims inspected during v2 debugging are development-only and excluded
    # from the final, blinded, independent human-validation set.
    dev=read_csv(dev_template)
    dev_keys={(r.get("dataset",""),r.get("case_id",""),r.get("variant_key",""),r.get("claim","").strip()) for r in dev}
    candidate=[]
    for p in paths:
        d=json.loads(p.read_text(encoding="utf-8"))
        sym=SymbolicEvidence(**d["seig"]["symbolic"]); result=SimpleNamespace(symbolic=sym)
        _,audit=seig.verify_report(d["raw_report_R0"],result,append_safety_note=False)
        for j,a in enumerate(audit):
            if a.get("category")=="heading": continue
            key=(str(d["dataset"]),str(d["case_id"]),str(d["variant_key"]),a["claim"].strip())
            if key in dev_keys: continue
            candidate.append({
                "dataset":d["dataset"],"case_id":d["case_id"],"variant_key":d["variant_key"],
                "claim_index":j,"claim":a["claim"],"checker_v3_status":a["status"],
                "checker_v3_rule_id":a["rule_id"],"checker_v3_action":a["action"],
            })
    rng=np.random.default_rng(2024)
    # Independent selection: stratify only by mask-source variant, never by
    # checker prediction/status/rule. 80 per variant = 400 total.
    per_variant=n_total//len(VARIANT_ORDER)
    selected=[]
    for vk in VARIANT_ORDER:
        pool=[x for x in candidate if x["variant_key"]==vk]
        n=min(per_variant,len(pool))
        idx=rng.choice(len(pool),size=n,replace=False)
        selected.extend(pool[int(i)] for i in idx)
    rng.shuffle(selected)
    blind=[]; keyrows=[]
    for i,x in enumerate(selected,1):
        blind.append({
            "annotation_id":i,"dataset":x["dataset"],"case_id":x["case_id"],"variant_key":x["variant_key"],
            "claim_index":x["claim_index"],"claim":x["claim"],
            "human_status":"","human_action":"","human_rule_id":"","annotator_id":"","notes":"",
        })
        keyrows.append({
            "annotation_id":i,"dataset":x["dataset"],"case_id":x["case_id"],"variant_key":x["variant_key"],
            "claim_index":x["claim_index"],"checker_v3_status":x["checker_v3_status"],
            "checker_v3_action":x["checker_v3_action"],"checker_v3_rule_id":x["checker_v3_rule_id"],
        })
    write_csv(out/"checker_v3_human_validation_BLINDED.csv",blind)
    write_csv(out/"checker_v3_human_validation_KEY_DO_NOT_SHOW_ANNOTATOR.csv",keyrows)
    meta={
        "seed":2024,"n":len(selected),"selection":"80 claims per mask-source variant; random within variant; independent of checker labels",
        "excluded_development_claims":len(dev_keys),"development_template":str(dev_template),
        "blinding":"checker predictions are stored only in KEY_DO_NOT_SHOW_ANNOTATOR.csv",
    }
    atomic_json(out/"human_validation_sampling_protocol.json",meta)


def main() -> None:
    ap=argparse.ArgumentParser()
    ap.add_argument("--source",default=DATA_ROOT + "/wysiwyr_real/stage6_seig_mllm_end2end")
    ap.add_argument("--output",default=DATA_ROOT + "/wysiwyr_real/stage6_checker_v3")
    ap.add_argument("--dev-template",default=DATA_ROOT + "/wysiwyr_real/stage6_checker_v2/checker_v2_human_validation_template.csv")
    args=ap.parse_args(); src=Path(args.source); out=Path(args.output); dev=Path(args.dev_template)
    if not src.exists(): raise SystemExit(f"Stage6 source not found: {src}")
    out.mkdir(parents=True,exist_ok=True)
    seig=SEIG(); unit_tests(seig)
    print("CHECKER V3 UNIT TESTS PASSED")
    atomic_json(out/"checker_rulebook_v3.json",seig.checker_rulebook())
    paths=sorted((src/"cases").glob("*/*/*.json"))
    if not paths: raise SystemExit(f"No Stage6 case JSON files under {src/'cases'}")
    print(f"Re-auditing {len(paths)} stored reports; NO MLLM inference will be run")
    rows=[]; idem_fail=[]
    for i,p in enumerate(paths,1):
        d=json.loads(p.read_text(encoding="utf-8")); sym=SymbolicEvidence(**d["seig"]["symbolic"]); result=SimpleNamespace(symbolic=sym)
        raw=d["raw_report_R0"]
        checked,audit=seig.verify_report(raw,result,append_safety_note=True); rawstats=audit_stats(audit,raw,"raw_v3")
        checked2,audit2=seig.verify_report(checked,result,append_safety_note=False); cstats=audit_stats(audit2,checked,"checked_v3")
        idempotent=int(checked2==checked and cstats["checked_v3_violation_count"]==0)
        if not idempotent:
            idem_fail.append({"dataset":d["dataset"],"case_id":d["case_id"],"variant_key":d["variant_key"],
                              "residual_violations":cstats["checked_v3_violation_count"],"text_equal":int(checked2==checked)})
        seg=d.get("segmentation_metrics",{}); sag=d.get("symbolic_agreement",{})
        row={"dataset":d["dataset"],"case_id":d["case_id"],"variant_key":d["variant_key"],
             "variant":d.get("variant",VARIANT_NAMES.get(d["variant_key"],d["variant_key"])),**seg,**sag,**rawstats,**cstats,
             "checker_v3_idempotent":idempotent}
        rows.append(row)
        rec={"dataset":d["dataset"],"case_id":d["case_id"],"variant_key":d["variant_key"],"variant":d.get("variant"),
             "raw_report_R0":raw,"checked_report_Rstar_v3":checked,"checker_audit_v3":audit,
             "checked_residual_audit_v3":audit2,"raw_v3_stats":rawstats,"checked_v3_residual_stats":cstats,"idempotent":bool(idempotent)}
        atomic_json(out/"cases"/d["dataset"]/str(d["case_id"])/f"{d['variant_key']}.json",rec)
        if i%250==0 or i==len(paths): print(f"  {i}/{len(paths)}")
    write_csv(out/"checker_v3_case_results.csv",rows)
    write_csv(out/"checker_v3_idempotence_failures.csv",idem_fail)
    write_csv(out/"checker_v3_summary_by_dataset_variant.csv",summary(rows,["dataset","variant_key","variant"]))
    write_csv(out/"checker_v3_summary_by_variant.csv",summary(rows,["variant_key","variant"]))
    build_human_validation_sample(paths,seig,out,dev,n_total=400)
    readme=f"""# Stage6 Checker v3 frozen re-audit\n\n- Source Stage6 reports: {src}\n- Stored reports re-audited: {len(rows)}\n- Qwen/MLLM re-generation: NO\n- Checker version: 3.0-frozen-reviewer-grade\n- Idempotence failures: {len(idem_fail)}\n\n## Why v3\nThe v2 development audit exposed residual false negatives for generic clinical-action recommendations (e.g., additional diagnostic testing/evaluation/monitoring) and lesion-type/pathology terms (e.g., polyp/growth/inflammation), plus numeric-only formatting fragments. v3 fixes those rule classes before independent validation.\n\n## Independence guardrail\nThe prior v2 400-claim sample is development-only and excluded from final validation. The v3 final 400-claim validation sample is selected without using checker predictions and is blinded: annotators receive only `checker_v3_human_validation_BLINDED.csv`; checker predictions remain in the separate key file.\n\n## Important\nChecker v3 is a deterministic rule-compliance evaluator, not an independent clinical-validity evaluator. Precision/recall/F1 must be reported against the blinded human gold labels before checker-derived compliance is used as external validation evidence.\n"""
    (out/"README_CHECKER_V3.md").write_text(readme,encoding="utf-8")
    print(f"CHECKER V3 RE-AUDIT COMPLETE: {out}")
    print(f"Idempotence failures: {len(idem_fail)}")
    if idem_fail: raise SystemExit("Checker v3 idempotence failed")

if __name__=="__main__": main()
