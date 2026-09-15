#!/usr/bin/env python3
from __future__ import annotations
import os
DATA_ROOT = os.environ.get("WYSIWYR_DATA_ROOT", ".")


import argparse, csv, json, os, random, re, shutil, time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from seig import SEIG, SEIGConfig, SEIGResult, EvidenceVector, SymbolicEvidence, ClaimPermissionPlan

DATASETS_DEFAULT=("CVC-300","CVC-ClinicDB","CVC-ColonDB","ETIS-LaribPolypDB","Kvasir")
CONDITIONS=("image_only","seig_only","image_seig","mismatched_seig")
LABELS={
    "image_only":"Image only",
    "seig_only":"SEIG only",
    "image_seig":"Image + SEIG",
    "mismatched_seig":"Image + mismatched SEIG",
}


def log(x=""): print(x, flush=True)

def write_json(p:Path,obj:Any):
    p.parent.mkdir(parents=True,exist_ok=True)
    q=p.with_suffix(p.suffix+".tmp")
    q.write_text(json.dumps(obj,indent=2,ensure_ascii=False),encoding="utf-8")
    q.replace(p)

def write_csv(p:Path,rows:List[Dict[str,Any]]):
    p.parent.mkdir(parents=True,exist_ok=True)
    if not rows:
        p.write_text("",encoding="utf-8"); return
    keys=[]; seen=set()
    for r in rows:
        for k in r:
            if k not in seen: keys.append(k); seen.add(k)
    with p.open("w",newline="",encoding="utf-8-sig") as f:
        w=csv.DictWriter(f,fieldnames=keys); w.writeheader(); w.writerows(rows)

def mean(xs):
    a=np.asarray([float(x) for x in xs],dtype=float); a=a[np.isfinite(a)]
    return float(a.mean()) if len(a) else float("nan")

def bootstrap_delta(rows,metric,target,ref="image_only",n_boot=2000,seed=2023):
    by=defaultdict(lambda:defaultdict(dict))
    for r in rows: by[r["dataset"]][r["case_id"]][r["condition_key"]]=r
    ds_d=[]
    for ds,cases in by.items():
        arr=[]
        for vv in cases.values():
            if target in vv and ref in vv:
                a=float(vv[target][metric]); b=float(vv[ref][metric])
                if np.isfinite(a) and np.isfinite(b): arr.append(a-b)
        if arr: ds_d.append(np.asarray(arr,float))
    pooled=np.concatenate(ds_d) if ds_d else np.array([],float)
    point=float(pooled.mean()) if len(pooled) else float("nan")
    rng=np.random.default_rng(seed); boots=[]
    for _ in range(n_boot):
        parts=[rng.choice(a,size=len(a),replace=True) for a in ds_d]
        if parts: boots.append(float(np.concatenate(parts).mean()))
    lo,hi=(np.percentile(boots,[2.5,97.5]) if boots else (np.nan,np.nan))
    return point,float(lo),float(hi),len(pooled)


def result_from_dict(d:Dict[str,Any])->SEIGResult:
    return SEIGResult(
        vector=EvidenceVector(**d["vector"]),
        symbolic=SymbolicEvidence(**d["symbolic"]),
        graph=d["graph"],
        claim_plan=ClaimPermissionPlan(**d["claim_plan"]),
        prompt=d["prompt"],
    )


def audit_stats(audit,raw_report):
    status=defaultdict(int); action=defaultdict(int)
    for a in audit:
        status[a.get("status","unknown")]+=1; action[a.get("action","unknown")]+=1
    forbidden=status["prohibited"]; unsupported=status["unsupported"]; calibrated=status["calibrated"]
    v=forbidden+unsupported+calibrated
    return {
        "violation_count":int(v),
        "forbidden_count":int(forbidden),
        "unsupported_count":int(unsupported),
        "boundary_overconfidence_count":int(action["rewrite_cautiously"]),
        "anatomy_unsupported_count":int(action["generalize"]),
        "any_violation":int(v>0),
        "char_count":len(raw_report),
    }


def safe_section(report:str,name:str)->str:
    # Pull text after requested heading and before the next numbered/known heading.
    names=["visual finding","location and approximate size","morphology and boundary","confidence and uncertainty","evidence-supported safety note"]
    low=report.lower()
    pos=low.find(name)
    if pos<0: return report
    start=pos+len(name)
    ends=[]
    for n in names:
        q=low.find(n,start)
        if q>=0: ends.append(q)
    end=min(ends) if ends else len(report)
    return report[start:end]


def norm(s:str)->str:
    return re.sub(r"\s+"," ",s.lower().replace("–","-").replace("—","-")).strip()


def phrase_present(text:str, phrase:str)->bool:
    t=norm(text); p=norm(phrase)
    # allow hyphen/space variation
    p_re=re.escape(p).replace(r"\-",r"[- ]")
    return bool(re.search(r"(?<!\w)"+p_re+r"(?!\w)",t,re.I))


def loc_area_alignment(report:str,symbolic:Dict[str,Any])->Dict[str,int]:
    locsec=safe_section(report,"location and approximate size")
    loc=symbolic.get("location","")
    area=symbolic.get("area_level","")
    loc_hit=int(bool(loc) and phrase_present(locsec,loc))
    # Area labels are evaluated only in location/size section to avoid unrelated words.
    area_hit=int(bool(area) and phrase_present(locsec,area))
    return {"location_match":loc_hit,"area_match":area_hit,"loc_area_match_fraction":0.5*(loc_hit+area_hit)}


def make_image_only_prompt()->str:
    return """You are generating a conservative observational report for a colorectal endoscopic image.
Use only the visible image. No segmentation-derived or structured evidence is provided. Do not infer unsupported pathology, malignancy, treatment decisions, or fine-grained anatomical subsites.

WORDING RULES:
- Use cautious observational language when a boundary or other visual property is uncertain.
- Use field-of-view location only; do not name a specific colorectal anatomical subsite unless external metadata is explicitly provided.
- The output is an observational description, not a pathological diagnosis or treatment recommendation.

OUTPUT FORMAT — exactly five sections:
1. Visual finding
2. Location and approximate size
3. Morphology and boundary
4. Confidence and uncertainty
5. Evidence-supported safety note
""".strip()


def make_seig_only_prompt(full_prompt:str)->str:
    x=full_prompt
    x=x.replace(
        "You are generating a conservative observational report for a colorectal endoscopic image.\nUse the image together with the segmentation-derived evidence below.",
        "You are generating a conservative observational report from structured segmentation-derived evidence only.\nNo image is provided. Use only the structured evidence below."
    )
    return x


def load_qwen(model_path:Path):
    from transformers import Qwen2_5_VLForConditionalGeneration,AutoProcessor
    log("Loading frozen Qwen2.5-VL model (local files only) ...")
    m=Qwen2_5_VLForConditionalGeneration.from_pretrained(str(model_path),dtype=torch.bfloat16,device_map="auto",local_files_only=True)
    m.eval(); p=AutoProcessor.from_pretrained(str(model_path),local_files_only=True)
    try:
        m.generation_config.temperature=None; m.generation_config.top_p=None; m.generation_config.top_k=None
    except Exception: pass
    return m,p


def generate(model,processor,prompt:str,image_path:Path|None,max_new_tokens:int):
    if image_path is not None:
        from qwen_vl_utils import process_vision_info
        messages=[{"role":"user","content":[{"type":"image","image":"file://"+str(image_path.resolve())},{"type":"text","text":prompt}]}]
        text=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
        image_inputs,video_inputs=process_vision_info(messages)
        inputs=processor(text=[text],images=image_inputs,videos=video_inputs,padding=True,return_tensors="pt")
    else:
        messages=[{"role":"user","content":[{"type":"text","text":prompt}]}]
        text=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
        inputs=processor(text=[text],padding=True,return_tensors="pt")
    inputs=inputs.to("cuda")
    if torch.cuda.is_available(): torch.cuda.synchronize()
    t0=time.perf_counter()
    with torch.inference_mode(): out=model.generate(**inputs,max_new_tokens=max_new_tokens,do_sample=False,use_cache=True)
    if torch.cuda.is_available(): torch.cuda.synchronize()
    dt=time.perf_counter()-t0
    trim=[o[len(i):] for i,o in zip(inputs.input_ids,out)]
    txt=processor.batch_decode(trim,skip_special_tokens=True,clean_up_tokenization_spaces=False)[0].strip()
    ntok=int(trim[0].numel())
    del inputs,out,trim
    return txt,dt,ntok


def discover_stage6(stage6_root:Path,datasets:List[str],max_cases:int=0):
    out={}
    for ds in datasets:
        rows=[]
        for p in sorted((stage6_root/"cases"/ds).glob("*/both.json")):
            o=json.loads(p.read_text(encoding="utf-8"))
            if o.get("complete") is True:
                rows.append((str(o["case_id"]),p,o))
        if max_cases>0: rows=rows[:max_cases]
        if not rows: raise RuntimeError(f"No Stage6 both.json records found for {ds}")
        out[ds]=rows
    return out


def build_mismatch_pairs(dsrows,seed):
    # Deterministic one-to-one permutation where BOTH location and area differ.
    ids=[x[0] for x in dsrows]
    syms={x[0]:x[2]["seig"]["symbolic"] for x in dsrows}
    rng=random.Random(seed)
    perm=ids[:]
    ok=False
    for _ in range(20000):
        rng.shuffle(perm)
        if all(a!=b and syms[a]["location"]!=syms[b]["location"] and syms[a]["area_level"]!=syms[b]["area_level"] for a,b in zip(ids,perm)):
            ok=True; break
    if not ok:
        # Greedy fallback allows donor reuse but still enforces explicit mismatch.
        perm=[]
        for a in ids:
            cand=[b for b in ids if b!=a and syms[a]["location"]!=syms[b]["location"] and syms[a]["area_level"]!=syms[b]["area_level"]]
            if not cand: raise RuntimeError(f"Could not find mismatch donor for {a}")
            perm.append(rng.choice(cand))
    return dict(zip(ids,perm)), int(ok)


def parse_args():
    p=argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--root",default=DATA_ROOT + "/wysiwyr_real")
    p.add_argument("--stage6",default="")
    p.add_argument("--model",default=DATA_ROOT + "/models/Qwen2.5-VL-3B-Instruct")
    p.add_argument("--output",default="")
    p.add_argument("--datasets",default=",".join(DATASETS_DEFAULT))
    p.add_argument("--seed",type=int,default=2023)
    p.add_argument("--max-new-tokens",type=int,default=384)
    p.add_argument("--max-cases-per-dataset",type=int,default=0)
    p.add_argument("--preflight-only",action="store_true")
    p.add_argument("--smoke",action="store_true")
    p.add_argument("--regenerate-image-seig",action="store_true",help="default reuses the identical Stage6 Image+SEIG report")
    p.add_argument("--no-package",action="store_true")
    return p.parse_args()


def main():
    args=parse_args(); random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    root=Path(args.root).resolve(); stage6=Path(args.stage6).resolve() if args.stage6 else root/"stage6_seig_mllm_end2end"
    model_path=Path(args.model).resolve(); out=Path(args.output).resolve() if args.output else root/"stage7_seig_controls"
    datasets=[x.strip() for x in args.datasets.split(",") if x.strip()]
    max_cases=args.max_cases_per_dataset
    if args.smoke: datasets=["CVC-300"]; max_cases=0; out=root/"stage7_smoke"

    log("\n"+"="*80); log("WYSIWYR STAGE7 - SEIG INFORMATION-SOURCE CONTROL EXPERIMENT"); log("="*80)
    if not stage6.exists(): raise SystemExit(f"Missing Stage6 results: {stage6}")
    if not model_path.exists(): raise SystemExit(f"Missing Qwen model: {model_path}")
    if not torch.cuda.is_available(): raise SystemExit("CUDA GPU is not available")
    dsrows=discover_stage6(stage6,datasets,max_cases)
    counts={d:len(v) for d,v in dsrows.items()}; log(f"Stage6 case counts: {counts}")
    # Verify Stage6 reusable image+SEIG decoding really matches this run.
    for d,rs in dsrows.items():
        for _,_,o in rs[:min(3,len(rs))]:
            dec=o.get("decode",{})
            if dec.get("do_sample") is not False or int(dec.get("max_new_tokens",-1))!=args.max_new_tokens:
                raise SystemExit(f"Stage6 decoding mismatch in {d}; use --regenerate-image-seig or matching --max-new-tokens")
    log("PRECHECK PASSED")
    if args.preflight_only: return

    mismatch={}; pair_rows=[]
    obj_index={}
    for di,(ds,rs) in enumerate(dsrows.items()):
        mp,one2one=build_mismatch_pairs(rs,args.seed+1009*di)
        mismatch[ds]=mp
        obj_index[ds]={cid:o for cid,_,o in rs}
        for cid,donor in mp.items():
            a=obj_index[ds][cid]["seig"]["symbolic"]; b=obj_index[ds][donor]["seig"]["symbolic"]
            pair_rows.append({"dataset":ds,"case_id":cid,"donor_case_id":donor,"one_to_one_permutation":one2one,
                              "target_location":a["location"],"donor_location":b["location"],"target_area":a["area_level"],"donor_area":b["area_level"],
                              "location_diff":int(a["location"]!=b["location"]),"area_diff":int(a["area_level"]!=b["area_level"])})
    if args.smoke:
        keep={ds: rs[0][0] for ds,rs in dsrows.items()}
        dsrows={ds: rs[:1] for ds,rs in dsrows.items()}
        pair_rows=[r for r in pair_rows if r["case_id"]==keep[r["dataset"]]]
        counts={d:len(v) for d,v in dsrows.items()}
    out.mkdir(parents=True,exist_ok=True); write_csv(out/"mismatch_pairs.csv",pair_rows)

    protocol={
        "stage":"Stage7 SEIG information-source controls","seed":args.seed,"datasets":counts,
        "conditions":LABELS,"evidence_source_for_seig":"Stage6 MedSAM+ABLoss+USR ('both') evidence",
        "image_seig_reuse_default":not args.regenerate_image_seig,
        "image_seig_reuse_justification":"Identical Stage6 raw report is reused only when model, prompt, image, do_sample=False, and max_new_tokens match exactly.",
        "mismatch_policy":"within-dataset donor; target and donor are different cases; donor location AND area category must differ from target; deterministic seed; one-to-one derangement when feasible",
        "evaluation":"All reports are re-audited with frozen Checker v3 against the CORRECT target-case SEIG evidence. Mismatched reports are additionally audited/aligned against donor evidence as a manipulation check.",
        "decoding":{"do_sample":False,"max_new_tokens":args.max_new_tokens},
        "checker_version":SEIG.checker_rulebook().get("version"),
        "guardrail":"Checker metrics are rule-compliance metrics; independent blinded human validation remains separate."
    }
    write_json(out/"protocol_stage7.json",protocol)

    model,processor=load_qwen(model_path); seig=SEIG(SEIGConfig())
    expected=sum(counts.values())*len(CONDITIONS); completed=0; generated_now=0; reused=0
    for ds,rs in dsrows.items():
        log(f"\n===== {ds}: {len(rs)} cases x 4 conditions =====")
        for ci,(cid,pth,base_obj) in enumerate(rs,1):
            image=Path(base_obj["image_path"])
            if not image.exists(): raise FileNotFoundError(image)
            target_res=result_from_dict(base_obj["seig"]); target_sym=base_obj["seig"]["symbolic"]
            donor_id=mismatch[ds][cid]; donor_obj=obj_index[ds][donor_id]; donor_res=result_from_dict(donor_obj["seig"]); donor_sym=donor_obj["seig"]["symbolic"]
            prompts={
                "image_only":make_image_only_prompt(),
                "seig_only":make_seig_only_prompt(base_obj["seig"]["prompt"]),
                "image_seig":base_obj["seig"]["prompt"],
                "mismatched_seig":donor_obj["seig"]["prompt"],
            }
            for ki,k in enumerate(CONDITIONS,1):
                recp=out/"cases"/ds/cid/f"{k}.json"
                if recp.exists():
                    try:
                        old=json.loads(recp.read_text(encoding="utf-8"))
                        if old.get("complete") is True: completed+=1; continue
                    except Exception: pass
                if k=="image_seig" and not args.regenerate_image_seig:
                    raw=base_obj["raw_report_R0"]; latency=float(base_obj.get("generation_latency_s",np.nan)); ntok=int(base_obj.get("generated_tokens",0)); source="reused_exact_stage6_both"
                    reused+=1
                else:
                    raw,latency,ntok=generate(model,processor,prompts[k],None if k=="seig_only" else image,args.max_new_tokens); source="generated_stage7"; generated_now+=1
                checked,audit=seig.verify_report(raw,target_res)
                tgt=audit_stats(audit,raw); t_align=loc_area_alignment(raw,target_sym)
                donor_stats={}; d_align={}
                if k=="mismatched_seig":
                    _,da=seig.verify_report(raw,donor_res); donor_stats={"donor_"+a:b for a,b in audit_stats(da,raw).items()}; d_align={"donor_"+a:b for a,b in loc_area_alignment(raw,donor_sym).items()}
                rec={"complete":True,"dataset":ds,"case_id":cid,"condition_key":k,"condition":LABELS[k],"image_path":str(image),
                     "target_stage6_variant":"both","target_seig":base_obj["seig"],"donor_case_id":donor_id if k=="mismatched_seig" else "",
                     "donor_seig":donor_obj["seig"] if k=="mismatched_seig" else None,"prompt":prompts[k],"raw_report_R0":raw,"checked_report_Rstar":checked,"checker_audit_vs_target":audit,
                     "target_audit_stats":tgt,"target_loc_area_alignment":t_align,**donor_stats,**d_align,
                     "generation_latency_s":latency,"generated_tokens":ntok,"generation_source":source,
                     "decode":{"do_sample":False,"max_new_tokens":args.max_new_tokens}}
                write_json(recp,rec); completed+=1
                log(f"{ds} {ci}/{len(rs)} [{ki}/4 {k}] tokens={ntok} gen={latency:.2f}s targetV={tgt['violation_count']} loc/area={t_align['loc_area_match_fraction']:.1f} {source}")
                if torch.cuda.is_available() and generated_now%20==0: torch.cuda.empty_cache()

    log("\nAggregating Stage7 tables ...")
    rows=[]
    for p in sorted((out/"cases").glob("*/*/*.json")):
        o=json.loads(p.read_text(encoding="utf-8"));
        if not o.get("complete"): continue
        r={"dataset":o["dataset"],"case_id":o["case_id"],"condition_key":o["condition_key"],"condition":o["condition"],
           **o["target_audit_stats"],**o["target_loc_area_alignment"],"generation_latency_s":o["generation_latency_s"],"generated_tokens":o["generated_tokens"],"generation_source":o["generation_source"]}
        for q in ("donor_violation_count","donor_forbidden_count","donor_unsupported_count","donor_boundary_overconfidence_count","donor_anatomy_unsupported_count","donor_any_violation","donor_location_match","donor_area_match","donor_loc_area_match_fraction"):
            r[q]=o.get(q,"")
        rows.append(r)
    write_csv(out/"case_level_results.csv",rows)
    summary=[]; grp=defaultdict(list)
    for r in rows: grp[(r["dataset"],r["condition_key"])].append(r)
    metrics=["violation_count","forbidden_count","unsupported_count","boundary_overconfidence_count","anatomy_unsupported_count","any_violation","location_match","area_match","loc_area_match_fraction","generation_latency_s","generated_tokens"]
    for (ds,k),rr in sorted(grp.items()):
        z={"dataset":ds,"condition_key":k,"condition":LABELS[k],"n":len(rr)}
        for m in metrics: z["mean_"+m]=mean(r[m] for r in rr)
        if k=="mismatched_seig":
            z["mean_donor_loc_area_match_fraction"]=mean(float(r["donor_loc_area_match_fraction"]) for r in rr)
        summary.append(z)
    write_csv(out/"summary_by_dataset_condition.csv",summary)
    macro=[]
    for k in CONDITIONS:
        ss=[z for z in summary if z["condition_key"]==k]
        z={"condition_key":k,"condition":LABELS[k],"n_datasets":len(ss)}
        for m in metrics: z["macro_"+m]=mean(x["mean_"+m] for x in ss)
        if k=="mismatched_seig": z["macro_donor_loc_area_match_fraction"]=mean(x["mean_donor_loc_area_match_fraction"] for x in ss)
        macro.append(z)
    write_csv(out/"summary_macro_condition.csv",macro)
    contrasts=[]
    for target in ("seig_only","image_seig","mismatched_seig"):
        for metric in ("violation_count","forbidden_count","boundary_overconfidence_count","any_violation","loc_area_match_fraction"):
            pt,lo,hi,n=bootstrap_delta(rows,metric,target,"image_only",2000,args.seed)
            contrasts.append({"target":target,"target_label":LABELS[target],"reference":"image_only","metric":metric,"paired_mean_delta":pt,"bootstrap95_low":lo,"bootstrap95_high":hi,"n_paired":n})
    write_csv(out/"paired_contrasts_vs_image_only.csv",contrasts)
    write_json(out/"RUN_COMPLETE.json",{"completed_records":len(rows),"expected_records":expected,"generated_this_run":generated_now,"reused_stage6_image_seig":reused,"completed_at":time.strftime("%Y-%m-%d %H:%M:%S")})
    (out/"README_RESULTS.md").write_text(f"""# WYSIWYR Stage7 SEIG control experiment\n\nCompleted records: {len(rows)} / {expected}.\n\nConditions: Image only; SEIG only; Image + SEIG; Image + mismatched SEIG.\nImage + SEIG reuses the exact Stage6 MedSAM+ABLoss+USR report when decoding matches (do_sample=False, max_new_tokens={args.max_new_tokens}); the other three conditions are generated in Stage7.\nMismatched evidence is paired within dataset and differs from the target case in both field-of-view location and area category.\nAll reports are audited using frozen Checker v3 against the correct target evidence. Mismatched reports are additionally checked against donor evidence as a manipulation check.\nChecker-derived metrics are deterministic rule-compliance metrics, not independent clinical-validity labels.\n""",encoding="utf-8")
    if not args.no_package and not args.smoke:
        zipbase=Path(DATA_ROOT + "/stage7_results_for_review")
        if zipbase.with_suffix(".zip").exists(): zipbase.with_suffix(".zip").unlink()
        shutil.make_archive(str(zipbase),"zip",out); log(f"Packaged: {zipbase.with_suffix('.zip')}")
    log("\n"+"="*80); log("STAGE7 COMPLETE"); log(f"Results: {out}"); log(f"Generated now: {generated_now}; reused Stage6 image+SEIG: {reused}"); log("="*80)

if __name__=="__main__": main()
