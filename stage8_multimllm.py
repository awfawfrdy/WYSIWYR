#!/usr/bin/env python3
from __future__ import annotations
import os
DATA_ROOT = os.environ.get("WYSIWYR_DATA_ROOT", ".")


import argparse, csv, json, random, re, shutil, time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from PIL import Image

from seig import SEIG, SEIGConfig, SEIGResult, EvidenceVector, SymbolicEvidence, ClaimPermissionPlan

DATASETS=("CVC-300","CVC-ClinicDB","CVC-ColonDB","ETIS-LaribPolypDB","Kvasir")
CONDS=("image_only","seig_only","image_seig","mismatched_seig")
LABELS={"image_only":"Image only","seig_only":"SEIG only","image_seig":"Image + SEIG","mismatched_seig":"Image + mismatched SEIG"}


def log(s=""): print(s, flush=True)

def write_json(p:Path,obj:Any):
    p.parent.mkdir(parents=True,exist_ok=True)
    q=p.with_suffix(p.suffix+".tmp")
    q.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding="utf-8")
    q.replace(p)

def write_csv(p:Path,rows:List[Dict[str,Any]]):
    p.parent.mkdir(parents=True,exist_ok=True)
    if not rows:
        p.write_text("",encoding="utf-8"); return
    keys=[]; seen=set()
    for r in rows:
        for k in r:
            if k not in seen: seen.add(k); keys.append(k)
    with p.open("w",newline="",encoding="utf-8-sig") as f:
        w=csv.DictWriter(f,fieldnames=keys); w.writeheader(); w.writerows(rows)

def mean(xs):
    a=np.asarray([float(x) for x in xs],float); a=a[np.isfinite(a)]
    return float(a.mean()) if len(a) else float("nan")

def result_from_dict(d):
    return SEIGResult(vector=EvidenceVector(**d["vector"]),symbolic=SymbolicEvidence(**d["symbolic"]),graph=d["graph"],claim_plan=ClaimPermissionPlan(**d["claim_plan"]),prompt=d["prompt"])

def audit_stats(audit,raw):
    st=defaultdict(int); ac=defaultdict(int)
    for x in audit:
        st[x.get("status","unknown")]+=1; ac[x.get("action","unknown")]+=1
    forbidden=st["prohibited"]; unsupported=st["unsupported"]; calibrated=st["calibrated"]
    v=forbidden+unsupported+calibrated
    return {"violation_count":int(v),"forbidden_count":int(forbidden),"unsupported_count":int(unsupported),"boundary_overconfidence_count":int(ac["rewrite_cautiously"]),"anatomy_unsupported_count":int(ac["generalize"]),"any_violation":int(v>0),"char_count":len(raw)}

def safe_section(report,name):
    names=["visual finding","location and approximate size","morphology and boundary","confidence and uncertainty","evidence-supported safety note"]
    low=report.lower(); pos=low.find(name)
    if pos<0: return report
    start=pos+len(name); ends=[q for n in names if (q:=low.find(n,start))>=0]
    return report[start:min(ends) if ends else len(report)]

def phrase_present(text,phrase):
    t=re.sub(r"\s+"," ",text.lower().replace("–","-").replace("—","-")).strip()
    p=re.sub(r"\s+"," ",str(phrase).lower().replace("–","-").replace("—","-")).strip()
    pr=re.escape(p).replace(r"\-",r"[- ]")
    return bool(re.search(r"(?<!\w)"+pr+r"(?!\w)",t,re.I)) if p else False

def loc_area_alignment(report,sym):
    sec=safe_section(report,"location and approximate size")
    lh=int(phrase_present(sec,sym.get("location",""))); ah=int(phrase_present(sec,sym.get("area_level","")))
    return {"location_match":lh,"area_match":ah,"loc_area_match_fraction":0.5*(lh+ah)}

# ---------- InternVL adapter ----------
IMAGENET_MEAN=(0.485,0.456,0.406); IMAGENET_STD=(0.229,0.224,0.225)

def iv_transform(input_size=448):
    from torchvision import transforms as T
    from torchvision.transforms.functional import InterpolationMode
    return T.Compose([T.Lambda(lambda x:x.convert("RGB")),T.Resize((input_size,input_size),interpolation=InterpolationMode.BICUBIC),T.ToTensor(),T.Normalize(mean=IMAGENET_MEAN,std=IMAGENET_STD)])

def iv_dynamic(image,min_num=1,max_num=12,image_size=448,use_thumbnail=True):
    ow,oh=image.size; ar=ow/oh
    ratios=set()
    for n in range(min_num,max_num+1):
        for i in range(1,n+1):
            for j in range(1,n+1):
                if min_num <= i*j <= max_num: ratios.add((i,j))
    ratios=sorted(ratios,key=lambda x:x[0]*x[1])
    target=min(ratios,key=lambda r:abs(ar-r[0]/r[1]))
    tw,th=image_size*target[0],image_size*target[1]
    resized=image.resize((tw,th))
    blocks=[]
    for k in range(target[0]*target[1]):
        x=(k%target[0])*image_size; y=(k//target[0])*image_size
        blocks.append(resized.crop((x,y,x+image_size,y+image_size)))
    if use_thumbnail and len(blocks)!=1: blocks.append(image.resize((image_size,image_size)))
    return blocks

def iv_load_image(path,max_num=12):
    im=Image.open(path).convert("RGB"); tf=iv_transform(448)
    px=torch.stack([tf(x) for x in iv_dynamic(im,max_num=max_num)])
    return px

def load_internvl(path):
    from transformers import AutoModel,AutoTokenizer
    tok=AutoTokenizer.from_pretrained(str(path),trust_remote_code=True,use_fast=False,local_files_only=True)
    model=AutoModel.from_pretrained(str(path),torch_dtype=torch.bfloat16,low_cpu_mem_usage=True,trust_remote_code=True,local_files_only=True).eval().cuda()
    return model,tok

def gen_internvl(model,tok,prompt,image_path,max_new_tokens):
    pixels=None; question=prompt
    if image_path is not None:
        pixels=iv_load_image(image_path,max_num=12).to(torch.bfloat16).cuda(); question="<image>\n"+prompt
    cfg={"max_new_tokens":max_new_tokens,"do_sample":False}
    torch.cuda.synchronize(); t0=time.perf_counter()
    with torch.inference_mode():
        out=model.chat(tok,pixels,question,cfg)
    torch.cuda.synchronize(); dt=time.perf_counter()-t0
    if isinstance(out,(tuple,list)): out=out[0]
    txt=str(out).strip(); ntok=len(tok.encode(txt,add_special_tokens=False))
    del pixels
    return txt,dt,ntok

# ---------- MiniCPM adapter ----------
def load_minicpm(path):
    from transformers import AutoModel,AutoTokenizer
    tok=AutoTokenizer.from_pretrained(str(path),trust_remote_code=True,local_files_only=True)
    kwargs=dict(trust_remote_code=True,torch_dtype=torch.bfloat16,low_cpu_mem_usage=True,local_files_only=True)
    try: model=AutoModel.from_pretrained(str(path),attn_implementation="sdpa",**kwargs)
    except Exception: model=AutoModel.from_pretrained(str(path),**kwargs)
    model=model.eval().cuda(); return model,tok

def gen_minicpm(model,tok,prompt,image_path,max_new_tokens):
    content=[]
    if image_path is not None: content.append(Image.open(image_path).convert("RGB"))
    content.append(prompt)
    msgs=[{"role":"user","content":content}]
    torch.cuda.synchronize(); t0=time.perf_counter()
    with torch.inference_mode():
        try: out=model.chat(image=None,msgs=msgs,tokenizer=tok,sampling=False,max_new_tokens=max_new_tokens)
        except TypeError: out=model.chat(image=None,msgs=msgs,tokenizer=tok,sampling=False)
    torch.cuda.synchronize(); dt=time.perf_counter()-t0
    txt=str(out).strip(); ntok=len(tok.encode(txt,add_special_tokens=False))
    return txt,dt,ntok


# ---------- HuatuoGPT-Vision adapter ----------
# Current reproducibility interface for the HuatuoGPT-Vision cross-MLLM condition.
# The historical model-specific adapter that generated the manuscript Table-30
# HuatuoGPT-Vision numbers was NOT retained. This adapter is provided as a current
# reproducibility interface only; it reuses the same Stage-8 image loading, SEIG
# evidence, structured prompt template, frozen checker and output schema, and only
# adds the Huatuo-specific loading/inference interface.
# Explicit model identifier (HuggingFace): see HUATUO_MODEL_ID below.
HUATUO_MODEL_ID="FreedomIntelligence/HuatuoGPT-Vision-7B-Qwen2.5VL"
HUATUO_MAX_NEW_TOKENS=512

def load_huatuo(model_path):
    from transformers import Qwen2_5_VLForConditionalGeneration,AutoProcessor
    log("Loading frozen HuatuoGPT-Vision (Qwen2.5-VL architecture) ...")
    m=Qwen2_5_VLForConditionalGeneration.from_pretrained(str(model_path),dtype=torch.bfloat16,device_map="auto")
    m.eval(); p=AutoProcessor.from_pretrained(str(model_path))
    try:
        m.generation_config.temperature=None; m.generation_config.top_p=None; m.generation_config.top_k=None
    except Exception: pass
    return m,p

def gen_huatuo(model,processor,prompt,image_path,max_new_tokens):
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


def select_manifest(stage7_root,cases_per_dataset,seed,out):
    mf=out/"sample_manifest.csv"
    if mf.exists():
        rows=list(csv.DictReader(mf.open(encoding="utf-8-sig")))
        return [(r["dataset"],r["case_id"]) for r in rows]
    rng=random.Random(seed); selected=[]; rows=[]
    for ds in DATASETS:
        ids=sorted([p.name for p in (stage7_root/"cases"/ds).iterdir() if p.is_dir()])
        if not ids: raise RuntimeError(f"No Stage7 cases for {ds}")
        if cases_per_dataset<=0 or cases_per_dataset>=len(ids): take=ids
        else: take=sorted(rng.sample(ids,cases_per_dataset))
        for cid in take: selected.append((ds,cid)); rows.append({"dataset":ds,"case_id":cid,"seed":seed})
    write_csv(mf,rows); return selected


def load_stage7_condition(stage7_root,ds,cid,k):
    p=stage7_root/"cases"/ds/cid/f"{k}.json"
    if not p.exists(): raise FileNotFoundError(p)
    return json.loads(p.read_text(encoding="utf-8"))


def aggregate(out):
    rows=[]
    for p in sorted((out/"cases").glob("*/*/*/*.json")):
        o=json.loads(p.read_text(encoding="utf-8"))
        if not o.get("complete"): continue
        r={"model_key":o["model_key"],"model_label":o["model_label"],"dataset":o["dataset"],"case_id":o["case_id"],"condition_key":o["condition_key"],"condition":o["condition"],**o["target_audit_stats"],**o["target_loc_area_alignment"],"generation_latency_s":o["generation_latency_s"],"generated_tokens":o["generated_tokens"],"generation_source":o["generation_source"]}
        if o["condition_key"]=="mismatched_seig":
            r["donor_loc_area_match_fraction"]=o.get("donor_loc_area_match_fraction","")
        rows.append(r)
    write_csv(out/"case_level_results.csv",rows)
    grp=defaultdict(list)
    for r in rows: grp[(r["model_key"],r["dataset"],r["condition_key"])].append(r)
    metrics=["violation_count","forbidden_count","unsupported_count","boundary_overconfidence_count","anatomy_unsupported_count","any_violation","location_match","area_match","loc_area_match_fraction","generation_latency_s","generated_tokens"]
    ss=[]
    for (mk,ds,k),rr in sorted(grp.items()):
        z={"model_key":mk,"model_label":rr[0]["model_label"],"dataset":ds,"condition_key":k,"condition":LABELS[k],"n":len(rr)}
        for m in metrics: z["mean_"+m]=mean(r[m] for r in rr)
        if k=="mismatched_seig": z["mean_donor_loc_area_match_fraction"]=mean(float(r["donor_loc_area_match_fraction"]) for r in rr if r.get("donor_loc_area_match_fraction","")!="")
        ss.append(z)
    write_csv(out/"summary_by_model_dataset_condition.csv",ss)
    macro=[]
    for mk in sorted({x["model_key"] for x in ss}):
        for k in CONDS:
            xx=[x for x in ss if x["model_key"]==mk and x["condition_key"]==k]
            if not xx: continue
            z={"model_key":mk,"model_label":xx[0]["model_label"],"condition_key":k,"condition":LABELS[k],"n_datasets":len(xx)}
            for m in metrics: z["macro_"+m]=mean(x["mean_"+m] for x in xx)
            if k=="mismatched_seig": z["macro_donor_loc_area_match_fraction"]=mean(x["mean_donor_loc_area_match_fraction"] for x in xx)
            macro.append(z)
    write_csv(out/"summary_macro_model_condition.csv",macro)
    return rows


def import_qwen_reuse(stage7_root,out,manifest):
    mk="qwen25vl3b"; ml="Qwen2.5-VL-3B-Instruct"
    for ds,cid in manifest:
        for k in CONDS:
            dest=out/"cases"/mk/ds/cid/f"{k}.json"
            if dest.exists():
                try:
                    if json.loads(dest.read_text(encoding="utf-8")).get("complete"): continue
                except Exception: pass
            s=load_stage7_condition(stage7_root,ds,cid,k)
            rec={"complete":True,"model_key":mk,"model_label":ml,"dataset":ds,"case_id":cid,"condition_key":k,"condition":LABELS[k],"raw_report_R0":s["raw_report_R0"],"checked_report_Rstar":s["checked_report_Rstar"],"target_audit_stats":s["target_audit_stats"],"target_loc_area_alignment":s["target_loc_area_alignment"],"donor_loc_area_match_fraction":s.get("donor_loc_area_match_fraction",""),"generation_latency_s":s["generation_latency_s"],"generated_tokens":s["generated_tokens"],"generation_source":"reused_exact_stage7_qwen","decode":s.get("decode",{})}
            write_json(dest,rec)


def run_backend(backend,model_path,stage7_root,out,manifest,max_new_tokens):
    if backend=="internvl": model,tok=load_internvl(model_path); mk="internvl25_2b"; ml="InternVL2.5-2B"; gen=gen_internvl
    elif backend=="minicpm": model,tok=load_minicpm(model_path); mk="minicpmv26"; ml="MiniCPM-V-2.6"; gen=gen_minicpm
    elif backend=="huatuo": model,tok=load_huatuo(model_path); mk="huatuogpt_vision_7b"; ml="HuatuoGPT-Vision"; gen=gen_huatuo
    else: raise ValueError(backend)
    seig=SEIG(SEIGConfig()); total=len(manifest)*4; done=0
    for ds,cid in manifest:
        for k in CONDS:
            done+=1; dest=out/"cases"/mk/ds/cid/f"{k}.json"
            if dest.exists():
                try:
                    if json.loads(dest.read_text(encoding="utf-8")).get("complete"):
                        if done%25==0: log(f"{ml} {done}/{total} [skip]")
                        continue
                except Exception: pass
            src=load_stage7_condition(stage7_root,ds,cid,k)
            prompt=src["prompt"]; image=None if k=="seig_only" else Path(src["image_path"])
            raw,lat,ntok=gen(model,tok,prompt,image,max_new_tokens)
            target= result_from_dict(src["target_seig"]); checked,audit=seig.verify_report(raw,target)
            stats=audit_stats(audit,raw); align=loc_area_alignment(raw,src["target_seig"]["symbolic"])
            donor_align=""
            if k=="mismatched_seig" and src.get("donor_seig"):
                donor_align=loc_area_alignment(raw,src["donor_seig"]["symbolic"])["loc_area_match_fraction"]
            rec={"complete":True,"model_key":mk,"model_label":ml,"dataset":ds,"case_id":cid,"condition_key":k,"condition":LABELS[k],"raw_report_R0":raw,"checked_report_Rstar":checked,"checker_audit_vs_target":audit,"target_audit_stats":stats,"target_loc_area_alignment":align,"donor_loc_area_match_fraction":donor_align,"generation_latency_s":lat,"generated_tokens":ntok,"generation_source":"generated_stage8","decode":{"do_sample":False,"max_new_tokens":max_new_tokens}}
            write_json(dest,rec)
            log(f"{ml} {done}/{total} {ds}/{cid} {k} tokens={ntok} gen={lat:.2f}s V={stats['violation_count']} loc/area={align['loc_area_match_fraction']:.1f}")
            if done%20==0: torch.cuda.empty_cache()
    del model,tok; torch.cuda.empty_cache()


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--stage7",default=DATA_ROOT + "/wysiwyr_real/stage7_seig_controls")
    ap.add_argument("--output",default=DATA_ROOT + "/wysiwyr_real/stage8_multimllm")
    ap.add_argument("--backend",choices=["reuse-qwen","internvl","minicpm","huatuo","aggregate"],required=True)
    ap.add_argument("--model",default="")
    ap.add_argument("--cases-per-dataset",type=int,default=40)
    ap.add_argument("--seed",type=int,default=2023)
    ap.add_argument("--max-new-tokens",type=int,default=384)
    ap.add_argument("--smoke",action="store_true")
    args=ap.parse_args()
    s7=Path(args.stage7); out=Path(args.output)
    if args.smoke:
        out=out.parent/"stage8_smoke"
    out.mkdir(parents=True,exist_ok=True)
    if not s7.exists(): raise SystemExit(f"Missing Stage7: {s7}")
    cpd=1 if args.smoke else args.cases_per_dataset
    manifest=select_manifest(s7,cpd,args.seed,out)
    log(f"Manifest cases: {len(manifest)} ({cpd}/dataset)")
    if args.backend=="reuse-qwen": import_qwen_reuse(s7,out,manifest)
    elif args.backend in ("internvl","minicpm","huatuo"):
        if args.backend=="huatuo": args.max_new_tokens=HUATUO_MAX_NEW_TOKENS
        mp=Path(args.model)
        if not mp.exists(): raise SystemExit(f"Missing model: {mp}")
        if not torch.cuda.is_available(): raise SystemExit("CUDA unavailable")
        run_backend(args.backend,mp,s7,out,manifest,args.max_new_tokens)
    elif args.backend=="aggregate": pass
    rows=aggregate(out)
    write_json(out/"protocol_stage8.json",{"stage":"Stage8 Multi-MLLM robustness","seed":args.seed,"cases_per_dataset":cpd,"n_sample_cases":len(manifest),"conditions":LABELS,"qwen_source":"exact Stage7 reuse","additional_models":["InternVL2.5-2B","MiniCPM-V-2.6","HuatuoGPT-Vision"],"decode":{"do_sample":False,"max_new_tokens":args.max_new_tokens},"checker":"frozen v3","note":"Reviewer-requested key SEIG controls repeated across multiple MLLMs; fixed stratified case sample shared across models."})
    log(f"Aggregated records now: {len(rows)}")
    if not args.smoke and args.backend=="aggregate":
        base=Path(DATA_ROOT + "/stage8_multimllm_results_for_review")
        if base.with_suffix(".zip").exists(): base.with_suffix(".zip").unlink()
        shutil.make_archive(str(base),"zip",out)
        log(f"RESULT ZIP: {base.with_suffix('.zip')}")
    log("STAGE8 STEP COMPLETE")

if __name__=="__main__": main()
