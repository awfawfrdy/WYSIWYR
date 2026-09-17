#!/usr/bin/env python3
"""
WYSIWYR Stage 5: validation-calibrated prompt + USR repair.

This stage reuses ALL trained weights from Stage 3/4. It does not retrain MedSAM,
ABLoss, or the prompt-free proposer. It fixes two Stage-4 calibration problems:
1) prompt boxes are selected by downstream MedSAM validation Dice (not box IoU alone),
   with a near-optimal high-coverage tie-break;
2) USR pixel parameters use a canonical 1024-px MedSAM reference, and pruning/repair
   thresholds are calibrated on the 580-image validation split using absolute soft
   probabilities. This avoids the Stage-4 incompatibility where m0=(p>0.5) but pruning
   also required p<0.5, making pruning impossible when normalization='none'.

Main-test leakage rule: GT is never used to construct test prompts or decide whether
USR is applied. All hyperparameters are frozen after validation-only calibration.
"""
from __future__ import annotations
import argparse, csv, json, os, random, time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import cv2
import numpy as np
import torch

import run_real_medsam_2x2 as s3
import stage4_promptfree as s4
import wysiwyr_autodl_allinone as core


def log(x=''): print(x, flush=True)
def write_json(p: Path, x: Any): p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(x,indent=2,ensure_ascii=False),encoding='utf-8')
def write_csv(p: Path, rows: Sequence[Dict[str,Any]]):
    p.parent.mkdir(parents=True,exist_ok=True)
    if not rows: p.write_text('',encoding='utf-8'); return
    keys=[]; seen=set()
    for r in rows:
        for k in r:
            if k not in seen: keys.append(k); seen.add(k)
    with p.open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=keys); w.writeheader(); w.writerows(rows)

def append_csv(p: Path, row: Dict[str,Any]):
    p.parent.mkdir(parents=True,exist_ok=True); exists=p.exists() and p.stat().st_size>0
    with p.open('a',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(row));
        if not exists: w.writeheader()
        w.writerow(row)

def mean(xs):
    a=np.asarray(xs,float); a=a[np.isfinite(a)]; return float(a.mean()) if len(a) else float('nan')

def seed_all(seed:int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

# -------------------------- robust proposal boxes --------------------------
def component_mask(binary: np.ndarray, mode: str) -> np.ndarray:
    m=(binary>0).astype(np.uint8)
    n,lab,stats,_=cv2.connectedComponentsWithStats(m,8)
    if n<=1: return np.zeros_like(m)
    areas=stats[1:,cv2.CC_STAT_AREA]
    order=np.argsort(-areas)+1
    if mode=='largest':
        return (lab==int(order[0])).astype(np.uint8)
    if mode=='top3_union':
        largest=float(areas[order[0]-1]); keep=[]
        for j in order[:3]:
            a=float(stats[j,cv2.CC_STAT_AREA])
            if a>=max(4.0,0.15*largest): keep.append(int(j))
        return np.isin(lab,keep).astype(np.uint8)
    raise ValueError(mode)

def box_from_prob(prob: np.ndarray, threshold: float, margin: float, mode: str):
    h,w=prob.shape; fallback='none'
    m=component_mask(prob>=threshold,mode)
    if int(m.sum())==0:
        adaptive=max(0.03,min(float(threshold),0.60*float(prob.max())))
        m=component_mask(prob>=adaptive,mode); fallback='adaptive_threshold'
    if int(m.sum())==0:
        yy,xx=np.unravel_index(int(np.argmax(prob)),prob.shape)
        rw=max(8,int(round(0.25*w))); rh=max(8,int(round(0.25*h)))
        x0=max(0,xx-rw//2); x1=min(w-1,xx+rw//2); y0=max(0,yy-rh//2); y1=min(h-1,yy+rh//2)
        m[y0:y1+1,x0:x1+1]=1; fallback='argmax_window'
    b=s3.mask_box(m); assert b is not None
    x0,y0,x1,y1=b; mx=int(round(margin*w)); my=int(round(margin*h))
    return m,(max(0,x0-mx),max(0,y0-my),min(w-1,x1+mx),min(h-1,y1+my)),fallback

def scale_box(b, src_size:int, h:int, w:int):
    x0,y0,x1,y1=b; sx=w/src_size; sy=h/src_size
    return (max(0,int(round(x0*sx))),max(0,int(round(y0*sy))),min(w-1,int(round((x1+1)*sx)-1)),min(h-1,int(round((y1+1)*sy)-1)))

# -------------------- validation prompt calibration -----------------------
@torch.no_grad()
def calibrate_prompt(val_pairs, proposer, prop_size, sam_b, device, out_root, amp):
    out=out_root/'calibration'; selected=out/'selected_prompt_params.json'
    if selected.exists():
        cfg=json.loads(selected.read_text(encoding='utf-8')); log(f'[SKIP] prompt calibration: {cfg}'); return cfg
    log('\n'+'='*78); log('STAGE5-A - VALIDATION-ONLY DOWNSTREAM PROMPT CALIBRATION'); log('='*78)
    grid=[]
    for mode in ('largest','top3_union'):
        for th in (0.15,0.25,0.35):
            for mg in (0.05,0.10,0.15,0.20,0.25):
                grid.append({'component_mode':mode,'threshold':th,'margin_fraction':mg,'dice':[],'cov':[],'biou':[],'area':[],'fallback':0})
    for i,(ip,mp) in enumerate(val_pairs,1):
        img=s3.read_rgb(ip); gt=s3.read_mask(mp); h,w=gt.shape
        pp=s4.proposer_prob(proposer,img,prop_size,device,amp)
        emb=s3.encode_image(sam_b,img,device)
        gt_small=cv2.resize(gt,(prop_size,prop_size),interpolation=cv2.INTER_NEAREST).astype(np.uint8)
        for g in grid:
            _,bs,fb=box_from_prob(pp,g['threshold'],g['margin_fraction'],g['component_mode'])
            z=s4.box_metrics(bs,gt_small); b=scale_box(bs,prop_size,h,w)
            prob=s3.decode_box_probability(sam_b,emb,b,h,w); pred=(prob>0.5).astype(np.uint8)
            g['dice'].append(core.dice_score(pred,gt)); g['cov'].append(z['lesion_coverage']); g['biou'].append(z['box_iou']); g['area'].append(z['box_area_ratio']); g['fallback']+=int(fb!='none')
        del emb
        if i==1 or i%50==0 or i==len(val_pairs): log(f'prompt calibration {i}/{len(val_pairs)}')
    rows=[]
    for g in grid:
        rows.append({'component_mode':g['component_mode'],'threshold':g['threshold'],'margin_fraction':g['margin_fraction'],
                     'validation_medsam_dice':mean(g['dice']),'mean_lesion_coverage':mean(g['cov']),
                     'mean_box_iou':mean(g['biou']),'mean_box_area_ratio':mean(g['area']),
                     'fallback_rate':g['fallback']/len(val_pairs)})
    write_csv(out/'prompt_grid.csv',rows)
    best_d=max(r['validation_medsam_dice'] for r in rows)
    near=[r for r in rows if r['validation_medsam_dice']>=best_d-0.002]
    best=max(near,key=lambda r:(r['mean_lesion_coverage'],r['mean_box_iou'],-r['mean_box_area_ratio']))
    cfg={**best,'selection_rule':'maximize validation MedSAM Dice; within 0.002 Dice of optimum choose highest lesion coverage, then box IoU'}
    write_json(selected,cfg); log(f'Selected prompt config: {cfg}'); return cfg

# -------------------------- USR calibration -------------------------------
def jitter_predictions(sam, emb, box, h,w, jitter_ref, seed):
    cfg=core.USRConfig(num_passes=8,jitter_px_ref=int(jitter_ref),reference_resolution=1024,
                       scale_spatial_params=True,aggregation_mode='soft',normalization_mode='none',seed=seed)
    eng=core.USR(cfg); rng=np.random.default_rng(seed); preds=[]
    for _ in range(cfg.num_passes):
        jb=eng.jitter_box(box,h,w,rng); preds.append(s3.decode_box_probability(sam,emb,jb,h,w))
    return eng.aggregate(preds)

def candidate_usr_cfg(seed,jitter,prune_p,prune_u,repair_p):
    return core.USRConfig(num_passes=8,jitter_px_ref=int(jitter),vote_threshold=0.5,
        inner_kernel_ref=3,outer_kernel_ref=9,prune_prob_threshold=float(prune_p),
        prune_uncertainty_threshold=float(prune_u),local_support_threshold=0.80,
        repair_prob_threshold=float(repair_p),max_area_drop=0.35,max_area_rise=0.25,
        max_component_increase=1,max_hole_increase=1,min_lcc_ratio=0.50,
        min_object_area_ref=100,max_hole_area_ref=100,reference_resolution=1024,
        scale_spatial_params=True,aggregation_mode='soft',normalization_mode='none',seed=seed)

@torch.no_grad()
def calibrate_usr(val_pairs, proposer, prop_size, prompt_cfg, sam_b, device, out_root, amp, seed):
    out=out_root/'calibration'; selected=out/'selected_usr_params.json'
    if selected.exists():
        cfg=json.loads(selected.read_text(encoding='utf-8')); log(f'[SKIP] USR calibration: {cfg}'); return cfg
    log('\n'+'='*78); log('STAGE5-B - VALIDATION-ONLY USR CALIBRATION'); log('='*78)
    specs=[]
    for jitter in (5,10):
        for prune_p in (0.55,0.60,0.65):
            for prune_u in (0.35,0.50):
                for repair_p in (0.70,0.85,0.95):
                    specs.append((jitter,prune_p,prune_u,repair_p))
    stats={s:{'dice':[],'bdice':[],'worse':[],'bdworse':[]} for s in specs}
    base_d=[]; base_bd=[]
    for i,(ip,mp) in enumerate(val_pairs,1):
        img=s3.read_rgb(ip); gt=s3.read_mask(mp); h,w=gt.shape
        pp=s4.proposer_prob(proposer,img,prop_size,device,amp)
        _,bs,_=box_from_prob(pp,float(prompt_cfg['threshold']),float(prompt_cfg['margin_fraction']),prompt_cfg['component_mode']); box=scale_box(bs,prop_size,h,w)
        emb=s3.encode_image(sam_b,img,device)
        pb=s3.decode_box_probability(sam_b,emb,box,h,w); mb=(pb>0.5).astype(np.uint8)
        bd0=core.boundary_dice(mb,gt); d0=core.dice_score(mb,gt); base_d.append(d0); base_bd.append(bd0)
        agg={}
        for jitter in (5,10): agg[jitter]=jitter_predictions(sam_b,emb,box,h,w,jitter,seed+i)
        for spec in specs:
            jitter,prune_p,prune_u,repair_p=spec; pbar,m0,u=agg[jitter]
            cfg=candidate_usr_cfg(seed,jitter,prune_p,prune_u,repair_p); r=core.USR(cfg).rectify(pbar,u,m0)
            d=core.dice_score(r.rectified_mask,gt); bd=core.boundary_dice(r.rectified_mask,gt)
            st=stats[spec]; st['dice'].append(d); st['bdice'].append(bd); st['worse'].append(d<d0); st['bdworse'].append(bd<bd0)
        del emb
        if i==1 or i%50==0 or i==len(val_pairs): log(f'USR calibration {i}/{len(val_pairs)}')
    b_d=mean(base_d); b_bd=mean(base_bd); rows=[]
    for spec in specs:
        jitter,prune_p,prune_u,repair_p=spec; st=stats[spec]
        rows.append({'jitter_px_at_1024':jitter,'prune_prob_threshold':prune_p,'prune_uncertainty_threshold':prune_u,'repair_prob_threshold':repair_p,
          'validation_dice':mean(st['dice']),'validation_boundary_dice':mean(st['bdice']),
          'dice_delta_vs_single':mean(st['dice'])-b_d,'boundary_dice_delta_vs_single':mean(st['bdice'])-b_bd,
          'dice_worsen_rate':float(np.mean(st['worse'])),'boundary_dice_worsen_rate':float(np.mean(st['bdworse'])),
          'single_pass_validation_dice':b_d,'single_pass_validation_boundary_dice':b_bd})
    write_csv(out/'usr_grid.csv',rows)
    feasible=[r for r in rows if r['validation_dice'] >= b_d-0.002]
    if feasible:
        best=max(feasible,key=lambda r:(r['validation_boundary_dice'],-r['dice_worsen_rate'],-r['boundary_dice_worsen_rate'],r['validation_dice']))
        rule='among configs within 0.002 Dice of single-pass baseline, maximize validation Boundary Dice; tie-break by lower worsen rates'
    else:
        best=max(rows,key=lambda r:(r['validation_dice'],r['validation_boundary_dice'],-r['dice_worsen_rate']))
        rule='fallback: no config met Dice non-inferiority; choose highest validation Dice then Boundary Dice'
    cfg={**best,'reference_resolution':1024,'T':8,'aggregation_mode':'soft','normalization_mode':'none','scale_spatial_params':True,'selection_rule':rule}
    write_json(selected,cfg); log(f'Selected USR config: {cfg}'); return cfg

# ------------------------------- test -------------------------------------
def save_mask(p,m): p.parent.mkdir(parents=True,exist_ok=True); cv2.imwrite(str(p),(m.astype(np.uint8)*255))
def save_map(p,x): p.parent.mkdir(parents=True,exist_ok=True); np.save(p,x.astype(np.float32))

def usr_run(sam,emb,box,h,w,cfgd,seed):
    cfg=candidate_usr_cfg(seed,int(cfgd['jitter_px_at_1024']),float(cfgd['prune_prob_threshold']),float(cfgd['prune_uncertainty_threshold']),float(cfgd['repair_prob_threshold']))
    eng=core.USR(cfg); rng=np.random.default_rng(seed); preds=[]
    for _ in range(cfg.num_passes):
        jb=eng.jitter_box(box,h,w,rng); preds.append(s3.decode_box_probability(sam,emb,jb,h,w))
    pbar,m0,u=eng.aggregate(preds); r=eng.rectify(pbar,u,m0); r.prediction_box=box; return r

@torch.no_grad()
def infer_dataset(ds,img_dir,mask_dir,proposer,prop_size,prompt_cfg,usr_cfg,sam_b,sam_a,device,out_root,amp,seed):
    pairs=s3.pair_by_stem(img_dir,mask_dir); out=out_root/'predictions'/ds
    dirs={}
    for v in ('baseline','abloss','usr','both'):
        for kind in ('masks','prob','unc'):
            dirs[v,kind]=out/v/kind; dirs[v,kind].mkdir(parents=True,exist_ok=True)
    prompt_csv=out/'prompt_protocol.csv'; lat_csv=out/'latency.csv'; diag_csv=out/'proposal_test_diagnostics.csv'
    log(f'\n===== STAGE5 CALIBRATED INFER {ds}: {len(pairs)} cases =====')
    for i,(ip,mp) in enumerate(pairs,1):
        stem=ip.stem
        if all((dirs[v,'masks']/f'{stem}.png').exists() for v in ('baseline','abloss','usr','both')):
            if i==1 or i%50==0 or i==len(pairs): log(f'{ds} {i}/{len(pairs)} [skip]')
            continue
        img=s3.read_rgb(ip); h,w=img.shape[:2]; t0=time.perf_counter()
        pp=s4.proposer_prob(proposer,img,prop_size,device,amp)
        pm_s,bs,fb=box_from_prob(pp,float(prompt_cfg['threshold']),float(prompt_cfg['margin_fraction']),prompt_cfg['component_mode']); box=scale_box(bs,prop_size,h,w)
        t_prop=time.perf_counter()-t0
        eb=s3.encode_image(sam_b,img,device); ea=s3.encode_image(sam_a,img,device)
        pb=s3.decode_box_probability(sam_b,eb,box,h,w); pa=s3.decode_box_probability(sam_a,ea,box,h,w)
        mb=(pb>0.5).astype(np.uint8); ma=(pa>0.5).astype(np.uint8)
        tu=time.perf_counter(); rb=usr_run(sam_b,eb,box,h,w,usr_cfg,seed+i); t_ub=time.perf_counter()-tu
        tu=time.perf_counter(); ra=usr_run(sam_a,ea,box,h,w,usr_cfg,seed+i); t_ua=time.perf_counter()-tu
        save_mask(dirs['baseline','masks']/f'{stem}.png',mb); save_mask(dirs['abloss','masks']/f'{stem}.png',ma)
        save_mask(dirs['usr','masks']/f'{stem}.png',rb.rectified_mask); save_mask(dirs['both','masks']/f'{stem}.png',ra.rectified_mask)
        save_map(dirs['baseline','prob']/f'{stem}.npy',pb); save_map(dirs['abloss','prob']/f'{stem}.npy',pa)
        save_map(dirs['usr','prob']/f'{stem}.npy',rb.aggregated_probability); save_map(dirs['both','prob']/f'{stem}.npy',ra.aggregated_probability)
        save_map(dirs['baseline','unc']/f'{stem}.npy',core.USR.bernoulli_entropy(pb)); save_map(dirs['abloss','unc']/f'{stem}.npy',core.USR.bernoulli_entropy(pa))
        save_map(dirs['usr','unc']/f'{stem}.npy',rb.uncertainty); save_map(dirs['both','unc']/f'{stem}.npy',ra.uncertainty)
        # GT only after all predictions are fixed.
        gt=s3.read_mask(mp); gt_s=cv2.resize(gt,(prop_size,prop_size),interpolation=cv2.INTER_NEAREST).astype(np.uint8); bm=s4.box_metrics(bs,gt_s)
        append_csv(diag_csv,{'case':stem,'component_mode':prompt_cfg['component_mode'],'threshold':prompt_cfg['threshold'],'margin_fraction':prompt_cfg['margin_fraction'],'fallback':fb,**bm})
        append_csv(prompt_csv,{'case':stem,'test_gt_prompt_used':0,'proposal_source':'prompt_free_UNet_image_only','same_box_for_baseline_and_abloss':1,'shared_prompt_box':str(box),'prompt_selected_on':'580-val downstream MedSAM Dice','usr_selected_on':'580-val only','usr_T':8,'usr_reference_resolution':1024,'usr_jitter_px_at_1024':usr_cfg['jitter_px_at_1024'],'usr_prune_prob':usr_cfg['prune_prob_threshold'],'usr_prune_unc':usr_cfg['prune_uncertainty_threshold'],'usr_repair_prob':usr_cfg['repair_prob_threshold']})
        append_csv(lat_csv,{'case':stem,'proposal_s':t_prop,'usr_baseline_s':t_ub,'usr_abloss_s':t_ua,'total_case_s':time.perf_counter()-t0})
        if i==1 or i%10==0 or i==len(pairs): log(f'{ds} {i}/{len(pairs)} box={box} fallback={fb} total={time.perf_counter()-t0:.2f}s')
        del eb,ea
        if i%20==0: torch.cuda.empty_cache()

def parser():
    p=argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--root',default=str(Path(os.environ.get('WYSIWYR_DATA_ROOT','.'))/'wysiwyr_real'),
                   help='Data/artefact root; defaults to $WYSIWYR_DATA_ROOT/wysiwyr_real'); p.add_argument('--seed',type=int,default=2023)
    p.add_argument('--amp',action=argparse.BooleanOptionalAction,default=True); p.add_argument('--datasets',default='')
    p.add_argument('--calibrate-only',action='store_true'); return p

def main():
    args=parser().parse_args(); seed_all(args.seed)
    if not torch.cuda.is_available(): raise SystemExit('CUDA GPU is required')
    device=torch.device('cuda:0'); root=Path(args.root).resolve(); out_root=root/'stage5_calibrated'; out_root.mkdir(parents=True,exist_ok=True)
    train_dir,test_dir,base_ckpt,b_ckpt,a_ckpt=s4.require_stage3_assets(root); medsam_repo=root/'third_party'/'MedSAM'
    train_pairs,val_pairs=s3.make_split_manifest(train_dir,root/'manifests',args.seed); tests=s3.discover_test_sets(test_dir)
    sel={x.strip() for x in args.datasets.split(',') if x.strip()};
    if sel: tests={k:v for k,v in tests.items() if k in sel}
    prop_ckpt=root/'stage4_promptfree'/'proposal'/'checkpoints'/'best.pt'
    if not prop_ckpt.exists(): raise SystemExit(f'Missing Stage4 proposer checkpoint: {prop_ckpt}')
    proposer,prop_size=s4.load_proposer(prop_ckpt,device)
    log('\n'+'='*78); log('WYSIWYR STAGE5 - VALIDATION-CALIBRATED PROMPT + USR'); log('='*78)
    log(f'GPU: {torch.cuda.get_device_name(0)} | train={len(train_pairs)} val={len(val_pairs)}')
    sam_b=s3.load_sam_state(base_ckpt,b_ckpt,medsam_repo,device); sam_a=s3.load_sam_state(base_ckpt,a_ckpt,medsam_repo,device)
    prompt_cfg=calibrate_prompt(val_pairs,proposer,prop_size,sam_b,device,out_root,args.amp)
    usr_cfg=calibrate_usr(val_pairs,proposer,prop_size,prompt_cfg,sam_b,device,out_root,args.amp,args.seed)
    protocol={'stage':'Stage5 validation-calibrated','seed':args.seed,'train_n':len(train_pairs),'val_n':len(val_pairs),
      'test_gt_prompt_used':False,'prompt':prompt_cfg,'usr':usr_cfg,
      'scientific_notes':['prompt hyperparameters selected only on fixed validation split','USR parameters selected only on fixed validation split','USR pixel lengths/areas referenced to 1024-px MedSAM canonical resolution','soft probability aggregation; no per-image min-max normalization','same prediction-derived box used for baseline and ABLoss']}
    write_json(out_root/'protocol_stage5.json',protocol)
    if args.calibrate_only:
        log('\nCALIBRATION COMPLETE'); return
    script_dir=Path(__file__).resolve().parent
    for ds,(img_dir,mask_dir) in tests.items():
        infer_dataset(ds,img_dir,mask_dir,proposer,prop_size,prompt_cfg,usr_cfg,sam_b,sam_a,device,out_root,args.amp,args.seed)
        s3.run_stage2_analysis(ds,img_dir,mask_dir,out_root,script_dir,args.seed)
    write_json(out_root/'RUN_COMPLETE.json',{'completed_at':time.strftime('%Y-%m-%d %H:%M:%S'),'datasets':list(tests),'test_gt_prompt_used':False,'results_root':str(out_root/'reviewer2_results')})
    log('\n'+'='*78); log('STAGE5 COMPLETE'); log(f'Calibration: {out_root/"calibration"}'); log(f'Results: {out_root/"reviewer2_results"}'); log('='*78)

if __name__=='__main__': main()
