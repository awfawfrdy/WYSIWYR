from __future__ import annotations
import os
DATA_ROOT = os.environ.get("WYSIWYR_DATA_ROOT", ".")

import argparse,csv,json
from collections import Counter
from pathlib import Path

def read(p):
    with open(p,newline='',encoding='utf-8-sig') as f:return list(csv.DictReader(f))

def prf(y_true,y_pred,positive):
    tp=sum(t==positive and p==positive for t,p in zip(y_true,y_pred)); fp=sum(t!=positive and p==positive for t,p in zip(y_true,y_pred)); fn=sum(t==positive and p!=positive for t,p in zip(y_true,y_pred))
    prec=tp/(tp+fp) if tp+fp else 0.0; rec=tp/(tp+fn) if tp+fn else 0.0; f1=2*prec*rec/(prec+rec) if prec+rec else 0.0
    return {'tp':tp,'fp':fp,'fn':fn,'precision':prec,'recall':rec,'f1':f1}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--human',required=True); ap.add_argument('--key',default=DATA_ROOT + '/wysiwyr_real/stage6_checker_v3/checker_v3_human_validation_KEY_DO_NOT_SHOW_ANNOTATOR.csv'); ap.add_argument('--output',default=DATA_ROOT + '/wysiwyr_real/stage6_checker_v3/human_validation_scores.json'); args=ap.parse_args()
    h=read(args.human); k={r['annotation_id']:r for r in read(args.key)}
    rows=[]
    for r in h:
        if not r.get('human_status','').strip(): continue
        q=k.get(r['annotation_id']);
        if q: rows.append((r,q))
    if not rows: raise SystemExit('No completed human labels found')
    yt=[r['human_status'].strip() for r,q in rows]; yp=[q['checker_v3_status'].strip() for r,q in rows]
    classes=['supported','calibrated','unsupported','prohibited']
    per={c:prf(yt,yp,c) for c in classes}
    macro={m:sum(per[c][m] for c in classes)/len(classes) for m in ['precision','recall','f1']}
    acc=sum(t==p for t,p in zip(yt,yp))/len(yt)
    # Binary violation classification is the main checker endpoint.
    tv=['violation' if x!='supported' else 'supported' for x in yt]; pv=['violation' if x!='supported' else 'supported' for x in yp]
    binary=prf(tv,pv,'violation')
    out={'n':len(rows),'accuracy_status4':acc,'macro_status4':macro,'per_status':per,'binary_violation':binary,'human_counts':Counter(yt),'checker_counts':Counter(yp)}
    Path(args.output).write_text(json.dumps(out,indent=2,default=dict),encoding='utf-8')
    print(json.dumps(out,indent=2,default=dict)); print('WROTE',args.output)
if __name__=='__main__':main()
