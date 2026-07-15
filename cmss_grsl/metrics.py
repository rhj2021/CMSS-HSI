from __future__ import annotations
import math
from typing import Any
import numpy as np
from scipy.ndimage import binary_dilation,binary_erosion
from sklearn.metrics import average_precision_score,roc_auc_score

def boundary_map(m):
    m=np.asarray(m,bool); return m^binary_erosion(m,structure=np.ones((3,3),bool),border_value=0)
def boundary_f1(pred,gt,mask=None,tolerance=2):
    valid=np.asarray(gt)>=0
    if mask is not None: valid&=np.asarray(mask,bool)
    p=np.asarray(pred,bool)&valid; y=(np.asarray(gt)==1)&valid; pb,gb=boundary_map(p),boundary_map(y)
    st=np.ones((2*tolerance+1,2*tolerance+1),bool); pbd=binary_dilation(pb,structure=st); gbd=binary_dilation(gb,structure=st)
    pr=np.sum(pb&gbd)/max(np.sum(pb),1); rc=np.sum(gb&pbd)/max(np.sum(gb),1); return float(2*pr*rc/max(pr+rc,1e-12))
def binary_metrics(pred,gt,mask=None,prob=None,boundary_tolerance=2)->dict[str,Any]:
    valid=np.asarray(gt)>=0
    if mask is not None: valid&=np.asarray(mask,bool)
    y=np.asarray(gt)[valid].astype(bool); p=np.asarray(pred)[valid].astype(bool)
    tp=int(np.sum(p&y)); tn=int(np.sum(~p&~y)); fp=int(np.sum(p&~y)); fn=int(np.sum(~p&y)); n=tp+tn+fp+fn
    oa=(tp+tn)/max(n,1); precision=tp/max(tp+fp,1); recall=tp/max(tp+fn,1); spec=tn/max(tn+fp,1); f1=2*precision*recall/max(precision+recall,1e-12); iou=tp/max(tp+fp+fn,1); pe=((tp+fp)*(tp+fn)+(fn+tn)*(fp+tn))/max(n*n,1); k=(oa-pe)/max(1-pe,1e-12)
    out={'OA':float(oa),'Precision':float(precision),'Recall':float(recall),'Specificity':float(spec),'F1':float(f1),'IoU':float(iou),'BalancedAccuracy':float(.5*(recall+spec)),'Kappa':float(k),'BoundaryF1':boundary_f1(pred,gt,mask,boundary_tolerance),'TP':tp,'TN':tn,'FP':fp,'FN':fn,'N':n}
    if prob is not None and len(np.unique(y))==2:
        s=np.asarray(prob)[valid]
        try: out['AUROC']=float(roc_auc_score(y,s)); out['AUPRC']=float(average_precision_score(y,s))
        except ValueError: out['AUROC']=out['AUPRC']=math.nan
    else: out['AUROC']=out['AUPRC']=math.nan
    return out
def find_best_f1_threshold(prob,gt,mask,grid_size=401):
    valid=(np.asarray(gt)>=0)&np.asarray(mask,bool); y=np.asarray(gt)[valid].astype(np.uint8); s=np.asarray(prob)[valid].astype(float); f=np.isfinite(s); y,s=y[f],s[f]
    if not len(y) or len(np.unique(y))<2: return .5,0.
    ts=np.unique(np.concatenate([[0,.5,1],np.quantile(s,np.linspace(0,1,grid_size))])); best=(.5,-1.)
    for t in ts:
        p=s>=t; tp=np.sum(p&(y==1)); fp=np.sum(p&(y==0)); fn=np.sum(~p&(y==1)); f1=2*tp/max(2*tp+fp+fn,1)
        if f1>best[1]: best=(float(t),float(f1))
    return best
def safe_gap_recovery(clean,degraded,corrected,eps=1e-8):
    g=clean-degraded; return float('nan') if g<=eps else float((corrected-degraded)/g)
def matching_metrics(pred_flow,true_flow,gt,valid_mask,best_distance=None,confidence_margin=None):
    e=np.sqrt(np.sum((np.asarray(pred_flow,float)-np.asarray(true_flow,float))**2,axis=-1)); valid=np.asarray(valid_mask,bool)&(np.asarray(gt)>=0); un=valid&(gt==0); ch=valid&(gt==1)
    out={}
    for m,n in [(valid,'all'),(un,'unchanged'),(ch,'changed')]:
        v=e[m]; out.update({f'EPE_{n}_mean':float(np.mean(v)) if v.size else math.nan,f'EPE_{n}_median':float(np.median(v)) if v.size else math.nan,f'Acc1_{n}':float(np.mean(v<=1)) if v.size else math.nan,f'Acc2_{n}':float(np.mean(v<=2)) if v.size else math.nan,f'N_{n}':int(v.size)})
    if best_distance is not None:
        d=np.asarray(best_distance,float); out['MatchDistance_unchanged_mean']=float(np.mean(d[un])) if un.any() else math.nan; out['MatchDistance_changed_mean']=float(np.mean(d[ch])) if ch.any() else math.nan
        y=np.asarray(gt)[valid]; s=d[valid]
        if len(np.unique(y))==2: out['ChangeAUROC_from_distance']=float(roc_auc_score(y,s)); out['ChangeAUPRC_from_distance']=float(average_precision_score(y,s))
    if confidence_margin is not None:
        c=np.asarray(confidence_margin,float); out['ConfidenceMargin_unchanged_mean']=float(np.mean(c[un])) if un.any() else math.nan; out['ConfidenceMargin_changed_mean']=float(np.mean(c[ch])) if ch.any() else math.nan
    return out
