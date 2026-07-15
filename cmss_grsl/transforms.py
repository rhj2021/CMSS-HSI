from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import numpy as np, torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter
from .utils import resolve_device,stable_hash_array,stable_hash_dict
@dataclass
class TransformResult: name:str; moving:np.ndarray; forward_flow:np.ndarray; valid_reference:np.ndarray; metadata:dict[str,Any]
def make_forward_flow(spec,height,width,seed=0):
    kind=str(spec.get('type','translation')).lower(); yy,xx=np.meshgrid(np.arange(height,dtype=np.float32),np.arange(width,dtype=np.float32),indexing='ij'); f=np.zeros((height,width,2),np.float32)
    if kind in {'translation','shift'}: f[...,0]=float(spec.get('dy',0)); f[...,1]=float(spec.get('dx',0))
    elif kind=='affine':
        a=np.deg2rad(float(spec.get('rotation_deg',0))); s=float(spec.get('scale',1)); tx=float(spec.get('dx',0)); ty=float(spec.get('dy',0)); cy=(height-1)/2; cx=(width-1)/2; x0=xx-cx; y0=yy-cy; xq=s*(np.cos(a)*x0-np.sin(a)*y0)+cx+tx; yq=s*(np.sin(a)*x0+np.cos(a)*y0)+cy+ty; f[...,0]=yq-yy; f[...,1]=xq-xx
    elif kind in {'row_shift','line_scan'}:
        amp=float(spec.get('amplitude',4)); period=float(spec.get('period',max(height/2,8))); phase=float(spec.get('phase',0)); harm=float(spec.get('harmonic',.25)); row=yy[:,0]; shift=amp*np.sin(2*np.pi*row/period+phase)+harm*amp*np.sin(4*np.pi*row/period+.7+phase); f[...,1]=shift[:,None]
    elif kind=='elastic':
        rng=np.random.default_rng(seed); sig=float(spec.get('sigma',18)); md=float(spec.get('max_displacement',3)); fy=gaussian_filter(rng.normal(size=(height,width)).astype(np.float32),sig,mode='reflect'); fx=gaussian_filter(rng.normal(size=(height,width)).astype(np.float32),sig,mode='reflect'); mag=np.sqrt(fy*fy+fx*fx); scale=md/max(float(np.percentile(mag,99.5)),1e-6); f[...,0]=fy*scale; f[...,1]=fx*scale
    else: raise ValueError(kind)
    return f
def _grid(y,x,h,w): return torch.stack([2*x/max(w-1,1)-1,2*y/max(h-1,1)-1],dim=-1)
def _sample_flow(flow_t,y,x):
    _,_,h,w=flow_t.shape; g=_grid(y,x,h,w).unsqueeze(0); return F.grid_sample(flow_t,g,mode='bilinear',padding_mode='border',align_corners=True).squeeze(0).permute(1,2,0)
def sample_hwc(img,y,x,device='cpu'):
    img=np.asarray(img,np.float32); h,w,_=img.shape; dev=resolve_device(device); it=torch.from_numpy(img).permute(2,0,1).unsqueeze(0).to(dev); yt=torch.from_numpy(np.asarray(y,np.float32)).to(dev); xt=torch.from_numpy(np.asarray(x,np.float32)).to(dev); out=F.grid_sample(it,_grid(yt,xt,h,w).unsqueeze(0),mode='bilinear',padding_mode='zeros',align_corners=True); return out.squeeze(0).permute(1,2,0).cpu().numpy().astype(np.float32)
def warp_image_from_forward_flow(img,flow,device='cpu',inverse_iterations=12):
    img=np.asarray(img,np.float32); h,w,_=img.shape; dev=resolve_device(device); yy,xx=torch.meshgrid(torch.arange(h,dtype=torch.float32,device=dev),torch.arange(w,dtype=torch.float32,device=dev),indexing='ij'); py,px=yy.clone(),xx.clone(); ft=torch.from_numpy(np.asarray(flow,np.float32)).permute(2,0,1).unsqueeze(0).to(dev)
    with torch.no_grad():
        for _ in range(inverse_iterations): sf=_sample_flow(ft,py,px); py=yy-sf[...,0]; px=xx-sf[...,1]
        sf=_sample_flow(ft,py,px); res=torch.sqrt((py+sf[...,0]-yy)**2+(px+sf[...,1]-xx)**2); valid=(py>=0)&(py<=h-1)&(px>=0)&(px<=w-1); it=torch.from_numpy(img).permute(2,0,1).unsqueeze(0).to(dev); out=F.grid_sample(it,_grid(py,px,h,w).unsqueeze(0),mode='bilinear',padding_mode='zeros',align_corners=True)
    stats={'inverse_residual_mean':float(res[valid].mean().cpu()) if valid.any() else float('nan'),'inverse_residual_max':float(res[valid].max().cpu()) if valid.any() else float('nan'),'valid_output_fraction':float(valid.float().mean().cpu())}
    return out.squeeze(0).permute(1,2,0).cpu().numpy().astype(np.float32),valid.cpu().numpy(),stats
def oracle_realign(moving,flow,device='cpu'):
    h,w,_=moving.shape; yy,xx=np.meshgrid(np.arange(h,dtype=np.float32),np.arange(w,dtype=np.float32),indexing='ij'); qy=yy+flow[...,0]; qx=xx+flow[...,1]; valid=(qy>=0)&(qy<=h-1)&(qx>=0)&(qx<=w-1); return sample_hwc(moving,qy,qx,device),valid
def apply_transform(img,spec,device='cpu',seed=0):
    name=str(spec.get('name',spec.get('type','transform'))); h,w,_=img.shape; flow=make_forward_flow(spec,h,w,seed); moving,vout,st=warp_image_from_forward_flow(img,flow,device,int(spec.get('inverse_iterations',12))); yy,xx=np.meshgrid(np.arange(h),np.arange(w),indexing='ij'); qy=yy+flow[...,0]; qx=xx+flow[...,1]; valid=(qy>=0)&(qy<=h-1)&(qx>=0)&(qx<=w-1); mag=np.sqrt(np.sum(flow*flow,axis=-1)); meta={'name':name,'spec':dict(spec),'spec_hash':stable_hash_dict(dict(spec)),'flow_hash':stable_hash_array(flow),'max_displacement':float(mag.max()),'mean_displacement':float(mag.mean()),'valid_reference_fraction':float(valid.mean()),**st}; return TransformResult(name,moving,flow,valid,meta)
