"""Trusted-controller FP64 references and online-state invariants. Never mounted in candidate workers."""
import math
import torch


def safe_softmax(scores,mask=None):
    scores=scores.double()
    if mask is not None:scores=scores.masked_fill(~mask,-torch.inf)
    maximum=scores.max(dim=-1,keepdim=True).values
    shifted=torch.where(torch.isfinite(maximum),scores-maximum,-torch.inf)
    weights=torch.exp(shifted);denominator=weights.sum(dim=-1,keepdim=True)
    return torch.where(denominator>0,weights/denominator.clamp_min(torch.finfo(torch.float64).tiny),torch.zeros_like(weights))


def attention(q,k,v,*,scale=None,causal=False,mask=None):
    # [B,H,S,D], causal alignment is bottom-right for incremental prefill/decode.
    scores=torch.einsum('bhsd,bhtd->bhst',q.double(),k.double())*(scale if scale is not None else q.shape[-1]**-.5)
    if causal:
        sq,sk=scores.shape[-2:];allowed=torch.arange(sk)[None,:]<=torch.arange(sq)[:,None]+sk-sq
        mask=allowed if mask is None else mask&allowed
    p=safe_softmax(scores,mask)
    return torch.einsum('bhst,bhtv->bhsv',p,v.double())


def block_state(scores,values):
    scores=scores.double();values=values.double()
    if scores.numel()==0:return (-torch.inf,torch.tensor(0.,dtype=torch.float64),torch.zeros(values.shape[-1],dtype=torch.float64))
    maximum=scores.max();weights=torch.where(torch.isfinite(maximum),torch.exp(scores-maximum),torch.zeros_like(scores))
    return maximum,weights.sum(),weights@values


def merge_states(left,right):
    m1,l1,a1=left;m2,l2,a2=right;m=torch.maximum(torch.as_tensor(m1),torch.as_tensor(m2))
    alpha=torch.exp(torch.as_tensor(m1)-m) if bool(torch.isfinite(m)) else torch.tensor(0.,dtype=torch.float64)
    beta=torch.exp(torch.as_tensor(m2)-m) if bool(torch.isfinite(m)) else torch.tensor(0.,dtype=torch.float64)
    return m,alpha*l1+beta*l2,alpha*a1+beta*a2


def online_attention(scores,values,block_size):
    state=block_state(scores[:0],values[:0])
    for start in range(0,len(scores),block_size):state=merge_states(state,block_state(scores[start:start+block_size],values[start:start+block_size]))
    m,l,a=state
    return (a/l if bool(l>0) else torch.zeros_like(a)),(m+torch.log(l) if bool(l>0) else torch.tensor(-torch.inf,dtype=torch.float64))


def rotary(x,positions,theta=10000.):
    x=x.double();d=x.shape[-1]
    if d%2:raise ValueError('even rotary dimension required')
    angle=positions.double()[...,None]*theta**(-torch.arange(0,d,2,dtype=torch.float64)/d);c,s=angle.cos(),angle.sin()
    while c.ndim<x.ndim:c=c.unsqueeze(-2);s=s.unsqueeze(-2)
    a,b=x[...,0::2],x[...,1::2];out=torch.empty_like(x);out[...,0::2]=a*c-b*s;out[...,1::2]=a*s+b*c;return out


def mla_core(q_abs,q_rope,c,k_rope,lengths,scale):
    # q_abs[B,H,R], cache[B,L,R], q_rope[B,H,DR]. Output is latent Z, natural-log LSE.
    b,h,r=q_abs.shape;out=torch.zeros((b,h,r),dtype=torch.float64);lse=torch.full((b,h),-torch.inf,dtype=torch.float64)
    for index,length in enumerate(lengths):
        length=int(length)
        if length==0:continue
        logits=(q_abs[index].double()@c[index,:length].double().T+q_rope[index].double()@k_rope[index,:length].double().T)*scale
        out[index]=safe_softmax(logits)@c[index,:length].double();lse[index]=torch.logsumexp(logits,dim=-1)
    return out,lse


def mla_layer_explicit(q_nope,q_rope,c,k_rope,u_k,u_v,lengths,scale):
    outputs=[]
    for i,length in enumerate(lengths):
        length=int(length)
        if length==0:outputs.append(torch.zeros((q_nope.shape[1],u_v.shape[-1]),dtype=torch.float64));continue
        keys=torch.einsum('lr,hrd->hld',c[i,:length].double(),u_k.double());values=torch.einsum('lr,hrv->hlv',c[i,:length].double(),u_v.double())
        logits=(torch.einsum('hd,hld->hl',q_nope[i].double(),keys)+q_rope[i].double()@k_rope[i,:length].double().T)*scale
        outputs.append(torch.einsum('hl,hlv->hv',safe_softmax(logits),values))
    return torch.stack(outputs)


def mla_layer_absorbed(q_nope,q_rope,c,k_rope,u_k,u_v,lengths,scale):
    q_abs=torch.einsum('bhd,hrd->bhr',q_nope.double(),u_k.double());z,lse=mla_core(q_abs,q_rope,c,k_rope,lengths,scale)
    return torch.einsum('bhr,hrv->bhv',z,u_v.double()),lse
