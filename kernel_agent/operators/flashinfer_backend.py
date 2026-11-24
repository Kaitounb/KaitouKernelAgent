"""Reviewed fa2 CUDA SDK adapters. No score/probability materialization in core paths."""
import math
import torch


def _version():
    import flashinfer
    if flashinfer.__version__!='0.2.8':raise RuntimeError('FlashInfer 0.2.8 required')
    return flashinfer


def attention(q,k,v,*,causal=False,scale=None,mask=None):
    """FP16 contiguous [B,S,H,D], bottom-right causal alignment; native output FP16."""
    sdk=_version()
    if not all(x.is_cuda and x.dtype==torch.float16 and x.is_contiguous() and x.ndim==4 and x.data_ptr()%16==0 for x in [q,k,v]):raise ValueError('16-byte-aligned CUDA contiguous FP16 BSHD required')
    if q.device!=k.device or q.device!=v.device or k.shape[:3]!=v.shape[:3] or q.shape[0]!=k.shape[0] or q.shape[-1]!=k.shape[-1] or q.shape[-1]!=v.shape[-1] or q.shape[2]%k.shape[2]:raise ValueError('attention shape/device mismatch')
    if q.shape[-1] not in [64,128] or not (0<q.shape[1]<=k.shape[1]):raise ValueError('unsupported head/query dimensions')
    if mask is not None and (mask.dtype!=torch.bool or mask.device!=q.device or mask.shape!=(q.shape[1],k.shape[1])):raise ValueError('shared boolean SQxSK device mask required')
    if mask is not None and causal:
        sq,sk=q.shape[1],k.shape[1];visible=torch.arange(sk,device=q.device)[None,:]<=torch.arange(sq,device=q.device)[:,None]+sk-sq;mask=mask&visible
    output=torch.stack([sdk.single_prefill_with_kv_cache(q[i],k[i],v[i],causal=causal,sm_scale=scale,custom_mask=mask,backend='fa2') for i in range(q.shape[0])])
    # Force the empty-row identity even if a native implementation yields NaN.
    if mask is not None:output=torch.where(mask.any(dim=-1)[None,:,None,None],output,0.)
    return output


class MLAPlan:
    """Single-token classic MLA Z/LSE, using SDK paging and split planner.

    Metadata is trusted host input, validated before native planning. Planning and
    workspace allocation are excluded from run-only timing and reported separately.
    """
    def __init__(self,*,lengths,indptr,indices,num_pages,page_size,heads,scale,device,workspace_bytes=128*1024*1024):
        _version();from flashinfer.mla import BatchMLAPagedAttentionWrapper
        if any(type(x) is not int for x in [*lengths,*indptr,*indices,num_pages,page_size,heads]):raise ValueError('integer host metadata required')
        self.lengths=list(lengths);self.device=torch.device(device);self.heads=heads;self.num_pages=num_pages;self.page_size=page_size
        if self.device.type!='cuda':raise ValueError('CUDA device required')
        if self.device.index is None:self.device=torch.device('cuda',torch.cuda.current_device())
        if not 1<=len(self.lengths)<=16 or any(not 0<=n<=8192 for n in self.lengths) or page_size not in [1,16,64] or heads not in [16,32,64,128] or not math.isfinite(scale) or scale<=0:raise ValueError('unsupported MLA contract')
        if len(indptr)!=len(lengths)+1 or indptr[0]!=0 or indptr[-1]!=len(indices) or any(not 0<=int(x)<num_pages for x in indices):raise ValueError('invalid page table')
        if any(indptr[i+1]-indptr[i]!=(n+page_size-1)//page_size for i,n in enumerate(self.lengths)):raise ValueError('page count/tail mismatch')
        if len(set(indices))!=len(indices):raise ValueError('first release requires nonoverlapping request pages')
        self.active=[i for i,n in enumerate(self.lengths) if n];self.wrapper=None;self.workspace=None
        if not self.active:return
        active_ptr=[0];active_indices=[]
        for i in self.active:active_indices.extend(indices[indptr[i]:indptr[i+1]]);active_ptr.append(len(active_indices))
        self.active_gpu=torch.tensor(self.active,dtype=torch.long,device=self.device)
        self.workspace=torch.empty(workspace_bytes,dtype=torch.uint8,device=self.device);self.wrapper=BatchMLAPagedAttentionWrapper(self.workspace,backend='fa2')
        self.wrapper.plan(torch.arange(len(self.active)+1,dtype=torch.int32),torch.tensor(active_ptr,dtype=torch.int32),torch.tensor(active_indices,dtype=torch.int32),torch.tensor([self.lengths[i] for i in self.active],dtype=torch.int32),heads,512,64,page_size,False,scale,torch.float16,torch.float16)

    def run(self,q_abs,q_rope,cache,k_rope):
        expected=[(len(self.lengths),self.heads,512),(len(self.lengths),self.heads,64),(self.num_pages,self.page_size,512),(self.num_pages,self.page_size,64)]
        for value,shape in zip([q_abs,q_rope,cache,k_rope],expected):
            if tuple(value.shape)!=shape or value.device!=self.device or value.dtype!=torch.float16 or not value.is_contiguous() or value.data_ptr()%16:raise ValueError('MLA input shape/device/dtype/layout/alignment mismatch')
        z=torch.zeros_like(q_abs);lse=torch.full(q_abs.shape[:2],-torch.inf,dtype=torch.float32,device=self.device)
        if self.wrapper is not None:
            active_z,log2_lse=self.wrapper.run(q_abs.index_select(0,self.active_gpu),q_rope.index_select(0,self.active_gpu),cache,k_rope,return_lse=True)
            z.index_copy_(0,self.active_gpu,active_z);lse.index_copy_(0,self.active_gpu,log2_lse*math.log(2.))
        return z,lse


def rotary(q,k,positions):
    """Interleaved RoPE, token-major NHD tensors, identical positions for query/key."""
    _version();from flashinfer.rope import apply_rope_pos_ids
    if q.ndim!=3 or k.ndim!=3 or q.shape[0]!=k.shape[0] or q.shape[-1]!=k.shape[-1] or q.shape[-1]%2:raise ValueError('RoPE NHD shapes differ')
    if positions.shape!=(q.shape[0],) or positions.dtype not in [torch.int32,torch.int64]:raise ValueError('one integer position per token required')
    if any(x.device!=q.device for x in [k,positions]) or not q.is_cuda or q.dtype!=torch.float16 or k.dtype!=q.dtype or any(not x.is_contiguous() or x.data_ptr()%16 for x in [q,k]):raise ValueError('aligned contiguous CUDA FP16 RoPE required')
    return apply_rope_pos_ids(q,k,positions,interleave=True)


def merge_partials(left,right,*,left_lengths,right_lengths,prepared_indices=None):
    """Combine native FP16 partial Z and natural LSE with empty-row identity.

    Calls the SDK CUDA merger only for rows with two nonempty segments; its LSE
    convention is log2. Empty states are never passed to that primitive.
    """
    _version();from flashinfer.cascade import merge_state
    za,sa=left;zb,sb=right
    if any(not x.is_cuda or x.device!=za.device or not x.is_contiguous() for x in [za,sa,zb,sb]) or any(x.dtype!=torch.float16 or x.data_ptr()%16 for x in [za,zb]) or any(x.dtype!=torch.float32 for x in [sa,sb]):raise ValueError('aligned contiguous CUDA partials required')
    if za.shape!=zb.shape or sa.shape!=sb.shape or sa.shape!=za.shape[:2] or len(left_lengths)!=len(za) or len(right_lengths)!=len(za):raise ValueError('partial shapes differ')
    if any(type(x) is not int or x<0 for x in [*left_lengths,*right_lengths]):raise ValueError('nonnegative split lengths required')
    z=torch.zeros_like(za);s=torch.full_like(sa,-torch.inf)
    only_left=[i for i,(a,b) in enumerate(zip(left_lengths,right_lengths)) if a>0 and b==0];only_right=[i for i,(a,b) in enumerate(zip(left_lengths,right_lengths)) if a==0 and b>0];both=[i for i,(a,b) in enumerate(zip(left_lengths,right_lengths)) if a>0 and b>0]
    for key,rows,source,source_s in [('left',only_left,za,sa),('right',only_right,zb,sb)]:
        if rows:
            ids=prepared_indices[key] if prepared_indices is not None else torch.tensor(rows,device=z.device);z.index_copy_(0,ids,source.index_select(0,ids));s.index_copy_(0,ids,source_s.index_select(0,ids))
    if both:
        ids=prepared_indices['both'] if prepared_indices is not None else torch.tensor(both,device=z.device);a=za.index_select(0,ids);b=zb.index_select(0,ids);la=(sa.index_select(0,ids)/math.log(2.)).contiguous();lb=(sb.index_select(0,ids)/math.log(2.)).contiguous();shape=a.shape
        # Pinned SDK launches (32, num_heads) for R512. Group independent heads
        # into sequence rows so H64/H128 never exceed CUDA's 1024-thread limit.
        if shape[1]>32:a=a.view(-1,32,shape[2]);b=b.view(-1,32,shape[2]);la=la.view(-1,32);lb=lb.view(-1,32)
        merged,log2_s=merge_state(a,la,b,lb);merged=merged.view(shape);log2_s=log2_s.view(shape[:2]);z.index_copy_(0,ids,merged);s.index_copy_(0,ids,log2_s*math.log(2.))
    return z,s


def append_cache(cache,rope_cache,ckv,kpe,*,old_lengths,append_lengths,indptr,indices):
    """Append already-positioned keys through the pinned CUDA cache writer.

    Host metadata describes the final page table; caller preserves old page mappings.
    Input rows are grouped by request. This function does not apply RoPE implicitly.
    """
    _version();from flashinfer.page import append_paged_mla_kv_cache
    if any(type(x) is not int or x<0 for x in [*old_lengths,*append_lengths,*indptr,*indices]):raise ValueError('nonnegative integer metadata required')
    if len(old_lengths)!=len(append_lengths) or len(indptr)!=len(old_lengths)+1 or not indptr or indptr[0]!=0 or indptr[-1]!=len(indices):raise ValueError('append metadata lengths differ')
    if cache.ndim!=3 or cache.shape[-1]!=512 or rope_cache.shape!=(*cache.shape[:2],64) or ckv.shape!=(sum(append_lengths),512) or kpe.shape!=(sum(append_lengths),64):raise ValueError('append tensor shape mismatch')
    if any(not x.is_cuda or x.device!=cache.device or x.dtype!=torch.float16 or not x.is_contiguous() or x.data_ptr()%16 for x in [cache,rope_cache,ckv,kpe]):raise ValueError('aligned CUDA contiguous FP16 append required')
    page_size=cache.shape[1];lengths=[a+b for a,b in zip(old_lengths,append_lengths)]
    if page_size not in [1,16,64] or not 1<=len(lengths)<=16 or any(n>8192 for n in lengths):raise ValueError('append outside MLA contract')
    if any(indptr[i+1]-indptr[i]!=(n+page_size-1)//page_size for i,n in enumerate(lengths)) or any(i>=cache.shape[0] for i in indices) or len(set(indices))!=len(indices):raise ValueError('append page bounds or overlap invalid')
    if not sum(append_lengths):return lengths
    batch=[];positions=[]
    for i,(old,n) in enumerate(zip(old_lengths,append_lengths)):batch.extend([i]*n);positions.extend(range(old,old+n))
    def device_int(x):return torch.tensor(x,dtype=torch.int32,device=cache.device)
    last=[(n-1)%page_size+1 if n else 0 for n in lengths]
    append_paged_mla_kv_cache(ckv,kpe,device_int(batch),device_int(positions),cache,rope_cache,device_int(indices),device_int(indptr),device_int(last))
    return lengths


class SplitMLAPlan:
    """Controlled native-CUDA graph transform; all partials and combine use the SDK."""
    def __init__(self,*,lengths,indptr,indices,num_pages,page_size,heads,scale,device,splits=1):
        from .partition import split_page_table
        rows=split_page_table(lengths,indptr,indices,page_size,splits);self.rows=rows;self.splits=splits
        # Validate the complete input once as well as each partial table; metadata
        # checks happen before planning any native partial.
        if not 1<=len(lengths)<=16 or any(n>8192 for n in lengths) or len(set(indices))!=len(indices) or any(i>=num_pages for i in indices):raise ValueError('unsupported complete MLA table')
        self.plans=[MLAPlan(**row,num_pages=num_pages,page_size=page_size,heads=heads,scale=scale,device=device) for row in rows]
        self.merge_indices=[];accumulated=rows[0]['lengths'][:]
        for row in rows[1:]:
            pairs=list(zip(accumulated,row['lengths']));groups={'left':[i for i,(a,b) in enumerate(pairs) if a>0 and b==0],'right':[i for i,(a,b) in enumerate(pairs) if a==0 and b>0],'both':[i for i,(a,b) in enumerate(pairs) if a>0 and b>0]};self.merge_indices.append({name:torch.tensor(ids,dtype=torch.long,device=self.plans[0].device) for name,ids in groups.items()});accumulated=[a+b for a,b in pairs]

    def run(self,q_abs,q_rope,cache,k_rope):
        state=self.plans[0].run(q_abs,q_rope,cache,k_rope);lengths=self.rows[0]['lengths'][:]
        for plan,row,indices in zip(self.plans[1:],self.rows[1:],self.merge_indices):
            state=merge_partials(state,plan.run(q_abs,q_rope,cache,k_rope),left_lengths=lengths,right_lengths=row['lengths'],prepared_indices=indices);lengths=[a+b for a,b in zip(lengths,row['lengths'])]
        return state
