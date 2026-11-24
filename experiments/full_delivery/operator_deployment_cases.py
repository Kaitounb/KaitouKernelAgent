"""Public fixed deployment anchors and deterministic FP16 inputs, no sealed workloads."""
import numpy as np

WORKLOADS=[
 {'id':'attention-prefill','operation':'attention','batch':1,'sq':256,'sk':512,'heads':8,'dim':64,'causal':True},
 {'id':'attention-decode','operation':'attention','batch':1,'sq':1,'sk':1024,'heads':8,'dim':128,'causal':False},
 {'id':'mla-medium','operation':'mla','heads':32,'lengths':[512]},
 {'id':'mla-long','operation':'mla','heads':128,'lengths':[8192]},
 {'id':'mla-ragged','operation':'mla','heads':128,'lengths':[127,511,1023,2049]},
]

def inputs(w,slot):
 rng=np.random.default_rng(1801+slot)
 def value(shape):return np.clip(rng.standard_normal(shape)*.25,-4,4).astype(np.float16)
 if w['operation']=='attention':
  b,sq,sk,h,d=[w[k] for k in ['batch','sq','sk','heads','dim']]
  return [value(shape) for shape in [(b,sq,h,d),(b,sk,h,d),(b,sk,h,d)]]
 b=len(w['lengths']);h=w['heads'];n=max(w['lengths'])
 qa,qr,c,kr=[value(shape) for shape in [(b,h,512),(b,h,64),(b,n,512),(b,n,64)]];ptr=[0]
 for length in w['lengths']:ptr.append(ptr[-1]+(length+15)//16)
 indices=np.random.default_rng(811).permutation(ptr[-1]);pages=np.zeros((ptr[-1],16,512),np.float16);ropes=np.zeros((ptr[-1],16,64),np.float16)
 for bi,length in enumerate(w['lengths']):
  for t in range(length):p=indices[ptr[bi]+t//16];s=t%16;pages[p,s]=c[bi,t];ropes[p,s]=kr[bi,t]
 return [qa,qr,pages,ropes],{'indptr':ptr,'indices':indices.tolist(),'dense_cache':c,'dense_rope':kr}
