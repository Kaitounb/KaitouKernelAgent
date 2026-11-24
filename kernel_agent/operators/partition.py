"""Host-only page-aligned split schedule; CUDA computation remains in FlashInfer."""


def split_page_table(lengths,indptr,indices,page_size,splits):
    if type(page_size) is not int or page_size not in [1,16,64] or type(splits) is not int or splits not in [1,2,4]:raise ValueError('unsupported page/split configuration')
    if not lengths or len(indptr)!=len(lengths)+1 or indptr[0]!=0 or indptr[-1]!=len(indices):raise ValueError('page metadata length mismatch')
    if any(type(x) is not int or x<0 for x in [*lengths,*indptr,*indices]) or any(indptr[i+1]-indptr[i]!=(n+page_size-1)//page_size for i,n in enumerate(lengths)):raise ValueError('page metadata invalid')
    result=[{'lengths':[],'indptr':[0],'indices':[]} for _ in range(splits)]
    for request,n in enumerate(lengths):
        pages=indices[indptr[request]:indptr[request+1]];per_part=(len(pages)+splits-1)//splits
        for part,row in enumerate(result):
            first=part*per_part;last=min(len(pages),first+per_part);selected=pages[first:last];length=min(max(n-first*page_size,0),len(selected)*page_size);row['lengths'].append(length);row['indices'].extend(selected);row['indptr'].append(len(row['indices']))
    return result
