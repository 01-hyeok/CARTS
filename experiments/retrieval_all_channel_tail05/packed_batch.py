"""True all-channel batches packed into canonical MLP GEMM row counts.

All channels are present in every model call. Only query/candidate axes are
chunked. Padding is discarded, never a real query/candidate or pool member.
"""
import torch
import torch.nn.functional as F
from utils.full_candidate_bank import encode_raw_channel_first
from all_channel_ops import score_all,global_mixture
from cublaslt_batch import CublasLtBatch
from cublas_batch import CublasBatch


class PackedChannelFull:
    def __init__(self,frozen):
        assert frozen.support=='FULL' and frozen.full_trained
        assert frozen.model.encoder.encoder_type=='mlp'
        assert not frozen.model.training and (frozen.heads is None or not frozen.heads.training)
        self.frozen=frozen;self.data=frozen.data;self.keys=None;self.shapes=[]
        self.lt=None;self.blas=None

    def short_query(self,x):
        """All C are the batch dimension of one cuBLASLt call per Linear.

        The default heuristic is queried on the original M/N/K shape, then
        applied once with batchCount=C. No future values or channel loops.
        """
        from models.RelationStage1 import transform_relation_features
        b,c,h=x.shape;enc=self.frozen.model.encoder
        assert b<c and enc.encoder_type=='mlp' and enc.self_fill=='linear'
        if self.lt is None:self.lt=CublasLtBatch()
        views=transform_relation_features(x.permute(0,2,1),self.frozen.model.relation_input_space)
        z=torch.stack(views,2).permute(3,0,2,1).contiguous().reshape(c,b,-1)
        compute=77 if torch.backends.cuda.matmul.allow_tf32 else 68
        for seq in (enc.encoder,enc.norm,enc.proj):
            layers=list(seq) if isinstance(seq,torch.nn.Sequential) else [seq]
            for layer in layers:
                if layer is enc.encoder[0]:
                    # The canonical first-layer kernel is exactly reproduced by
                    # the single flattened all-channel batch (past-only layer
                    # audits H96 B12 and H720 B7). Its split-K Lt alternative
                    # does not support a strided channel batch on this runtime.
                    if (compute==77 and layer.in_features==336) or (compute==68 and layer.in_features==720):
                        # Canonical H336 B7 uses a different first-layer
                        # accumulation than the flattened 21-channel call.
                        z=self.lt.linear(z,layer,compute,1,0)
                    else:z=layer(z.reshape(c*b,-1)).reshape(c,b,-1)
                elif isinstance(layer,torch.nn.Linear):
                    # Past-input-only layer audit of legacy 1.12 CUDA11.3:
                    # B=1 second encoder layer uses FP32 GEMV; B17..20
                    # projection uses FP32 GEMM despite global TF32 enabled.
                    if compute==77 and b==1 and layer is enc.encoder[3]:
                        if self.blas is None:self.blas=CublasBatch()
                        z=self.blas.sgemm_linear(z,layer)
                    elif compute==77 and b in (19,20) and layer is enc.encoder[3]:
                        z=layer(z.reshape(c*b,-1)).reshape(c,b,-1)
                    elif compute==77 and 17<=b<=20 and seq is enc.proj:
                        z=self.lt.linear(z,layer,68,1,0)
                    else:z=self.lt.linear(z,layer,compute,1,0)
                else:z=layer(z)
        if enc.retrieval_similarity!='l2':z=F.normalize(z.reshape(c*b,-1),dim=-1).reshape(c,b,-1)
        z=z.permute(1,0,2).contiguous();dim=z.shape[-1]
        if self.frozen.heads is None:return F.normalize(z.reshape(b*c,dim),dim=-1).reshape(b,c,1,dim)
        if compute==77 and (b==1 or 17<=b<=20):
            weight=self.frozen.heads.W.reshape(-1,dim)[None].expand(c,-1,-1).contiguous()
            q=self.lt.score(z.permute(1,0,2).contiguous(),weight,77 if b==1 else 68,1,0).reshape(c,b,-1,dim)
            return F.normalize(q,dim=-1).permute(1,0,2,3).contiguous()
        return self.frozen.heads(z.reshape(b*c,dim)).reshape(b,c,-1,dim)

    def encode(self,x,rows,project=False):
        b,c,h=x.shape;assert c==self.data.c and b*c<=rows
        raw=x.contiguous().reshape(b*c,h,1)
        if b*c<rows:raw=torch.cat([raw,torch.zeros(rows-b*c,h,1,device=raw.device,dtype=raw.dtype)])
        z=encode_raw_channel_first(self.frozen.model,raw,0)
        if project:
            z=F.normalize(z,dim=-1)[:,None] if self.frozen.heads is None else self.frozen.heads(z)
            return z[:b*c].reshape(b,c,z.shape[1],z.shape[2])
        return F.normalize(z,dim=-1)[:b*c].reshape(b,c,-1)

    @torch.no_grad()
    def build_keys(self):
        f=self.frozen;d=self.data;f.precision();n=len(d.starts)
        chunk=n//d.c;assert chunk>0;parts=[]
        for a in range(0,n,chunk):
            b=min(a+chunk,n);pos=torch.arange(a,b,device=d.device)[:,None]+d.offsets
            x=d.series[pos].permute(0,2,1).contiguous()
            parts.append(self.encode(x,n))
            if len(self.shapes)<2:self.shapes.append(dict(kind='true_channel_batched_candidates',input=list(x.shape),encoder_input=[n,d.h,1],padding_rows=n-(b-a)*d.c,all_channels=True))
        self.keys=torch.cat(parts).permute(1,0,2).contiguous()

    @torch.no_grad()
    def probability(self,split,start,end):
        f=self.frozen;d=self.data;f.precision()
        if self.keys is None:self.build_keys()
        x,_,starts=d.query(split,start,end);b=end-start
        rows=max(b,d.c);chunk=rows//d.c;parts=[]
        if b<d.c:
            q=self.short_query(x)
        else:
            for a in range(0,b,chunk):
                xx=x[a:min(a+chunk,b)]
                parts.append(self.encode(xx,rows,project=True))
            q=torch.cat(parts)
        self.shapes.append(dict(kind='true_channel_batched_queries',original_query_batch=b,all_channels=d.c,
                               query_chunk=chunk,actual_encoder_batch_rows=rows,all_channels_in_each_model_call=True,
                               canonical_row_count_preserved=rows==b,channel_loops=False,
                               tail_backend='single cuBLASLt batchCount=C preserving canonical per-matrix M/N/K' if b<d.c else None))
        if b<32:
            if self.lt is None:self.lt=CublasLtBatch()
            c,s,dim=d.c,q.shape[2],q.shape[3]
            # Legacy short unbatched matmul and strided bmm select different
            # TF32 kernels. Preserve original per-matrix shape/heuristic,
            # then submit all C matrices in one batchCount=C Lt operation.
            qq=q.permute(1,0,2,3).contiguous().reshape(c,b*s,dim)
            compute=77 if torch.backends.cuda.matmul.allow_tf32 else 68
            choice=1 if compute==77 and b*s==2 else 0
            scores=self.lt.score(qq,self.keys,compute,1,choice).reshape(c,b,s,-1).permute(1,0,2,3).contiguous()
        else:scores=score_all(q,self.keys)
        mask=d.mask(starts)[:,None,:].expand(b,d.c,len(d.starts))
        return scores[:,:,0].masked_fill(~mask,-float('inf')) if f.heads is None else global_mixture(scores,mask)
