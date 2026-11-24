"""Prepare one reviewable bounded DeepSeek source request without reading a key or calling a model."""
import argparse,json
from pathlib import Path
from .identity import ROOT,identity,sha


def prepare(output):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    source=ROOT/'kernels/gemm/wmma_sm86.cu';parent=identity(4)
    task={'gpu':'RTX3070 / SM86','parent_alias':'current','source_file':'kernels/gemm/wmma_sm86.cu','source':source.read_text(),
          'contract':'FP16 row-major inputs, FP32 accumulation/output, finite values in [-4,4]. Original host dispatcher/cuBLAS tails unchanged.',
          'request':'Propose one small structural change: process two consecutive 16-wide WMMA K tiles per loop iteration, with correct single-tile tail. Preserve FP32 accumulation order. Do not add barriers, host code, dependencies or change warp mapping.',
          'constraints':'Only unified diff for the supplied CUDA file. Preserve the license, includes and complete gemm_template host launcher. No evaluator, contract, timing, tolerance or budget edits. No claim of measured gain.'}
    schema={'schema_version':1,'parent_id':'current','patch':'diff --git a/kernels/gemm/wmma_sm86.cu b/kernels/gemm/wmma_sm86.cu\n...','reason':'Testable hypothesis, <=1000 chars'}
    request={'model':'deepseek-flash','messages':[{'role':'system','content':'Return only JSON with exactly these fields: '+json.dumps(schema)},{'role':'user','content':json.dumps(task)}],
             'max_tokens':768,'temperature':.2,'thinking':'disabled','response_format':'json_object','timeout_seconds':60,'max_retries':0,'maximum_calls':1}
    (output/'request.json').write_text(json.dumps(request,indent=2)+'\n')
    (output/'parent-binding.json').write_text(json.dumps({'actual_parent':parent,'request_alias':'current','source_sha256':sha(source)},indent=2)+'\n')
    status={'status':'PREPARED_NOT_SENT','real_model_calls':0,'credentials_read':False,'gpu_runs':0,'heldout_read':False,'source_change_implemented':False,'p3_01_accepted':False}
    (output/'status.json').write_text(json.dumps(status,indent=2)+'\n');return status


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);a=parser.parse_args()
    if a.output.exists():parser.error('Use a new directory')
    print(json.dumps(prepare(a.output)))


if __name__=='__main__':main()
