"""Build a public minimal CUDA13/PyTorch14 counterpart runtime from the installed Ubuntu24 image."""
import base64,hashlib,json
from tools.ssh_3060 import ROOT,execute_python
DOCKERFILE=r'''FROM ubuntu:latest
ENV DEBIAN_FRONTEND=noninteractive
RUN sed -i 's|http://archive.ubuntu.com/ubuntu|http://mirrors.aliyun.com/ubuntu|g; s|http://security.ubuntu.com/ubuntu|http://mirrors.aliyun.com/ubuntu|g' /etc/apt/sources.list.d/ubuntu.sources && apt-get -o Acquire::Retries=2 update && apt-get install -y --no-install-recommends ca-certificates curl python3.12 python3.12-venv libgomp1 && curl -fL --retry 2 https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb -o /tmp/cuda-keyring.deb && dpkg -i /tmp/cuda-keyring.deb && apt-get update && apt-get install -y --no-install-recommends cuda-sanitizer-13-0 && rm -rf /var/lib/apt/lists/* /tmp/cuda-keyring.deb
RUN python3.12 -m venv /opt/venv && /opt/venv/bin/python -m pip install --no-cache-dir --index-url https://pypi.tuna.tsinghua.edu.cn/simple --upgrade pip && /opt/venv/bin/python -c "from pathlib import Path;Path('/tmp/torch-official.txt').write_text('torch==2.14.0 --hash=sha256:fecffb58f51fd643d213acd68da21cc3fc19bea05a3bc64b4ee55128f47a4963\n')" && /opt/venv/bin/python -m pip install --no-cache-dir --index-url https://pypi.tuna.tsinghua.edu.cn/simple --no-deps --require-hashes -r /tmp/torch-official.txt && /opt/venv/bin/python -m pip install --no-cache-dir --index-url https://pypi.tuna.tsinghua.edu.cn/simple numpy==2.5.3 torch==2.14.0
ENV PATH=/opt/venv/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
ENV LD_LIBRARY_PATH=/opt/venv/lib/python3.12/site-packages/torch/lib:/opt/venv/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:/opt/venv/lib/python3.12/site-packages/nvidia/cublas/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64
ENV NVIDIA_VISIBLE_DEVICES=all NVIDIA_DRIVER_CAPABILITIES=compute,utility NVIDIA_REQUIRE_CUDA=cuda>=13.0 HOME=/tmp PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
RUN /opt/venv/bin/python -c "import json,numpy,torch;assert numpy.__version__=='2.5.3' and torch.__version__=='2.14.0+cu130' and torch.version.cuda=='13.0';print(json.dumps({'torch':torch.__version__,'cuda':torch.version.cuda,'numpy':numpy.__version__}))" && /opt/venv/bin/python -m pip freeze --all > /runtime-packages.txt && dpkg-query -W > /runtime-system-packages.txt
RUN /opt/venv/bin/python -c "from pathlib import Path;import os;p=Path('/usr/local/cuda/bin/compute-sanitizer');candidates=[x for x in Path('/usr/local/cuda-13.0').rglob('compute-sanitizer') if x.is_file() and os.access(x,os.X_OK)];assert candidates;p.parent.mkdir(parents=True,exist_ok=True);p.symlink_to(candidates[0]) if not p.exists() else None" && /usr/local/cuda/bin/compute-sanitizer --version
WORKDIR /work
USER 65534:65534
ENTRYPOINT ["/opt/venv/bin/python"]
'''
def main():
 out=ROOT/'evidence/20261124-3060-next-runtime-build';out.mkdir(exist_ok=False);(out/'Dockerfile').write_text(DOCKERFILE)
 payload='DOCKERFILE='+repr(base64.b64encode(DOCKERFILE.encode()).decode())+'\n'+r'''
import base64,hashlib,json,subprocess,tempfile,time
from pathlib import Path
parent=Path.home()/'nka-persistent';parent.mkdir(mode=0o700,exist_ok=True);root=Path(tempfile.mkdtemp(prefix='gemm-next-runtime-',dir=parent));root.chmod(0o700);(root/'Dockerfile').write_bytes(base64.b64decode(DOCKERFILE));tag='nka-reviewed:3060-next-torch214-cu130-20261124';r=subprocess.run(['docker','image','inspect',tag],capture_output=True,text=True,timeout=30);assert r.returncode!=0,'never overwrite an existing target runtime';r=subprocess.run(['docker','image','inspect','ubuntu:latest'],capture_output=True,text=True,timeout=30);assert r.returncode==0;before=json.loads(r.stdout)[0];assert before['Id']=='sha256:7a398144c5a2fa7dbd9362e460779dc6659bd9b19df50f724250c62ca7812eb3';(root/'base-binding.json').write_text(json.dumps({'image_id':before['Id'],'repo_digests':before.get('RepoDigests'),'dockerfile_sha256':hashlib.sha256((root/'Dockerfile').read_bytes()).hexdigest()},indent=2));started=time.monotonic()
with (root/'build.stdout').open('w') as o,(root/'build.stderr').open('w') as e:r=subprocess.run(['docker','build','--pull=false','--progress','plain','-t',tag,str(root)],stdout=o,stderr=e,timeout=3600)
v={'status':'BUILT' if r.returncode==0 else 'BUILD_FAILED','root':str(root),'tag':tag,'build_exit':r.returncode,'seconds':time.monotonic()-started,'new_gpu_runs':0,'new_model_calls':0,'same_math_binaries_reused':True,'new_image_not_old_qualified_image':True}
if r.returncode==0:
 q=subprocess.run(['docker','image','inspect',tag],capture_output=True,text=True,timeout=30);assert q.returncode==0;d=json.loads(q.stdout)[0];v.update(image_id=d['Id'],size_bytes=d['Size'])
(root/'result.json').write_text(json.dumps(v,indent=2));print(json.dumps(v))
'''
 (out/'driver.py').write_text(payload);r=execute_python(payload,3700);(out/'transport.stdout').write_text(r.stdout);(out/'transport.stderr').write_text(r.stderr);assert r.returncode==0;v=json.loads(r.stdout);(out/'result.json').write_text(json.dumps(v,indent=2));print(json.dumps(v))
if __name__=='__main__':main()
