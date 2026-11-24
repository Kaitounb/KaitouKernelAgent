"""Bind a new immutable preview to the actual replacement3060/image, retaining old wheels."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 build=json.loads((ROOT/'evidence/20261124-3060-next-runtime-build/result.json').read_text());assert build['status']=='BUILT';image=build['image_id'];old_image='sha256:7ed9d551ead159c3a4755f2a3ac84a01caf902bf3821495007828dc5eb18a4b7';gpu='GPU-0140b895-e4e9-7c67-aba1-c7a953df1847';assert sha(ROOT/'artifacts/wheels/nvidia_kernel_agent-0.1.3rc1-py3-none-any.whl')=='c81d6a10d78b034d60f5d2163bdf55720eeb06a5ee0e9140076dc26dcad102e9';assert not (ROOT/'artifacts/wheels/nvidia_kernel_agent-0.1.3rc2-py3-none-any.whl').exists();out=ROOT/'evidence/20261124-3060-next-release-binding';out.mkdir(exist_ok=False)
 for name in ['kernel_agent/cdi_runner.py','tools/cdi_container_entry.py']:
  p=ROOT/name;s=p.read_text();assert old_image in s;p.write_text(s.replace(old_image,image))
 p=ROOT/'kernel_agent/cdi_qualification.py';s=p.read_text();assert 'GPU-d464a726-39c2-9ad5-d856-f5d00c1fa870' in s;p.write_text(s.replace('GPU-d464a726-39c2-9ad5-d856-f5d00c1fa870',gpu))
 p=ROOT/'kernel_agent/data/cdi/releases.json';v=json.loads(p.read_text())
 for m in v['versions'].values():assert m['image_id']==old_image;m['image_id']=image;m['evidence']='evidence/20261124-3060-next-qualification/verification.json'
 p.write_text(json.dumps(v,indent=2)+'\n')
 for name in ['pyproject.toml','kernel_agent/__init__.py']:
  p=ROOT/name;s=p.read_text();assert '0.1.3rc1' in s;p.write_text(s.replace('0.1.3rc1','0.1.3rc2'))
 p=ROOT/'experiments/full_delivery/build_cdi_release.py';p.write_text(p.read_text().replace("['0.1.3rc1','0.1.3']","['0.1.3rc1','0.1.3rc2','0.1.3']"))
 for name in ['install_cdi_release.py','cdi_release_validate.py','cdi_independent_host.py','run_cdi_independent.py','audit_cdi_independent.py','issue_cdi_qualification.py','resume_cdi_release_stage.py']:
  p=ROOT/'experiments/full_delivery'/name;s=p.read_text().replace('0.1.3rc1','0.1.3rc2').replace('3060-rc1-independent-evidence','3060-rc2-independent-evidence').replace('20261124-3060-independent','20261124-3060-next-independent').replace('20261124-sm86-3060-counterpart','20261124-3060-next-qualification').replace('20261124-3060-qualification','20261124-3060-next-certificate').replace('GPU-d464a726-39c2-9ad5-d856-f5d00c1fa870',gpu).replace(old_image,image).replace('rtx3060-cdi-rc1-independent-v1','rtx3060-cdi-rc2-independent-v1');p.write_text(s)
 value={'status':'NEW_TARGET_RC2_BOUND_NOT_QUALIFIED','target_uuid':gpu,'driver':'580.95.05','image_id':image,'preview_version':'0.1.3rc2','prior_preview_immutable_sha256':'c81d6a10d78b034d60f5d2163bdf55720eeb06a5ee0e9140076dc26dcad102e9','new_model_calls':0,'production_release':False};(out/'result.json').write_text(json.dumps(value,indent=2));print(json.dumps(value))
if __name__=='__main__':main()
