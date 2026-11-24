"""Collect only public3060 release directories; reuse reviewed lossless collector."""
import sys
from pathlib import Path
from tools.ssh_3060 import ROOT

def main():
 source=(ROOT/'tools/collect_3060_public_run.py').read_text();old="/home/featurize/nka-persistent/gemm-counterpart-[A-Za-z0-9_-]+";assert source.count(old)==1;source=source.replace(old,"/home/featurize/nka-persistent/gemm-release-[A-Za-z0-9_.-]+")
 # Same output format and known public blobs, no private directory scans.
 exec(compile(source,str(ROOT/'tools/collect_3060_public_run.py'),'exec'),{'__name__':'__main__'})
if __name__=='__main__':main()
