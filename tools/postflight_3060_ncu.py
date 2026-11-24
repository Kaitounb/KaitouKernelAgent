"""Reuse the accepted production read-only health check after the separate NCU diagnostic."""
from tools.ssh_3060 import ROOT

def main():
 source=(ROOT/'tools/postflight_cdi_release.py').read_text();old='evidence/20261124-3060-final-postflight';assert source.count(old)==1;source=source.replace(old,'evidence/20261124-3060-ncu-final-postflight');exec(compile(source,str(ROOT/'tools/postflight_cdi_release.py'),'exec'),{'__name__':'__main__'})
if __name__=='__main__':main()
