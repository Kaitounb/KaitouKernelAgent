"""CPU replay of the complete timestamp-corrected, non-qualifying scheduling trace."""
import argparse,hashlib,json,sqlite3,statistics,zipfile
from pathlib import Path

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def audit(folder):
    archive=folder/'archive';root=archive/'remote';collection=json.loads((archive/'collection.json').read_text());assert sha(archive/'raw.zip')==collection['archive_sha256']
    with zipfile.ZipFile(archive/'raw.zip') as z:manifest=json.loads(z.read('manifest.json'))
    assert all((root/n).stat().st_size==m['bytes'] and sha(root/n)==m['sha256'] for n,m in manifest.items())
    c=sqlite3.connect('file:'+str(root/'trace.sqlite')+'?mode=ro',uri=True);rows=json.loads((root/'diagnostic-segments.json').read_text())['rows'];assert len(rows)==54;assert c.execute('SELECT COUNT(DISTINCT globalPid) FROM CUPTI_ACTIVITY_KIND_GRAPH_TRACE').fetchone()[0]==1;out=[]
    for row in rows:
        ranges=c.execute('SELECT start,end FROM NVTX_EVENTS WHERE text=?',(row['range'],)).fetchall();assert len(ranges)==1;start,end=ranges[0]
        graphs=c.execute('SELECT g.start,g.end,r.start,r.end FROM CUPTI_ACTIVITY_KIND_GRAPH_TRACE g JOIN CUPTI_ACTIVITY_KIND_RUNTIME r ON g.correlationId=r.correlationId WHERE r.start>=? AND r.end<=? ORDER BY g.start',(start,end)).fetchall();assert len(graphs)==row['replays']
        durations=[(g[1]-g[0])/1e6 for g in graphs];gaps=[(b[0]-a[1])/1e6 for a,b in zip(graphs,graphs[1:])];api=[(g[3]-g[2])/1e6 for g in graphs]
        out.append({**row,'gpu_graph_count':len(graphs),'graph_span_ms':(graphs[-1][1]-graphs[0][0])/1e6,'sum_graph_ms':sum(durations),'graph_duration_median_ms':statistics.median(durations),'graph_duration_max_ms':max(durations),'graph_gap_sum_ms':sum(gaps),'graph_gap_max_ms':max(gaps),'runtime_graph_launch_max_ms':max(api),'nvtx_span_ms':(end-start)/1e6})
    original=json.loads((folder/'analysis.json').read_text());assert original['rows']==out
    result={'status':'PASS','scope':'CPU SHA archive and all54 native100W NVTX/runtime/GPU Graph count/duration/gap replay; profiler overhead and CPU limits differ, not qualification','raw_files_verified':len(manifest),'segments_replayed':len(out),'gpu_graph_activities':c.execute('SELECT COUNT(*) FROM CUPTI_ACTIVITY_KIND_GRAPH_TRACE').fetchone()[0],'original_analysis_reproduced':True,'timestamp_correction_precision_limitation_retained':True,'new_gpu_calls':0,'new_model_calls':0,'production_release':False}
    (folder/'replay-verification.json').write_text(json.dumps(result,indent=2)+'\n');return result
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('folder',type=Path);a=p.parse_args();print(json.dumps(audit(a.folder.resolve())))
