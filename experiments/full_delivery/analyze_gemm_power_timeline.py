"""Read-only remote SQL reduction of the collected diagnostic, not a pass gate."""
import json
from tools.ssh_3070 import ROOT,execute_python

def main():
 out=ROOT/'evidence/20261124-sm86-gemm-power-timeline';root=json.loads((out/'warm-corrected-result.json').read_text())['remote_root']
 payload='ROOT='+repr(root)+'\n'+r"""
import json,sqlite3,statistics
from pathlib import Path
root=Path(ROOT);rows=json.loads((root/'diagnostic-segments.json').read_text())['rows'];c=sqlite3.connect('file:'+str(root/'trace.sqlite')+'?mode=ro',uri=True);out=[]
assert len(rows)==54
for row in rows:
 ranges=c.execute('SELECT start,end FROM NVTX_EVENTS WHERE text=?',(row['range'],)).fetchall();assert len(ranges)==1;start,end=ranges[0]
 graphs=c.execute('SELECT g.start,g.end,r.start,r.end FROM CUPTI_ACTIVITY_KIND_GRAPH_TRACE g JOIN CUPTI_ACTIVITY_KIND_RUNTIME r ON g.correlationId=r.correlationId WHERE r.start>=? AND r.end<=? ORDER BY g.start',(start,end)).fetchall();assert len(graphs)==row['replays']
 durations=[(g[1]-g[0])/1e6 for g in graphs];gaps=[(b[0]-a[1])/1e6 for a,b in zip(graphs,graphs[1:])];api=[(g[3]-g[2])/1e6 for g in graphs]
 out.append({**row,'gpu_graph_count':len(graphs),'graph_span_ms':(graphs[-1][1]-graphs[0][0])/1e6,'sum_graph_ms':sum(durations),'graph_duration_median_ms':statistics.median(durations),'graph_duration_max_ms':max(durations),'graph_gap_sum_ms':sum(gaps),'graph_gap_max_ms':max(gaps),'runtime_graph_launch_max_ms':max(api),'nvtx_span_ms':(end-start)/1e6})
print(json.dumps({'scope':'profiled native diagnostic only, CPU limits differ from Docker; timestamp conversion precision limited','gpu_graph_activities':c.execute('SELECT COUNT(*) FROM CUPTI_ACTIVITY_KIND_GRAPH_TRACE').fetchone()[0],'rows':out}))
"""
 (out/'analysis-driver.py').write_text(payload);r=execute_python(payload,120);(out/'analysis.stderr').write_text(r.stderr);assert r.returncode==0;v=json.loads(r.stdout);(out/'analysis.json').write_text(json.dumps(v,indent=2));rows=v['rows'];print(json.dumps({'segments':len(rows),'gpu_graph_activities':v['gpu_graph_activities'],'cpu_enqueue_max_ms':max(x['cpu_enqueue_ms'] for x in rows),'cpu_enqueue_median_ms':__import__('statistics').median(x['cpu_enqueue_ms'] for x in rows),'graph_gap_max_ms':max(x['graph_gap_max_ms'] for x in rows),'graph_duration_max_ms':max(x['graph_duration_max_ms'] for x in rows),'graph_duration_median_range_ms':[min(x['graph_duration_median_ms'] for x in rows),max(x['graph_duration_median_ms'] for x in rows)],'production_release':False}))
if __name__=='__main__':main()
