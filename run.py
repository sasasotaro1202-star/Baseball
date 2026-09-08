import argparse,json,os
from pipelines import shard_runner
p=argparse.ArgumentParser();p.add_argument('command',choices=['shard-plan']);p.add_argument('--statcast-windows',type=int,default=52);p.add_argument('--emit-matrix',action='store_true');a=p.parse_args()
plan,report,estimate=shard_runner.build_plan(a.statcast_windows)
if not report['ok']:raise SystemExit(json.dumps(report,ensure_ascii=False))
matrix=[{'source':src,'index':s.index,'total':s.total} for src,ss in plan.items() for s in ss]
out={'verification':report,'estimate':estimate,'matrix_jobs':len(matrix)};print(json.dumps(out,ensure_ascii=False,indent=2))
if a.emit_matrix and os.getenv('GITHUB_OUTPUT'):
 open(os.environ['GITHUB_OUTPUT'],'a').write('matrix='+json.dumps({'include':matrix},separators=(',',':'))+'\n')
