"""Sequential unattended campaign with explicit time limits and durable logs.

The model processes exit between generation and judging, releasing VRAM. The
notebook's kernel is never contacted. No publishing or external LLM API calls.
"""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',default='artifacts/safety_campaign_v1')
    p.add_argument('--hours',type=float,default=4.5)
    p.add_argument('--judge-reserve-minutes',type=float,default=50)
    p.add_argument('--deadline-epoch',type=float)
    a=p.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    deadline=a.deadline_epoch or time.time()+a.hours*3600
    generate_deadline=deadline-a.judge_reserve_minutes*60
    stages=[]
    def stage(name,args):
        start=time.time()
        with (out/f'{name}.log').open('a') as log:
            process=subprocess.run([sys.executable,'-u','-m',*args],stdout=log,stderr=subprocess.STDOUT)
        stages.append({'stage':name,'started_utc':datetime.fromtimestamp(start,timezone.utc).isoformat(),
                       'seconds':time.time()-start,'exit_code':process.returncode})
        (out/'timings.json').write_text(json.dumps(stages,indent=2)+'\n')
        return process.returncode==0
    if not (out/'benchmark_prompts.csv').exists():
        if not stage('prepare',['scripts.campaign_data',str(out)]):raise SystemExit('Data preparation failed; inspect prepare.log.')
    okay=stage('generation',['scripts.safety_campaign','--out',str(out),'--stop-epoch',str(generate_deadline)])
    if not okay:
        (out/'PIPELINE_ERROR.txt').write_text('Generation failed: inspect generation.log. Partial batches remain available.\n')
    if (out/'responses.jsonl').exists():
        stage('judge',['scripts.campaign_judge','--out',str(out),'--stop-epoch',str(deadline-300)])
        if not stage('report',['scripts.campaign_report','--out',str(out)]):raise SystemExit('Report failed; inspect report.log.')
    else:raise SystemExit('No generated responses; inspect generation.log.')


if __name__=='__main__':main()
