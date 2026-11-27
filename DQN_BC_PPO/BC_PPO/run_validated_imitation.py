"""Audit -> demonstration verification -> controlled BC comparison -> optional PPO.

All simulation uses the frozen road/traffic fixture. No driving rules, action
filters, detailed lane instructions or native safety help are added to policies.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results/validated_imitation_v94'
SOURCE=ROOT/'results/course_experiments_v92/selected_final'
DATA=[ROOT/'results/demos_audited_native_v94/demonstrations.npz',
      ROOT/'results/demos_audited_recovery_v94/demonstrations.npz']
REPORT={}
CHILDREN=[]


def read(p):
    for attempt in range(7):
        try:return json.loads(Path(p).read_text(encoding='utf-8-sig'))
        except PermissionError:
            if attempt==6:raise
            time.sleep(min(.05*2**attempt,.5))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,a):
    p=Path(p);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(a,ensure_ascii=False,indent=2),encoding='utf-8')
    for attempt in range(7):
        try:
            os.replace(tmp,p)
            return
        except PermissionError:
            if attempt==6:raise
            time.sleep(min(.05*2**attempt,.5))
def status(phase,**items):
    REPORT.update(status=phase,pid=os.getpid(),**items);save(OUT/'status.json',REPORT)
    print(phase,flush=True)


def launch(args,log):
    with Path(log).open('w',encoding='utf-8') as stream:
        p=subprocess.Popen([sys.executable,'-u',*map(str,args)],cwd=ROOT,
            stdout=stream,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,
            creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    CHILDREN.append(p)
    REPORT.setdefault('commands',[]).append({'pid':p.pid,'args':list(map(str,args)),'log':str(log)})
    save(OUT/'status.json',REPORT)
    return p


def wait(jobs,limit):
    deadline=time.monotonic()+limit
    while True:
        codes=[p.poll() for p in jobs]
        if any(code is not None and code!=0 for code in codes):raise RuntimeError('Child process failed: '+str(codes))
        if all(code==0 for code in codes):return
        if time.monotonic()>deadline:raise TimeoutError('Bounded stage deadline exceeded')
        time.sleep(10)


def score(m):
    counts=m['end_reason_counts'];n=m['episodes']
    blocked=sum(counts.get(k,0) for k in ['route_blocked','idle','stuck','time_limit'])/n
    return (bool(m['task_target_met']),m['arrival_rate']-.5*m['collision_rate']-.3*m['red_light_rate']-.4*blocked,
            m['arrival_rate'],-m['collision_rate'])


def evaluate(source,name,episodes=20,seed=15000):
    from freeze_candidate import freeze
    frozen=OUT/(name+'_frozen')
    if not frozen.exists():freeze(source,frozen)
    output=frozen/f'evaluation_{seed}_{episodes}'
    if not (output/'acceptance.json').exists():
        p=launch(['parallel_evaluate.py',frozen,'--episodes',str(episodes),'--seed-start',str(seed),
            '--workers','4','--require-course-profile','--output',output],OUT/(name+f'_eval_{seed}.log'))
        wait([p],5400)
    metrics=read(output/'acceptance.json')
    if not metrics['evaluation_valid'] or not metrics['course_profile_verified']:raise RuntimeError('Invalid evaluation')
    return {'run':str(frozen),'metrics':metrics,'evaluation':str(output)}


def main():
    import numpy as np
    import torch
    from algorithms.ppo import PPO
    from train_warmstart_ppo import saved_environment,load_demonstrations
    from freeze_candidate import freeze
    os.chdir(ROOT);OUT.mkdir(parents=True,exist_ok=True)
    for name in ['run_validated_imitation.py','course_contract_audit.py','collect_audited_course_data.py',
                 'audit_demonstration_actions.py','report_imitation_actions.py']:
        shutil.copyfile(ROOT/name,OUT/name)
    REPORT.update(source=str(SOURCE),source_sha256=sha(SOURCE/'model.pt'),
        dev_seeds=[15000,15019],final_unseen_seeds=[16000,16099],
        ppo_gate='BC development arrival>=60%,collision<=20%,and no lower balanced score than baseline',
        goal='Collision<5% plus high normal completion; stalled vehicles never qualify')
    status('checking_contract_audit')
    audit=read(ROOT/'results/course_validation_v93/audit_result.json')
    REPORT['contract_audit']=audit
    if not audit['passed']:raise RuntimeError('Observation/action contract audit not passed')
    status('waiting_for_audited_demonstrations')
    deadline=time.monotonic()+3*3600
    for data in DATA:
        while not (data.parent/'status.json').exists():
            log=ROOT/'results'/(data.parent.name+'.console.log')
            if log.exists() and 'Traceback (most recent call last)' in log.read_text(encoding='utf-8',errors='replace')[-5000:]:
                raise RuntimeError('Demonstration collector failed: '+str(log))
            if time.monotonic()>deadline:raise TimeoutError('Demonstration collection deadline exceeded')
            time.sleep(10)
        result=read(data.parent/'status.json')
        if result['status']!='completed' or not result['alignment']['passed']:raise RuntimeError('Data alignment audit failed')
    status('checking_demonstrated_braking_and_lane_actions')
    for data in DATA:
        output=OUT/(data.parent.name+'_action_audit')
        if not (output/'audit_result.json').exists():
            p=launch(['audit_demonstration_actions.py',data,output],OUT/(data.parent.name+'_action_audit.log'))
            wait([p],3600)
        result=read(output/'audit_result.json')
        REPORT.setdefault('event_audits',[]).append(result)
        if not result['passed']:
            status('blocked_action_projection',blocking_audit=str(output/'audit_result.json'))
            return
    config=read(SOURCE/'run_config.json')
    REPORT['datasets']=[]
    for data in DATA:
        load_demonstrations([data],104,expected_observation=config['observation'])
        with np.load(data) as a:
            REPORT['datasets'].append({'path':str(data),'sha256':sha(data),'rows':len(a['episode']),
                'train_rows':int((a['episode']%5!=4).sum()),'heldout_rows':int((a['episode']%5==4).sum())})
    fresh=OUT/'fresh_initialization'
    if not fresh.exists():
        fresh.mkdir();cfg=read(SOURCE/'run_config.json');cfg['hparams']['seed']=353
        cfg['stage']='Untrained neural initialization for audited imitation learning'
        env=saved_environment(cfg);torch.set_num_threads(1)
        agent=PPO(env.obs_dim,2,**{**cfg['hparams'],'device':'cpu'});agent.save(str(fresh/'model.pt'));env.close()
        cfg['frozen_checkpoint']={'source':'random_initialization','sha256':sha(fresh/'model.pt')}
        save(fresh/'run_config.json',cfg)
    status('training_audited_bc')
    jobs=[]
    common=['train_local_bc.py','--data',*DATA,'--epochs','120','--underbrake-weight','4',
        '--launch-weight','1','--eval-episodes','10','--eval-seed-start','15000',
        '--set-residual','--metric-sensors','--train-lanes','--train-backbone','--uniform-sampling',
        '--lane-weight-power','.5','--censored-stop-loss','--rollout-every','20',
        '--rollout-episodes','5','--checkpoint-every','20']
    for name,initial,lr in [('fresh_bc',fresh,'.0003'),('warm_bc',SOURCE,'.0001')]:
        output=OUT/name
        if (output/'status.json').exists() and read(output/'status.json').get('status')=='completed':continue
        if output.exists():raise RuntimeError('Preserving existing incomplete BC run: '+str(output))
        jobs.append(launch([*common,'--init-run',initial,'--output',output,'--lr',lr,'--lane-lr',lr],OUT/(name+'.log')))
    wait(jobs,3*3600)
    status('reporting_heldout_braking_launch_and_lateral_quality')
    jobs=[]
    for name,source in [('baseline',SOURCE),('fresh_bc',OUT/'fresh_bc'),('warm_bc',OUT/'warm_bc')]:
        output=OUT/(name+'_action_quality.json')
        jobs.append(launch(['report_imitation_actions.py',source,'--data',*DATA,'--output',output],
            OUT/(name+'_action_quality.log')))
    wait(jobs,1800)
    status('comparing_bc_on_full_route')
    candidates=[evaluate(SOURCE,'baseline'),evaluate(OUT/'fresh_bc','fresh_bc'),evaluate(OUT/'warm_bc','warm_bc')]
    best=max(candidates,key=lambda x:score(x['metrics']));baseline=candidates[0]
    REPORT['bc_development']=candidates;REPORT['bc_selected']=best
    m=best['metrics'];enable=(m['arrival_rate']>=.6 and m['collision_rate']<=.2 and score(m)>=score(baseline['metrics']))
    REPORT['ppo_enabled']=enable
    if enable and not m['task_target_met']:
        status('conservative_ppo_refinement')
        jobs=[]
        common=['train_warmstart_ppo.py','--init-run',best['run'],'--data',*DATA,
            '--steps','65536','--rollout','2048','--epochs','3','--lr','.000005',
            '--gamma','.999','--gae-lambda','.95','--clip','.05','--target-kl','.005',
            '--bc-coef','1','--lane-temperature','.5','--warmup-updates','2',
            '--eval-every','8','--eval-episodes','10','--eval-seed-start','15000',
            '--reward-scale','.002','--risk-coef','.2','--censored-stop-loss','--strict-speed-mode',
            '--collision-penalty','500','--red-light-penalty','300','--failure-penalty','300','--arrival-bonus','500']
        for name,extra,seed in [('ppo_acceleration',['--acceleration-only','--std','.025'],401),
                               ('ppo_joint',['--std','.04'],402)]:
            output=OUT/name
            if output.exists():raise RuntimeError('Preserving existing PPO run')
            jobs.append(launch([*common,'--output',output,'--seed',str(seed),*extra],OUT/(name+'.log')))
        wait(jobs,4*3600)
        candidates.extend([evaluate(OUT/'ppo_acceleration','ppo_acceleration'),evaluate(OUT/'ppo_joint','ppo_joint')])
        best=max(candidates,key=lambda x:score(x['metrics']))
    status('final_unseen_100',selected=best,all_development=candidates)
    final=evaluate(Path(best['run']),'selected_final',100,16000)
    qualified=bool(final['metrics']['task_target_met'] and final['metrics']['course_profile_verified'])
    status('completed',qualified=qualified,final_evaluation=final,
        next_action='Review demonstrated-event and segment failure metrics before further training')
    save(ROOT/'results/recommended_run.json',{'qualified':qualified,
        'recommended_run':final['run'] if qualified else None,'latest_evaluated_run':final['run'],
        'actual_evaluation':final['metrics'],'report':str(OUT/'status.json'),'no_assisted_claim':True})


if __name__=='__main__':
    try:main()
    except BaseException as error:
        OUT.mkdir(parents=True,exist_ok=True)
        status('failed',error=repr(error),traceback=traceback.format_exc())
        for p in CHILDREN:
            if p.poll() is None:p.terminate()
        raise
