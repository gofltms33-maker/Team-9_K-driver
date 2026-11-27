"""Continue model B with broader training seeds and paired300+300evaluation.

Roads, traffic demand/composition, native physics and termination definitions
are immutable. All scenario focus is offline data sampling, never AV control.
"""
from pathlib import Path
from datetime import datetime,timedelta,timezone
from collections import Counter
import shutil
import time
import traceback

import run_validated_imitation as w
from freeze_candidate import freeze
from run_nine_hour_safety import execution_state
from run_navigation_preserving_safety import stalls

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results/broad_seed_refinement_v112'
A=ROOT/'results/diverse_imitation_v107/selected_frozen'
B=ROOT/'results/current_model_refinement_v111/selected_frozen'


def blocks(rows):
    result=[]
    for start in range(0,len(rows),100):
        part=sorted(rows,key=lambda r:r['seed'])[start:start+100];counts=Counter(x['end_reason'] for x in part)
        result.append(dict(episodes=len(part),seed_start=part[0]['seed'],seed_stop=part[-1]['seed'],
            collision_rate=counts['collision']/len(part),arrival_rate=counts['arrival']/len(part),end_reason_counts=dict(counts)))
    return result


def admissible(candidate,a,b):
    c=candidate['metrics'];x=a['metrics'];y=b['metrics']
    return (all(m['evaluation_valid'] and m['course_profile_verified'] for m in [c,x,y]) and
        c['episodes']==x['episodes']==y['episodes'] and c['collision_rate']<=y['collision_rate'] and
        c['arrival_rate']>=y['arrival_rate'] and stalls(c)<=min(stalls(x),stalls(y)) and
        c['end_reason_counts'].get('route_blocked',0)<=min(x['end_reason_counts'].get('route_blocked',0),y['end_reason_counts'].get('route_blocked',0)) and
        c['red_light_rate']<=min(x['red_light_rate'],y['red_light_rate']))


def qualifies(candidate):
    m=candidate['metrics']
    return (m['evaluation_valid'] and m['course_profile_verified'] and m['episodes']==300 and
        m['collision_rate']<.05 and m['arrival_rate']>=.95 and stalls(m)==0 and
        all(b['episodes']==100 and b['collision_rate']<.05 and b['arrival_rate']>=.95 for b in candidate['seed_blocks']))


def rank(candidate):
    m=candidate['metrics']
    return qualifies(candidate),-max(b['collision_rate'] for b in candidate['seed_blocks']),-m['collision_rate'],m['arrival_rate'],-stalls(m)


def wait_one(p,limit=7200):
    try:w.wait([p],limit)
    except Exception:
        if p.poll() is None:p.terminate()
        raise


def evaluate(source,name,seed,episodes=300):
    assert_preserved()
    target=OUT/(name+'_frozen');freeze(source,target)
    output=target/f'evaluation_{seed}_{episodes}'
    p=w.launch(['parallel_evaluate.py',target,'--episodes',str(episodes),'--seed-start',str(seed),
        '--workers','6','--require-course-profile','--output',output],OUT/(name+'.eval.log'))
    wait_one(p)
    metrics=w.read(output/'acceptance.json')
    if not metrics['evaluation_valid'] or not metrics['course_profile_verified']:raise RuntimeError('Invalid evaluation')
    rows=w.read(output/'evaluation.json')['episodes']
    return dict(run=str(target),evaluation=str(output),metrics=metrics,seed_blocks=blocks(rows))


def assert_preserved():
    for path,digest in w.REPORT.get('preserved_file_hashes',{}).items():
        if w.sha(Path(path))!=digest:raise RuntimeError('Preserved experiment file changed: '+path)


def collection(source,folder,seeds,lead,window):
    return w.launch(['collect_audited_course_data.py',source,folder,'--kind','recovery','--episodes','0',
        '--seed-start','54100','--include-seeds',*map(str,seeds),'--lead-steps',str(lead),
        '--window-steps',str(window),'--min-successful-episodes','1'],OUT/(folder.name+'.log'))


def checked_data(folder,process,required=False):
    try:wait_one(process)
    except Exception as error:
        w.REPORT.setdefault('rejected_data',[]).append(dict(folder=str(folder),error=repr(error)))
        if required:raise
        return None
    audit=w.read(folder/'data_audit.json')
    if not audit['alignment']['passed']:raise RuntimeError('Independent alignment failed')
    target=OUT/(folder.name+'_action_check')
    p=w.launch(['audit_demonstration_actions.py',folder/'demonstrations.npz',target],OUT/(folder.name+'.action.log'))
    wait_one(p,3600);check=w.read(target/'audit_result.json')
    valid=bool(check['results']) and all(x['passed'] for x in check['results'])
    w.REPORT.setdefault('data_checks',[]).append(dict(folder=str(folder),samples=audit['samples'],
        alignment_passed=True,action_events=check['tested_events'],all_checked_actions_passed=valid,
        general_coverage_passed=check['passed']))
    if not valid:
        if required:raise RuntimeError('Required route-recovery actions failed replay')
        return None
    if folder.name=='route_recovery':
        audit['training_focus']=True
        audit['training_focus_scope']='Audited early route-recovery rows only; no new labels, observations or runtime rules'
        w.save(folder/'data_audit.json',audit)
    return folder/'demonstrations.npz'


def main():
    OUT.mkdir(parents=True,exist_ok=False);w.OUT=OUT;w.REPORT.clear()
    shutil.copyfile(__file__,OUT/Path(__file__).name)
    start=datetime.now(timezone(timedelta(hours=9)));deadline=time.monotonic()+3*3600
    cfg=w.read(B/'run_config.json');data=[Path(x['path']) for x in cfg['data_sources']]
    from parallel_evaluate import validate_course_config
    validate_course_config(cfg)
    protected=dict(cfg['protected_hashes'])
    for relative in ['env/sumo_env.py','env/mdp_config.py','env/road_config.py',
                     'env/traffic_mix.py','env/roads/traffic_mix.py','env/traffic_scenarios.py',
                     'env/sumo/highway.net.xml','env/sumo/highway.rou.xml',
                     'utils/networks.py','algorithms/ppo.py']:
        path=ROOT/relative
        if path.exists():protected[str(path)]=w.sha(path)
    protected[str(A/'model.pt')]=w.sha(A/'model.pt')
    protected[str(B/'model.pt')]=w.sha(B/'model.pt')
    if not all(w.sha(Path(p))==h for p,h in protected.items()):raise RuntimeError('Protected road/traffic changed')
    awake=execution_state(True)
    try:
        w.status('broad_training_data_collection',source_b=str(B),source_b_sha256=w.sha(B/'model.pt'),
            preserved_a=str(A),source_a_sha256=w.sha(A/'model.pt'),started_at_kst=start.isoformat(),
            target_finish_kst=(start+timedelta(hours=3)).isoformat(),
            development_seeds=[37400,37699],private_seeds=[37700,37999],
            native_training_seeds=[54000,54047],training_probe_seeds=[55000,55099],
            road_hv_geometry_composition_generation_timeouts_preserved=True,
            teacher_training_only=True,student_modes=[96,0],no_runtime_driving_assistance=True,
            temporary_sleep_prevention=awake)
        w.REPORT['preserved_file_hashes']=protected
        natives=[]
        for i in range(2):
            folder=OUT/f'native_{i}'
            p=w.launch(['collect_audited_course_data.py',B,folder,'--kind','native','--episodes','24',
                '--seed-start',str(54000+24*i)],OUT/f'native_{i}.log');natives.append((folder,p))
        w.status('mining_training_only_failures')
        probe=evaluate(B,'training_probe',55000,100)
        w.REPORT['training_probe']=probe
        probe_rows=w.read(Path(probe['evaluation'])/'evaluation.json')['episodes']
        old=w.read(ROOT/'results/current_model_refinement_v111/private_failure_audit/result.json')['results']
        routes=sorted({r['seed'] for r in probe_rows+old if r['end_reason']=='route_blocked'})
        other=sorted({r['seed'] for r in probe_rows+old if r['end_reason'] not in ['arrival','route_blocked']})
        route_folder=OUT/'route_recovery';other_folder=OUT/'interaction_recovery'
        route_job=collection(B,route_folder,routes,80,240) if routes else None
        other_job=collection(B,other_folder,other,12,180) if other else None
        w.status('checking_broader_demonstrations',route_failure_seeds=routes,interaction_failure_seeds=other)
        for folder,p in natives:
            extra=checked_data(folder,p)
            if extra:data.append(extra)
        if route_job:
            extra=checked_data(route_folder,route_job,required=True)
            if extra:data.append(extra)
        if other_job:
            extra=checked_data(other_folder,other_job)
            if extra:data.append(extra)
        if deadline-time.monotonic()<1800:raise TimeoutError('Insufficient remaining learning/evaluation budget')
        epochs=15 if deadline-time.monotonic()>5400 else 8
        common=['train_local_bc.py','--init-run',B,'--data',*data,'--epochs',str(epochs),
            '--set-residual','--metric-sensors','--train-lanes','--lane-weight-power','0',
            '--lane-class-fraction','.025','--priority-fraction','.2','--stop-hold-fraction','.2',
            '--censored-stop-loss','--stop-hold-weight','3','--underbrake-weight','4','--brake-weight-scope','all',
            '--launch-weight','1','--reference-coef','.02','--lane-reference-coef','.02',
            '--lane-safety-coef','.5','--spatial-safety-horizon','2',
            '--eval-episodes','5','--eval-seed-start','37400']
        settings=[('lateral',['--lane-only','--lr','.00001','--lane-lr','.00001']),
                  ('joint_adapters',['--lr','.00002','--lane-lr','.00001','--predictive-safety-coef','2',
                    '--predictive-reaction-time','1','--predictive-comfort-decel','3.5','--spatial-safety-coef','1'])]
        jobs=[]
        assert_preserved()
        w.status('training_balanced_b_continuations',data_paths=list(map(str,data)),epochs=epochs)
        for name,extra in settings:
            jobs.append(w.launch([*common,'--output',OUT/name,*extra],OUT/(name+'.log')))
        for p in jobs:wait_one(p)
        w.status('development_300_seed_comparison')
        a=evaluate(A,'a_development',37400);b=evaluate(B,'b_development',37400);candidates=[]
        for name,_ in settings:
            last=OUT/(name+'_last');freeze(OUT/name,last,checkpoint='model_last.pt')
            candidates.append(evaluate(last,name+'_development',37400))
            w.status('development_300_seed_comparison',a_development=a,b_development=b,candidates=candidates)
        best=max([b]+[c for c in candidates if admissible(c,a,b)],key=rank)
        w.status('independent_300_seed_comparison',selected=best,retained_b=best is b)
        pa=evaluate(A,'a_private',37700);pb=evaluate(B,'b_private',37700)
        final=evaluate(Path(best['run']),'selected_private',37700)
        preserved=admissible(final,pa,pb);qualified=preserved and qualifies(final)
        assert_preserved()
        w.status('completed_target_verified' if qualified else 'completed_experiments_goal_not_met',
            qualified=qualified,objective_achieved=qualified,final_evaluation=final,
            paired_a=pa,paired_baseline=pb,paired_nonregression_passed=preserved,
            ended_at_kst=datetime.now(timezone(timedelta(hours=9))).isoformat(),
            protected_files_unchanged=all(w.sha(Path(p))==h for p,h in protected.items()))
        if qualified:
            w.save(ROOT/'results/recommended_run.json',dict(qualified=True,recommended_run=final['run'],
                actual_evaluation=final['metrics'],seed_blocks=final['seed_blocks'],report=str(OUT/'status.json')))
    finally:execution_state(False)


if __name__=='__main__':
    if OUT.exists():raise SystemExit('Existing experiment preserved; no duplicate launch')
    try:main()
    except BaseException as error:
        OUT.mkdir(parents=True,exist_ok=True);w.OUT=OUT
        w.status('failed',error=repr(error),traceback=traceback.format_exc(),objective_achieved=False)
        for p in w.CHILDREN:
            if p.poll() is None:p.terminate()
        raise
