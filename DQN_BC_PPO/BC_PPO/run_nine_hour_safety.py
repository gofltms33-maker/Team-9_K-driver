"""Time-bounded local learning/evaluation supervisor; never drives a vehicle.

Existing experiments are preserved. After their measured outcomes, new rounds
adapt supervised learning to replayed failure types, validate on paired seeds,
and require separate private tests before a model is recommended.
"""
import argparse
from collections import Counter
from datetime import datetime,timedelta,timezone
import os
from pathlib import Path
import random
import shutil
import time
import traceback

import run_validated_imitation as w
from run_navigation_preserving_safety import goal
from run_sequential_safety import nonregressing,rank

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results/nine_hour_safety_v110'
FAMILIES=['adaptive_safety_followup_v106','diverse_imitation_v107','sequential_safety_v109']
ANCHOR=ROOT/'results/safety_bottleneck_v103/r3_joint_frozen'
KST=timezone(timedelta(hours=9))


def process_alive(pid):
    if not pid:return False
    if os.name=='nt':
        import ctypes
        from ctypes import wintypes
        kernel=ctypes.WinDLL('kernel32',use_last_error=True)
        kernel.OpenProcess.argtypes=[wintypes.DWORD,wintypes.BOOL,wintypes.DWORD]
        kernel.OpenProcess.restype=wintypes.HANDLE
        kernel.WaitForSingleObject.argtypes=[wintypes.HANDLE,wintypes.DWORD]
        kernel.CloseHandle.argtypes=[wintypes.HANDLE]
        handle=kernel.OpenProcess(0x00100000,False,int(pid))
        if not handle:return None if ctypes.get_last_error()==5 else False
        try:return kernel.WaitForSingleObject(handle,0)==0x102
        finally:kernel.CloseHandle(handle)
    try:os.kill(pid,0);return True
    except OSError:return False


def execution_state(active):
    if os.name!='nt':return False
    import ctypes
    from ctypes import wintypes
    function=ctypes.windll.kernel32.SetThreadExecutionState
    function.argtypes=[wintypes.DWORD];function.restype=wintypes.DWORD
    result=function(0x80000001 if active else 0x80000000)
    return bool(result)


def terminal(report):
    return report.get('status','').startswith(('completed','failed','skipped','blocked'))


def learning_focus(contact_types,end_reasons=None):
    reasons=end_reasons or {}
    if not contact_types and reasons.get('route_blocked',0):return 'lateral'
    if contact_types.get('side',0)>=max(contact_types.get('collision',0),contact_types.get('junction',0),1):
        return 'lateral'
    if contact_types.get('junction',0)>contact_types.get('collision',0):return 'crossing'
    return 'stop'


def execute(args,name,limit=14400):
    p=w.launch(args,OUT/(name+'.log'))
    try:w.wait([p],limit)
    except Exception:
        if p.poll() is None:p.terminate()
        raise


def check_private(final,baseline):
    return nonregressing(final['metrics'],baseline['metrics']) and goal(final['metrics'])


def promote(final,paired,confirmation,round_index):
    if not check_private(final,paired) or not goal(confirmation['metrics']):return False
    w.status('completed_target_verified',qualified=True,objective_achieved=True,
        final_evaluation=final,paired_baseline=paired,confirmation=confirmation,round=round_index)
    w.save(ROOT/'results/recommended_run.json',dict(qualified=True,recommended_run=final['run'],
        latest_evaluated_run=final['run'],actual_evaluation=final['metrics'],
        confirmation=confirmation,report=str(OUT/'status.json'),no_assisted_claim=True))
    return True


def candidate_sources(reports):
    paths=[ANCHOR]
    for report in reports:
        for key in ['final_evaluation','selected']:
            candidate=report.get(key)
            if isinstance(candidate,dict) and candidate.get('run'):paths.append(Path(candidate['run']))
    unique=[];seen=set()
    for path in paths:
        if not (path/'model.pt').exists():continue
        digest=w.sha(path/'model.pt')
        if digest not in seen:seen.add(digest);unique.append(path)
    return unique[:5]


def recovery_data(source,round_index,known):
    import numpy as np
    folder=OUT/f'r{round_index}_recovery'
    random.Random(2200+round_index).shuffle(known)
    args=['collect_audited_course_data.py',source,folder,'--kind','recovery',
        '--episodes','24','--seed-start',str(52000+50*round_index),'--include-seeds',*map(str,known[:24]),
        '--lead-steps','2','--window-steps','120','--min-successful-episodes','1']
    try:execute(args,f'r{round_index}_collection')
    except Exception as error:
        w.REPORT.setdefault('rejected_data',[]).append(dict(folder=str(folder),error=repr(error)));return None
    audit=w.read(folder/'data_audit.json')
    if not audit['alignment']['passed']:raise RuntimeError('Recovery alignment failed')
    dataset=folder/'demonstrations.npz'
    with np.load(dataset) as archive:
        valid=archive['accel_valid']&archive['lane_valid'];indices=np.flatnonzero(valid)
        lateral=np.flatnonzero(valid&(archive['action'][:,1]!=0))
    if not len(indices):return None
    checks=[]
    for index in sorted({int(indices[0]),int(lateral[0]) if len(lateral) else int(indices[-1])}):
        target=OUT/f'r{round_index}_action_checks'
        execute(['audit_demonstration_actions.py',dataset,target,'--index',str(index)],
                f'r{round_index}_action_{index}',1800)
        checks.append(w.read(target/f'event_{index}.json'))
    valid=all(c['passed'] for c in checks)
    w.REPORT.setdefault('recovery_audits',[]).append(dict(path=str(dataset),samples=audit['samples'],
        retained_episodes=sum(e.get('retained',False) for e in audit['episodes']),
        alignment_passed=True,action_events=len(checks),checked_actions_passed=valid))
    return dataset if valid else None


def fit(source,data,index,focus,joint,epochs):
    name=f'r{index}_'+('joint' if joint else focus)
    model=OUT/name
    args=['train_local_bc.py','--init-run',source,'--output',model,'--data',*data,
        '--epochs',str(epochs),'--set-residual','--metric-sensors','--censored-stop-loss',
        '--underbrake-weight','4','--brake-weight-scope','all','--launch-weight','1',
        '--priority-fraction','.45','--reference-coef','.01',
        '--predictive-reaction-time','1','--predictive-comfort-decel','3.5','--spatial-safety-horizon','2',
        '--eval-episodes','5','--eval-seed-start',str(35100+300*(index-1))]
    if focus=='lateral' and not joint:
        args+=['--lane-only','--train-lanes','--train-lane-backbone','--lr','.00003',
            '--lane-lr','.00003','--lane-class-fraction','.01','--lane-weight-power','0',
            '--lane-safety-coef','3','--lane-reference-coef','.03']
    elif focus=='stop' and not joint:
        args+=['--lr','.0001','--train-accel-backbone','--stop-hold-fraction','.5',
            '--stop-hold-weight','12','--predictive-safety-coef','4','--spatial-safety-coef','2']
    else:
        args+=['--lr','.00003','--lane-lr','.000008','--train-lanes','--train-backbone',
            '--lane-class-fraction','.01','--lane-weight-power','0','--lane-safety-coef','2',
            '--lane-reference-coef','.03','--stop-hold-fraction','.3','--stop-hold-weight','8',
            '--predictive-safety-coef','4','--spatial-safety-coef','3']
    execute(args,name,21600)
    return model,name


def main(hours):
    OUT.mkdir(parents=True,exist_ok=False);w.OUT=OUT;w.REPORT.clear()
    shutil.copyfile(__file__,OUT/Path(__file__).name)
    start=datetime.now(KST);deadline=time.monotonic()+hours*3600
    awake=execution_state(True)
    try:
        w.status('waiting_for_existing_experiments',started_at_kst=start.isoformat(),
            no_new_learning_after_kst=(start+timedelta(hours=hours)).isoformat(),
            budget_hours=hours,temporary_sleep_prevention=awake,existing_families=FAMILIES,
            no_runtime_driving_assistance=True,road_hv_rules_preserved=True,
            reserved_round_seeds=[dict(development=35100+300*i,private=35200+300*i,confirmation=35300+300*i) for i in range(3)])
        while True:
            reports=[w.read(ROOT/f'results/{name}/status.json') for name in FAMILIES]
            inherited=next((r for r in reports if r.get('qualified') and
                r.get('final_evaluation',{}).get('metrics',{}).get('episodes')==100 and
                goal(r['final_evaluation']['metrics'])),None)
            if inherited:
                w.status('completed_existing_target_verified',qualified=True,objective_achieved=True,
                    final_evaluation=inherited['final_evaluation'],
                    note='Existing family independently verified the goal; its saved recommendation is preserved.')
                return
            finished=all(terminal(r) or process_alive(r.get('pid')) is False for r in reports)
            w.REPORT['existing_job_states']=[dict(name=n,status=r.get('status'),pid=r.get('pid'),
                alive=process_alive(r.get('pid'))) for n,r in zip(FAMILIES,reports)]
            w.save(OUT/'status.json',w.REPORT)
            if finished or time.monotonic()>=deadline:break
            time.sleep(30)
        sources=candidate_sources(reports);known=set()
        for report in reports:
            final=report.get('final_evaluation')
            if final:
                rows=w.read(Path(final['evaluation'])/'evaluation.json')['episodes']
                known.update(r['seed'] for r in rows if r['end_reason']!='arrival')
        for name in ['anchored_safety_ppo_v105','safety_bottleneck_v103']:
            known.update(r['seed'] for r in w.read(ROOT/f'results/{name}/final_collision_audit/result.json')['results'])
        for index in range(1,4):
            remaining=deadline-time.monotonic()
            if remaining<5400:break
            seed=35100+300*(index-1)
            try:
                w.status('fresh_paired_candidate_comparison',round=index,remaining_seconds=round(remaining))
                comparison=[w.evaluate(path,f'r{index}_source_{i}',100,seed) for i,path in enumerate(sources)]
                anchor=comparison[0];best=max([anchor]+[c for c in comparison[1:]
                    if nonregressing(c['metrics'],anchor['metrics'])],key=rank)
                w.REPORT.setdefault('source_comparisons',[]).append(dict(round=index,candidates=comparison,selected=best))
                paired=None
                if not goal(best['metrics']):
                    audit=OUT/f'r{index}_failure_audit'
                    w.status('diagnosing_current_failures',round=index,selected=best)
                    execute(['audit_saved_failures.py',best['run'],Path(best['evaluation'])/'evaluation.json',
                        '--output',audit,'--workers','3'],f'r{index}_failure_audit',5400)
                    replay=w.read(audit/'result.json')
                    if not replay['all_replays_match']:raise RuntimeError('Failure replay mismatch')
                    types=Counter(c['type'] for r in replay['results'] for c in r['contacts'])
                    focus=learning_focus(types,Counter(r['end_reason'] for r in replay['results']))
                    w.status('collecting_next_recovery_examples',round=index,selected=best,
                        measured_contact_types=dict(types),learning_focus=focus)
                    config=w.read(Path(best['run'])/'run_config.json')
                    data=list(dict.fromkeys([Path(x['path']) for x in config['data_sources']]+
                        [ROOT/f'results/diverse_imitation_v107/native_{i}/demonstrations.npz' for i in range(4)]))
                    new_data=recovery_data(Path(best['run']),index,sorted(known))
                    if new_data:data.append(new_data)
                    if deadline-time.monotonic()<3600:break
                    epochs=30 if deadline-time.monotonic()>10800 else 20
                    before=best;trials=[]
                    for joint in ([False,True] if focus!='crossing' else [True]):
                        if deadline-time.monotonic()<3600:break
                        w.status('training_next_safety_candidate',round=index,learning_focus=focus,joint=joint,selected=best)
                        model,name=fit(Path(before['run']),data,index,focus,joint,epochs)
                        trials.append(w.evaluate(model,name,100,seed))
                    best=max([before]+[c for c in trials if nonregressing(c['metrics'],before['metrics'])],key=rank)
                    w.REPORT.setdefault('trial_comparisons',[]).append(dict(round=index,before=before,trials=trials,selected=best))
                    sources=list(dict.fromkeys([ANCHOR,Path(best['run']),*[Path(c['run']) for c in trials]]))[:4]
                w.status('independent_private_evaluation',round=index,selected=best)
                paired=w.evaluate(ANCHOR,f'r{index}_paired_initial',100,seed+100)
                final=w.evaluate(Path(best['run']),f'r{index}_selected',100,seed+100)
                w.REPORT.setdefault('private_tests',[]).append(dict(round=index,candidate=final,baseline=paired,
                    nonregressing=nonregressing(final['metrics'],paired['metrics'])))
                known.update(r['seed'] for r in w.read(Path(final['evaluation'])/'evaluation.json')['episodes']
                             if r['end_reason']!='arrival')
                if check_private(final,paired):
                    confirmation=w.evaluate(Path(best['run']),f'r{index}_confirmation',100,seed+200)
                    if promote(final,paired,confirmation,index):return
                w.status('result_below_goal_continuing',round=index,final_evaluation=final,paired_baseline=paired)
            except Exception as error:
                w.REPORT.setdefault('round_errors',[]).append(dict(round=index,error=repr(error),traceback=traceback.format_exc()))
                w.status('round_failed_preserving_models_and_continuing',round=index)
        w.status('completed_nine_hour_window',qualified=False,objective_achieved=False,
            ended_at_kst=datetime.now(KST).isoformat(),
            note='No goal-qualified model claimed. Existing jobs were preserved; no further learning round launched with insufficient remaining time.')
    finally:
        execution_state(False)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--hours',type=float,default=9)
    args=parser.parse_args()
    if not 0<args.hours<=24:parser.error('Hours must be within(0,24]')
    if OUT.exists():parser.error('Existing overnight supervisor preserved; refusing duplicate launch')
    try:main(args.hours)
    except BaseException as error:
        OUT.mkdir(parents=True,exist_ok=True);w.OUT=OUT
        w.status('failed',error=repr(error),traceback=traceback.format_exc(),objective_achieved=False)
        for child in w.CHILDREN:
            if child.poll() is None:child.terminate()
        raise
