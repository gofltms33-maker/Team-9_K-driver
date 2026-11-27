"""Local bounded supervisor: replay -> classify -> verified teacher data -> fit -> paired tests.

Never drives the AV, edits roads/HVs, or launches a language model. Private
metrics never choose a checkpoint. Each next round retires old development
seeds used for demonstrations and allocates fresh development/private seeds.
"""
from pathlib import Path
from datetime import datetime,timedelta,timezone
from collections import Counter
import argparse
import os
import time
import shutil
import traceback
import run_validated_imitation as w
import run_collision_state_learning as learning
from run_nine_hour_safety import execution_state,process_alive
from run_navigation_preserving_safety import stalls

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results/overnight_adaptive_v118'
INITIAL=ROOT/'results/collision_state_learning_v117'
STATE={}


def status(phase,**items):
    STATE.update(status=phase,pid=os.getpid(),updated_kst=datetime.now(timezone(timedelta(hours=9))).isoformat(),**items)
    w.save(OUT/'status.json',STATE);print(phase,flush=True)


def goal(result):
    m=result['metrics']
    return (m['evaluation_valid'] and m['course_profile_verified'] and m['collision_rate']<.05
            and m['arrival_rate']>=.95 and stalls(m)==0 and m['red_light_rate']==0)


def choose_source(report):
    """Private test can veto a regression; never select a different checkpoint by it."""
    c,b=report['final_evaluation'],report['paired_baseline']
    dev=report['selected']
    if learning.nonregression(c,b) and learning.quality(c)>learning.quality(b):return dev
    return report['baseline_development']


def reserve(index):
    lo=1020000+index*1000;hi=lo+400
    path=ROOT/'results/safety_evaluation_seed_audit.json';a=w.read(path)
    if any(not(hi<=x or lo>=y) for x,y in a['reserved_half_open_ranges']):raise RuntimeError('Reserved seed overlap')
    a['reserved_half_open_ranges'].append([lo,hi])
    a.setdefault('overnight_adaptive_rounds',[]).append(dict(round=index,development=[lo,lo+299],private=[lo+300,hi-1]))
    w.save(path,a);return lo,lo+300


def wait_initial():
    retries=0
    while True:
        report=w.read(INITIAL/'status.json');phase=report['status']
        if phase.startswith('completed'):return report
        if phase=='failed' and retries<2:
            retries+=1;status('recovering_interrupted_initial_evaluation',attempt=retries)
            p=w.launch(['run_collision_state_learning.py','--resume'],OUT/f'initial_resume_{retries}.log')
            try:w.wait([p],8*3600)
            except Exception:pass
            continue
        if phase in ['failed','paused_by_user'] or process_alive(report.get('pid')) is False:
            raise RuntimeError('Initial resumed experiment is not running: '+phase)
        status('waiting_for_resumed_v117',initial_phase=phase)
        time.sleep(30)


def audit_source(current,folder):
    folder.mkdir(parents=True,exist_ok=True)
    p=w.launch(['audit_saved_failures.py',current['run'],Path(current['evaluation'])/'evaluation.json',
        '--output',folder,'--workers','4'],learning.OUT/'failure_replay.log')
    w.wait([p],3*3600);a=w.read(folder/'result.json')
    if not a['all_replays_match']:raise RuntimeError('Failure replay mismatch')
    roads=Counter((r['road'],r['end_reason']) for r in a['results'])
    contacts=Counter(c['type'] for r in a['results'] for c in r['contacts'])
    hist=STATE.setdefault('repeated_locations',{})
    for (road,reason),count in roads.items():hist[road+'|'+reason]=hist.get(road+'|'+reason,0)+count
    analysis=dict(source=current['run'],episode_count=current['metrics']['episodes'],replayed_failures=len(a['results']),
        all_replays_match=True,contact_types=dict(contacts),locations=[dict(road=k[0],reason=k[1],count=v) for k,v in roads.most_common()],
        cumulative_repeated_locations=hist,
        interpretation='Measured failure groups drive offline demonstration timing and choice of trainable neural adapter. Road names are report labels, never policy rules.')
    w.save(folder/'analysis.json',analysis);status('failures_classified',latest_analysis=analysis)
    return a


def cohorts(a):
    side=[];front=[];junction=[];blocked=[]
    for r in a['results']:
        if r['end_reason'] in ['route_blocked','idle','stuck','time_limit']:blocked.append(r['seed'])
        elif any(c['type']=='side' for c in r['contacts']):side.append(r['seed'])
        elif any(c['type']=='junction' for c in r['contacts']):junction.append(r['seed'])
        else:front.append(r['seed'])
    groups=[]
    for name,seeds,leads,window,exact in [('side',side,[1,4],8,True),('braking',front,[4,16],16,True),
            ('crossing',junction,[8,16],16,True),('blocked',blocked,[80],120,False)]:
        for lead in leads:
            if seeds:groups.append((name+'_'+str(lead),seeds,lead,window,exact))
    return groups


def collect(a):
    jobs=[]
    # Limit concurrent collectors: their teacher operates only during collection.
    groups=cohorts(a);data=[]
    for offset in range(0,len(groups),3):
        jobs=[]
        for name,seeds,lead,window,exact in groups[offset:offset+3]:
            folder=learning.OUT/name
            p=w.launch(['collect_audited_course_data.py',learning.SOURCE,folder,'--kind','recovery',
                '--episodes','0','--seed-start','70000','--include-seeds',*map(str,seeds),
                '--lead-steps',str(lead),'--window-steps',str(window),'--min-successful-episodes','1'],learning.OUT/(name+'.collection.log'))
            jobs.append((folder,p,exact))
        for folder,p,exact in jobs:
            try:
                w.wait([p],3*3600);data.append(learning.check_data(folder,require_exact_first=exact))
            except Exception as error:
                w.REPORT.setdefault('rejected_datasets',[]).append(dict(folder=str(folder),error=repr(error)))
                status('rejected_unverified_teacher_data',last_rejection=str(error))
    if not data:raise RuntimeError('No independently validated recovery data')
    return data


def train_and_compare(current,index,history_data):
    folder=OUT/f'round_{index:03d}';folder.mkdir();learning.OUT=folder;learning.SOURCE=Path(current['run'])
    w.OUT=folder;w.REPORT.clear();w.REPORT['protected_files']=STATE['protected_files']
    protected=STATE['protected_files']
    protected[str(learning.SOURCE/'model.pt')]=w.sha(learning.SOURCE/'model.pt')
    dev,private=reserve(index)
    w.status('replaying_failures',source=current['run'],development_seeds=[dev,dev+299],private_seeds=[private,private+99])
    status('replaying_previous_failures',active_round=index,round_report=str(folder/'status.json'))
    a=audit_source(current,folder/'failure_audit')
    status('collecting_failure_specific_demonstrations')
    newdata=collect(a);data=list(dict.fromkeys(history_data[-4:]+newdata))
    status('training_corrective_candidates',training_data=list(map(str,data)))
    jobs=[]
    settings=[('lateral',[]),('joint',['--joint'])]
    for name,flags in settings:
        jobs.append(w.launch(['train_trust_region_correction.py','--init-run',learning.SOURCE,'--output',folder/name,
            '--data',*data,'--rehearsal',ROOT/'results/evidence_guided_refinement_v115/rehearsal/rehearsal.npz',
            '--iterations','1600',*flags],folder/(name+'.train.log')))
    for p in jobs:w.wait([p],3*3600)
    for name,flags in settings:learning.invariance(folder/name,bool(flags))
    status('paired_300_seed_development')
    baseline=learning.evaluate(learning.SOURCE,'baseline_development',300,dev)
    candidates=[]
    for name,_ in settings:candidates.append(learning.evaluate(folder/name,name+'_development',300,dev))
    # If corrective BC repeatedly cannot improve, add actual on-policy rewards.
    if index%2==0 and not any(learning.nonregression(c,baseline) and learning.quality(c)>learning.quality(baseline) for c in candidates):
        status('reward_finetuning_after_failed_corrective_bc')
        cfg=w.read(learning.SOURCE/'run_config.json');demos=list(dict.fromkeys([Path(x['path']) for x in cfg['data_sources']]+data))
        counts=Counter(c['type'] for r in a['results'] for c in r['contacts'])
        mode='--lane-only' if counts['side']>=counts['collision']+counts['junction'] else '--acceleration-only'
        job=w.launch(['train_warmstart_ppo.py','--init-run',learning.SOURCE,'--output',folder/'ppo','--data',*demos,
            mode,'--steps','131072','--rollout','2048','--epochs','2','--lr','.000002','--gamma','.999',
            '--clip','.03','--target-kl','.002','--bc-coef','.05','--lane-weight-power','0','--std','.03',
            '--lane-temperature','1','--warmup-updates','4','--eval-every','16','--eval-episodes','50',
            '--eval-seed-start',str(dev),'--train-seed-start','70000','--train-seed-stop','90000',
            '--collision-penalty','300','--failure-penalty','300','--arrival-bonus','300',
            '--red-light-penalty','300','--risk-shaping','potential','--strict-speed-mode'],folder/'ppo.train.log')
        w.wait([job],6*3600);candidates.append(learning.evaluate(folder/'ppo','ppo_development',300,dev))
    selected=max([baseline]+[c for c in candidates if learning.nonregression(c,baseline)],key=learning.quality)
    w.status('final_100_seed_comparison',baseline_development=baseline,candidates=candidates,selected=selected)
    status('paired_100_seed_final')
    paired=learning.evaluate(learning.SOURCE,'baseline_final',100,private)
    final=paired if selected is baseline else learning.evaluate(Path(selected['run']),'selected_final',100,private)
    qualified=goal(selected) and goal(final) and learning.nonregression(final,paired)
    w.status('completed_target_verified' if qualified else 'completed_experiments_goal_not_met',qualified=qualified,
        baseline_development=baseline,selected=selected,paired_baseline=paired,final_evaluation=final,candidates=candidates)
    report=dict(w.REPORT);STATE.setdefault('rounds',[]).append(dict(round=index,report=str(folder/'status.json'),
        qualified=qualified,selected_development=selected['metrics'],final=final['metrics'],baseline=paired['metrics']))
    return report,newdata


def main(hours,resume=False):
    OUT.mkdir(parents=True,exist_ok=resume)
    if resume:
        old=w.read(OUT/'status.json')
        if old.get('rounds') or old['status'] not in ['waiting_for_resumed_v117','recovering_interrupted_initial_evaluation']:
            raise RuntimeError('Only the initial wait may use this restart mode')
        w.save(OUT/'status_before_wait_restart.json',old)
    w.OUT=OUT;w.REPORT.clear()
    shutil.copyfile(__file__,OUT/Path(__file__).name)
    start=datetime.now(timezone(timedelta(hours=9)));deadline=time.monotonic()+hours*3600
    STATE.update(started_kst=start.isoformat(),planning_until_kst=(start+timedelta(hours=hours)).isoformat(),
        hours=hours,protected_files=dict(w.read(INITIAL/'status.json')['protected_files']),
        no_road_hv_or_simulator_rule_edits=True,no_runtime_driving_assistance=True,
        evaluation_protocol='paired 300 development + separate paired 100 final; fresh seeds each round')
    execution_state(True)
    try:
        report=wait_initial();current=choose_source(report)
        STATE['initial_result']=dict(final=report['final_evaluation']['metrics'],selected=current['run'])
        history=[];index=0;errors=0
        while True:
            if goal(report['selected']) and goal(report['final_evaluation']) and learning.nonregression(report['final_evaluation'],report['paired_baseline']):
                status('completed_target_verified',qualified=True,final_evaluation=report['final_evaluation'])
                w.save(ROOT/'results/recommended_run.json',dict(qualified=True,recommended_run=report['final_evaluation']['run'],
                    actual_evaluation=report['final_evaluation']['metrics'],report=str(OUT/'status.json')));return
            if time.monotonic()>=deadline-45*60:break
            index+=1
            try:
                report,newdata=train_and_compare(current,index,history);current=choose_source(report)
                history.extend(newdata);errors=0
                status('round_completed_goal_not_met',current_source=current['run'],latest_final=report['final_evaluation']['metrics'])
            except Exception as error:
                if 'Protected input changed' in str(error):raise
                errors+=1;STATE.setdefault('execution_errors',[]).append(dict(round=index,error=repr(error),traceback=traceback.format_exc()))
                status('round_failed_preserving_previous_model',consecutive_errors=errors)
                if errors>=3:raise RuntimeError('Three consecutive round execution failures; consult recorded errors')
                # A failed data round needs genuinely new traffic examples, not
                # repeated identical failed collection with changed parameters.
                learning.OUT=OUT/f'round_{index:03d}';w.OUT=learning.OUT
                probe_seed=1100000+index*1000
                current=learning.evaluate(Path(current['run']),'replacement_training_probe',100,probe_seed)
        status('completed_time_window_goal_not_met',qualified=False,current_source=current['run'],
            note='Previous models and all measured results are preserved. Target was not achieved within this scheduling window.')
    finally:execution_state(False)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--hours',type=float,default=12);p.add_argument('--resume-wait',action='store_true');a=p.parse_args()
    if OUT.exists() and not a.resume_wait:raise SystemExit('Existing overnight run preserved; duplicate launch refused')
    try:main(a.hours,a.resume_wait)
    except BaseException as error:
        OUT.mkdir(parents=True,exist_ok=True);status('failed',error=repr(error),traceback=traceback.format_exc(),qualified=False)
        for job in w.CHILDREN:
            if job.poll() is None:job.terminate()
        raise
