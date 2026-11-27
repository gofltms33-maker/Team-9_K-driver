"""Ordered safety learning: lateral decisions -> stops -> roundabout -> PPO.

All focus choices are offline training-data choices. No road/HV/rule changes,
scenario-dependent policy switch, runtime action veto or teacher takeover is
used in evaluation. Each accepted stage must preserve paired route/completion.
"""
from pathlib import Path
import argparse
import shutil
import time
import traceback

import run_validated_imitation as w
from run_navigation_preserving_safety import goal,stalls

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results/sequential_safety_v109'
SOURCE=ROOT/'results/safety_bottleneck_v103/r3_joint_frozen'
DEV=38900


def nonregressing(candidate,baseline):
    return (candidate['evaluation_valid'] and candidate['course_profile_verified'] and
        baseline['evaluation_valid'] and baseline['course_profile_verified'] and candidate['episodes']==baseline['episodes'] and
        candidate['end_reason_counts'].get('route_blocked',0)<=baseline['end_reason_counts'].get('route_blocked',0) and
        stalls(candidate)<=stalls(baseline) and candidate['arrival_rate']>=baseline['arrival_rate'] and
        candidate['collision_rate']<=baseline['collision_rate'])


def rank(candidate):
    m=candidate['metrics']
    return goal(m),-m['collision_rate'],m['arrival_rate'],-m['red_light_rate'],-stalls(m)


def wait_collection_jobs(jobs):
    deadline=time.monotonic()+14400
    while any(p is not None and p.poll() is None for p,_ in jobs):
        if time.monotonic()>deadline:
            for p,_ in jobs:
                if p is not None and p.poll() is None:p.terminate()
            break
        time.sleep(5)


def collect(source,index,seeds,seed_start=None,minimum_successful_episodes=5):
    jobs=[];chunks=2 if index==1 else 1
    for part in range(chunks):
        folder=OUT/f's{index}_recovery_{part}'
        if folder.exists():
            saved=folder/'status.json'
            if not saved.exists() or w.read(saved).get('status')!='completed':
                raise RuntimeError('Preserving incomplete collection: '+str(folder))
            jobs.append((None,folder))
            w.REPORT.setdefault('reused_collections',[]).append(str(folder))
            continue
        p=w.launch(['collect_audited_course_data.py',source,folder,'--kind','recovery',
            '--episodes',str(12//chunks),'--seed-start',str((51000+20*(index-1) if seed_start is None else seed_start)+6*part),
            '--include-seeds',*map(str,seeds[part::chunks]),'--lead-steps','2','--window-steps','120',
            '--min-successful-episodes',str(minimum_successful_episodes)],
            OUT/f's{index}_recovery_{part}.log')
        jobs.append((p,folder))
    wait_collection_jobs(jobs);accepted=[]
    for p,folder in jobs:
        if (p is not None and p.poll()!=0) or not (folder/'status.json').exists():
            w.REPORT.setdefault('unusable_collections',[]).append(str(folder));continue
        audit=w.read(folder/'data_audit.json')
        if not audit['alignment']['passed']:raise RuntimeError('Recovery alignment failed')
        projection=OUT/(folder.name+'_action_audit')
        if not (projection/'audit_result.json').exists():
            p=w.launch(['audit_demonstration_actions.py',folder/'demonstrations.npz',projection],
                       OUT/(folder.name+'_action_audit.log'))
            try:w.wait([p],3600)
            except Exception as error:
                if p.poll() is None:p.terminate()
                w.REPORT.setdefault('unusable_collections',[]).append(dict(folder=str(folder),error=repr(error)))
                continue
        checked=w.read(projection/'audit_result.json')
        # Focused stop/merge windows may lack the general audit's six diverse
        # events. Record that limited coverage; never accept a failed replay.
        valid=bool(checked['results']) and all(r['passed'] for r in checked['results'])
        w.REPORT.setdefault('collection_action_checks',[]).append(dict(folder=str(folder),
            events=checked['tested_events'],all_checked_events_passed=valid,
            general_coverage_passed=checked['passed']))
        if valid:accepted.append(folder/'demonstrations.npz')
        else:w.REPORT.setdefault('unusable_collections',[]).append(str(folder))
    return accepted


def roundabout_subset(paths):
    import numpy as np
    folder=OUT/'roundabout_teacher_subset';folder.mkdir()
    pieces=[];parents=[]
    for path in paths:
        audit=w.read(path.parent/'data_audit.json')
        if not audit['alignment']['passed']:raise RuntimeError('Parent alignment failed')
        with np.load(path) as archive:
            road=archive['audit_before_road'].astype(str)
            mask=np.char.startswith(road,'ar_rb')|np.char.startswith(road,':ar_rb')
            pieces.append({k:archive[k][mask] for k in ['observation','action','episode','accel_valid','lane_valid']})
            parents.append(dict(path=str(path),sha256=w.sha(path),copied_rows=int(mask.sum())))
    data={k:np.concatenate([part[k] for part in pieces]) for k in pieces[0]}
    if len(data['episode'])<100:raise RuntimeError('Insufficient roundabout teacher data')
    np.savez_compressed(folder/'demonstrations.npz',**data)
    cfg=w.read(paths[0].parent/'run_config.json');cfg['stage']='Offline subset of audited teacher rows; no new labels or observations'
    w.save(folder/'run_config.json',cfg)
    w.save(folder/'data_audit.json',dict(label_schema=3,training_focus=True,expert_suffix=False,
        samples=len(data['episode']),parents=parents,
        alignment=dict(passed=True,scope='Exact copied rows of independently aligned parent datasets'),
        no_synthetic_labels=True,no_runtime_selection=True))
    return folder/'demonstrations.npz'


def assert_frozen(source,model,index):
    if index not in (1,2):return
    import torch
    before=torch.load(source/'model.pt',map_location='cpu',weights_only=True)
    prefixes=('accel_trunk.','accel_head.','accel_adapter.') if index==1 else ('trunk.','lane_head.','lane_adapter.')
    names=[k for k in before if k.startswith(prefixes)]
    for checkpoint in ['model.pt','model_last.pt']:
        after=torch.load(model/checkpoint,map_location='cpu',weights_only=True)
        changed=[k for k in names if not torch.equal(before[k],after[k])]
        w.REPORT.setdefault('frozen_parameter_checks',[]).append(dict(stage=index,checkpoint=checkpoint,
            checked_tensors=len(names),changed_tensors=changed,passed=bool(names) and not changed))
        if not names or changed:raise RuntimeError('Frozen actor parameters changed')


def train_stage(source,data,index):
    model=OUT/f's{index}_bc'
    args=['train_local_bc.py','--init-run',source,'--output',model,'--data',*data,
        '--epochs','40','--set-residual','--metric-sensors','--censored-stop-loss',
        '--brake-weight-scope','all','--underbrake-weight','4','--launch-weight','1',
        '--reference-coef','.01','--predictive-reaction-time','1','--predictive-comfort-decel','3.5',
        '--spatial-safety-horizon','2','--eval-episodes','5','--eval-seed-start',str(DEV)]
    if index==1:
        args+=['--lr','.00003','--lane-lr','.00003','--train-lanes','--lane-only',
            '--train-lane-backbone','--lane-class-fraction','.01','--lane-weight-power','0',
            '--priority-fraction','.4','--lane-safety-coef','2','--lane-reference-coef','.03']
    elif index==2:
        args+=['--lr','.0001','--train-accel-backbone','--stop-hold-fraction','.4',
            '--stop-hold-weight','8','--priority-fraction','.2',
            '--predictive-safety-coef','4','--spatial-safety-coef','2']
    else:
        args+=['--lr','.00002','--lane-lr','.000005','--train-lanes','--train-backbone',
            '--lane-class-fraction','.01','--lane-weight-power','0','--priority-fraction','.35',
            '--stop-hold-fraction','.2','--stop-hold-weight','4','--predictive-safety-coef','4',
            '--lane-safety-coef','2','--spatial-safety-coef','2','--lane-reference-coef','.05']
    p=w.launch(args,OUT/f's{index}_bc.log');w.wait([p],21600);assert_frozen(source,model,index)
    return model


def compare(model,index,before):
    from freeze_candidate import freeze
    candidates=[w.evaluate(model,f's{index}',100,DEV)]
    last=OUT/f's{index}_last';freeze(model,last,checkpoint='model_last.pt')
    candidates.append(w.evaluate(last,f's{index}_last',100,DEV))
    admissible=[before]+[c for c in candidates if nonregressing(c['metrics'],before['metrics'])]
    best=max(admissible,key=rank)
    w.REPORT.setdefault('stage_comparisons',[]).append(dict(stage=index,before=before,
        candidates=candidates,selected=best,retained_previous=best is before))
    w.status('stage_comparison_completed',stage=index,selected=best)
    return best


def main(resume=False):
    OUT.mkdir(parents=True,exist_ok=resume);w.OUT=OUT;w.REPORT.clear()
    if resume:
        prior=w.read(OUT/'status.json')
        if prior.get('status')!='failed':raise RuntimeError('Resume requires a failed, inactive manager')
        if prior.get('source_sha256')!=w.sha(SOURCE/'model.pt'):raise RuntimeError('Source checkpoint changed')
        w.REPORT.update(prior)
        w.REPORT.setdefault('recovered_errors',[]).append(dict(error=prior.get('error'),
            traceback=prior.get('traceback'),stage_errors=prior.get('stage_errors',[])))
        for key in ['error','traceback','stage_errors','objective_achieved']:w.REPORT.pop(key,None)
        archive=OUT/'manager_before_resume.py'
        if not archive.exists():shutil.copyfile(OUT/Path(__file__).name,archive)
    shutil.copyfile(__file__,OUT/Path(__file__).name)
    w.status('stage1_collecting_unsafe_lane_recoveries',source=str(SOURCE),source_sha256=w.sha(SOURCE/'model.pt'),
        ordered_stages=['unsafe_lane_decisions','stop_hold_and_restart','roundabout_interaction','PPO_refinement'],
        development_seeds=[DEV,DEV+99],private_seeds=[39800,39899],
        original_road_hv_route_preserved=True,student_modes=[96,0],no_runtime_action_assistance=True,
        progression='Keep previous policy whenever paired route/stall/completion/collision constraints fail; continue next stage')
    known={1:set(),2:set(),3:set()}
    for name in ['anchored_safety_ppo_v105','safety_bottleneck_v103']:
        report=w.read(ROOT/f'results/{name}/final_collision_audit/result.json')
        if not report['all_replays_match']:raise RuntimeError('Failure replay mismatch')
        for row in report['results']:
            if any(c['type']=='side' for c in row['contacts']):known[1].add(row['seed'])
            if any(c['type']=='collision' for c in row['contacts']):known[2].add(row['seed'])
            if any(c['lane'].startswith(('ar_rb',':ar_rb')) for c in row['contacts']):known[3].add(row['seed'])
    native=[ROOT/f'results/diverse_imitation_v107/native_{i}/demonstrations.npz' for i in range(4)]
    data=[Path(x['path']) for x in w.read(SOURCE/'run_config.json')['data_sources']]+native
    try:data+=collect(SOURCE,1,sorted(known[1]))
    except Exception as error:
        w.REPORT.setdefault('stage_errors',[]).append(dict(stage=1,part='initial_collection',error=repr(error)))
    w.status('stage1_paired_baseline_evaluation')
    best=w.evaluate(SOURCE,'initial',100,DEV);w.REPORT['initial_development']=best
    for index in (1,2,3):
        previous=best
        try:
            if index>1:
                w.status('collecting_next_stage_recoveries',stage=index,selected=best)
                data+=collect(Path(best['run']),index,sorted(known[index]))
            if index==3:data.append(roundabout_subset(native))
            w.status('training_ordered_imitation_stage',stage=index,selected=best,data_paths=list(map(str,data)))
            model=train_stage(Path(best['run']),data,index);best=compare(model,index,best)
        except Exception as error:
            best=previous;w.REPORT.setdefault('stage_errors',[]).append(dict(stage=index,error=repr(error),traceback=traceback.format_exc()))
            w.status('stage_rejected_continuing_from_previous',stage=index,selected=best)
    w.status('stage4_ppo_refinement',stage=4,selected=best)
    try:
        model=OUT/'s4_ppo'
        p=w.launch(['train_warmstart_ppo.py','--init-run',best['run'],'--data',*data,'--output',model,
            '--steps','131072','--rollout','2048','--epochs','3','--lr','.000005','--gamma','.999',
            '--gae-lambda','.95','--clip','.05','--target-kl','.005','--bc-coef','1','--std','.025',
            '--lane-temperature','.5','--warmup-updates','2','--eval-every','16',
            '--eval-episodes','5','--eval-seed-start',str(DEV),'--reward-scale','.001',
            '--risk-coef','.5','--predictive-safety-coef','2','--predictive-reaction-time','1',
            '--censored-stop-loss','--strict-speed-mode','--collision-penalty','2000',
            '--red-light-penalty','1000','--failure-penalty','2000','--arrival-bonus','2000',
            '--seed','1909','--train-seed-start','40000','--train-seed-stop','900000'],OUT/'s4_ppo.log')
        w.wait([p],28800);best=compare(model,4,best)
    except Exception as error:
        w.REPORT.setdefault('stage_errors',[]).append(dict(stage=4,error=repr(error),traceback=traceback.format_exc()))
    w.status('paired_private_verification',selected=best)
    paired=w.evaluate(SOURCE,'paired_initial',100,39800)
    final=w.evaluate(Path(best['run']),'selected',100,39800)
    preserved=nonregressing(final['metrics'],paired['metrics']);qualified=preserved and goal(final['metrics'])
    w.status('completed_target_verified' if qualified else 'completed_experiments_goal_not_met',
        qualified=qualified,objective_achieved=qualified,paired_nonregression_passed=preserved,
        final_evaluation=final,paired_baseline=paired,selected=best)
    if qualified:
        w.save(ROOT/'results/recommended_run.json',dict(qualified=True,recommended_run=final['run'],
            latest_evaluated_run=final['run'],actual_evaluation=final['metrics'],
            report=str(OUT/'status.json'),no_assisted_claim=True))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    if OUT.exists() and not args.resume:parser.error('Existing experiment preserved; use --resume only after failure')
    if args.resume and (not (OUT/'status.json').exists() or w.read(OUT/'status.json').get('status')!='failed'):
        parser.error('Only a failed inactive manager can be resumed')
    try:main(args.resume)
    except BaseException as error:
        OUT.mkdir(parents=True,exist_ok=True);w.OUT=OUT
        w.status('failed',error=repr(error),traceback=traceback.format_exc(),objective_achieved=False)
        for child in w.CHILDREN:
            if child.poll() is None:child.terminate()
        raise
