"""Learn braking with identical navigation weights and paired nonregression checks.

Selection constraints apply to offline model evaluations only. All vehicle
actions remain direct neural outputs; there is no runtime veto or rule driver.
"""
from pathlib import Path
import shutil
import traceback

import run_validated_imitation as w

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results/navigation_preserving_safety_v108'
SOURCE=ROOT/'results/safety_bottleneck_v103/r3_joint_frozen'


def stalls(m):
    return sum(m['end_reason_counts'].get(k,0) for k in
               ['route_blocked','idle','stuck','time_limit'])/m['episodes']


def eligible(candidate,baseline):
    """Compare full evaluations on the same seeds; do not trade away completion."""
    return (candidate['evaluation_valid'] and candidate['course_profile_verified'] and
            baseline['evaluation_valid'] and baseline['course_profile_verified'] and
            candidate['episodes']==baseline['episodes'] and
            candidate['end_reason_counts'].get('route_blocked',0)/candidate['episodes'] <=
            baseline['end_reason_counts'].get('route_blocked',0)/baseline['episodes'] and
            stalls(candidate)<=stalls(baseline) and
            candidate['arrival_rate']>=baseline['arrival_rate'] and
            candidate['red_light_rate']<=baseline['red_light_rate'] and
            candidate['collision_rate']<baseline['collision_rate'])


def goal(m):
    return (m['evaluation_valid'] and m['course_profile_verified'] and
            m['collision_rate']<.05 and m['arrival_rate']>=.95 and stalls(m)==0)


def navigation_invariance(source,candidate):
    import torch
    original=torch.load(source/'model.pt',map_location='cpu',weights_only=True)
    result=[]
    for checkpoint in ['model.pt','model_last.pt']:
        revised=torch.load(candidate/checkpoint,map_location='cpu',weights_only=True)
        names=[k for k in original if k.startswith(('trunk.','lane_head.','lane_adapter.'))]
        changed=[k for k in names if not torch.equal(original[k],revised[k])]
        row=dict(checkpoint=checkpoint,checked_tensors=len(names),changed_tensors=changed,passed=not changed)
        result.append(row)
        if not names or changed:raise RuntimeError('Frozen navigation changed: '+str(row))
    return result


def main():
    OUT.mkdir(parents=True,exist_ok=False);w.OUT=OUT;w.REPORT.clear()
    shutil.copyfile(__file__,OUT/Path(__file__).name)
    config=w.read(SOURCE/'run_config.json');data=[Path(x['path']) for x in config['data_sources']]
    for i in range(4):
        folder=ROOT/f'results/diverse_imitation_v107/native_{i}'
        audit=w.read(folder/'data_audit.json')
        projection=w.read(ROOT/f'results/diverse_imitation_v107/action_audit_{i}/audit_result.json')
        if not audit['alignment']['passed'] or not projection['passed']:
            raise RuntimeError('New demonstrations failed audit')
        data.append(folder/'demonstrations.npz')
    w.status('paired_baseline_evaluation',source=str(SOURCE),source_sha256=w.sha(SOURCE/'model.pt'),
        development_seeds=[38800,38899],private_seeds=[39700,39799],
        frozen_navigation=True,no_action_assistance=True,student_modes=[96,0],
        original_road_hv_route_preserved=True,
        candidate_selection='strictly fewer collisions; no increase in route blockage, stalls, red violations or decrease in arrival on paired seeds')
    baseline=w.evaluate(SOURCE,'baseline',100,38800)
    w.status('longitudinal_only_imitation',baseline_development=baseline)
    common=['train_local_bc.py','--init-run',SOURCE,'--data',*data,
        '--epochs','60','--lr','.0001','--set-residual','--metric-sensors',
        '--train-accel-backbone','--brake-weight-scope','all','--launch-weight','1',
        '--stop-hold-fraction','.25','--censored-stop-loss','--reference-coef','.01',
        '--predictive-reaction-time','1','--predictive-comfort-decel','3.5',
        '--spatial-safety-horizon','2','--eval-episodes','5','--eval-seed-start','38800']
    settings=[('braking',4,2,1),('strong_braking',8,8,4)]
    jobs=[]
    for name,underbrake,predictive,spatial in settings:
        jobs.append(w.launch([*common,'--output',OUT/name,'--underbrake-weight',str(underbrake),
            '--predictive-safety-coef',str(predictive),'--spatial-safety-coef',str(spatial)],OUT/f'{name}.log'))
    w.wait(jobs,21600)
    candidates=[]
    from freeze_candidate import freeze
    for name,*_ in settings:
        model=OUT/name
        w.REPORT.setdefault('navigation_invariance',{})[name]=navigation_invariance(SOURCE,model)
        candidates.append(w.evaluate(model,name,100,38800))
        last=OUT/(name+'_last');freeze(model,last,checkpoint='model_last.pt')
        candidates.append(w.evaluate(last,name+'_last',100,38800))
        w.status('paired_development_comparison',candidates=candidates)
    accepted=[c for c in candidates if eligible(c['metrics'],baseline['metrics'])]
    if not accepted:
        w.status('completed_no_nonregressing_candidate',qualified=False,objective_achieved=False,
            candidates=candidates,selected=baseline,reason='Retained baseline because safety/completion constraints failed')
        return
    best=max(accepted,key=lambda c:(goal(c['metrics']),-c['metrics']['collision_rate'],c['metrics']['arrival_rate']))
    w.status('paired_private_verification',selected=best,candidates=candidates)
    paired=w.evaluate(SOURCE,'paired_baseline',100,39700)
    final=w.evaluate(Path(best['run']),'selected',100,39700)
    improved=eligible(final['metrics'],paired['metrics']);qualified=improved and goal(final['metrics'])
    w.status('completed_target_verified' if qualified else 'completed_experiments_goal_not_met',
        qualified=qualified,objective_achieved=qualified,paired_nonregression_passed=improved,
        final_evaluation=final,paired_baseline=paired,selected=best,candidates=candidates,
        promoted=bool(qualified))
    if qualified:
        w.save(ROOT/'results/recommended_run.json',dict(qualified=True,recommended_run=final['run'],
            latest_evaluated_run=final['run'],actual_evaluation=final['metrics'],
            report=str(OUT/'status.json'),no_assisted_claim=True))


if __name__=='__main__':
    try:main()
    except BaseException as error:
        OUT.mkdir(parents=True,exist_ok=True);w.OUT=OUT
        w.status('failed',error=repr(error),traceback=traceback.format_exc(),objective_achieved=False)
        for child in w.CHILDREN:
            if child.poll() is None:child.terminate()
        raise
