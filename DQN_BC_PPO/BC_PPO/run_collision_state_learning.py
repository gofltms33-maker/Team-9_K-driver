"""Learn from reproduced PPO crash states, keeping road and traffic fixtures fixed."""
from pathlib import Path
from collections import Counter
import shutil
import time
import traceback
import numpy as np
import torch
import run_validated_imitation as w
from freeze_candidate import freeze
from run_nine_hour_safety import execution_state
from run_broad_seed_refinement import blocks
from run_navigation_preserving_safety import stalls

ROOT=Path(__file__).resolve().parent
SOURCE=ROOT/'results/evidence_guided_refinement_v115/lateral_ppo_development_frozen'
OUT=ROOT/'results/collision_state_learning_v117'
DEV=1010000
PRIVATE=1010300


def preserved():
    for p,h in w.REPORT['protected_files'].items():
        if w.sha(Path(p))!=h:raise RuntimeError('Protected input changed: '+p)


def evaluate(source,name,n,seed):
    preserved();run=OUT/(name+'_frozen')
    if not run.exists():freeze(source,run)
    elif w.sha(run/'model.pt')!=w.sha(source/'model.pt'):raise RuntimeError('Frozen source differs on resume')
    output=run/f'evaluation_{seed}_{n}'
    if (output/'acceptance.json').exists():job=None
    elif output.exists():job=w.launch(['resume_parallel_evaluation.py',output,'--workers','4'],OUT/(name+'.resume.log'))
    else:job=w.launch(['parallel_evaluate.py',run,'--episodes',str(n),'--seed-start',str(seed),
                  '--workers','4','--require-course-profile','--output',output],OUT/(name+'.log'))
    if job:w.wait([job],8*3600)
    m=w.read(output/'acceptance.json')
    if not m['evaluation_valid'] or not m['course_profile_verified']:raise RuntimeError('Invalid evaluation')
    return dict(run=str(run),metrics=m,evaluation=str(output),seed_blocks=blocks(w.read(output/'evaluation.json')['episodes']))


def nonregression(c,b):
    x,y=c['metrics'],b['metrics']
    return (x['episodes']==y['episodes'] and x['collision_rate']<=y['collision_rate']
        and x['arrival_rate']>=y['arrival_rate'] and stalls(x)<=stalls(y)
        and x['red_light_rate']<=y['red_light_rate'])


def quality(c):
    m=c['metrics'];return (m['arrival_rate'],-m['collision_rate'],-stalls(m))


def check_data(folder,require_exact_first=True):
    a=w.read(folder/'data_audit.json')
    if not a['alignment']['passed']:raise RuntimeError('Teacher data alignment failed')
    if not a['successful_episodes']:raise RuntimeError('No successful teacher rescues')
    target=OUT/(folder.name+'_action_check')
    p=w.launch(['audit_demonstration_actions.py',folder/'demonstrations.npz',target],OUT/(folder.name+'.audit.log'))
    w.wait([p],2*3600);audit=w.read(target/'audit_result.json')
    if not audit['results'] or not all(r['passed'] for r in audit['results']):raise RuntimeError('Teacher action projection failed')
    # Only first teacher rows are exact learner-visited states. Later rows are
    # explicitly teacher rollout data; never claim them to be student states.
    exact=[]
    with np.load(folder/'demonstrations.npz') as d:
        for seed in np.unique(d['episode']):
            trace=w.read(ROOT/f'outputs/trace_{SOURCE.name}_{seed}.json')
            first=int(np.flatnonzero(d['episode']==seed)[0]);step=next(r for r in a['episodes'] if r['seed']==int(seed))['handover_step']+1
            frame=next((f for f in trace['frames'] if f['step']==step),None)
            if frame is None and not require_exact_first:continue
            if frame is None or not np.allclose(d['observation'][first],frame['obs'],atol=1e-6,rtol=0):
                raise RuntimeError('Learner-state/teacher handover observation mismatch')
            exact.append(dict(seed=int(seed),step=step,learner_action=frame['action'],teacher_action=d['action'][first].tolist()))
    w.save(folder/'learner_state_alignment.json',dict(passed=True,first_rows=exact))
    w.REPORT.setdefault('verified_datasets',[]).append(dict(path=str(folder),samples=a['samples'],
        successful_teacher_episodes=a['successful_episodes'],action_checks_passed=True,
        exact_learner_first_states=len(exact),sampled_general_coverage=audit['passed']))
    return folder/'demonstrations.npz'


def invariance(candidate,joint):
    old=torch.load(SOURCE/'model.pt',map_location='cpu',weights_only=True)
    new=torch.load(candidate/'model.pt',map_location='cpu',weights_only=True)
    allowed=('lane_adapter.','accel_adapter.') if joint else ('lane_adapter.',)
    changed=[k for k in old if not k.startswith(allowed) and not torch.equal(old[k],new[k])]
    if changed:raise RuntimeError('Changed frozen backbone: '+str(changed))


def main(resume=False):
    OUT.mkdir(parents=True,exist_ok=resume);w.OUT=OUT;w.REPORT.clear();torch.set_num_threads(1)
    if resume:
        previous=w.read(OUT/'status.json')
        if previous['status'] not in ['paused_by_user','failed']:raise RuntimeError('Cannot resume an active or finished run')
        count=previous.get('resume_count',0)+1;w.save(OUT/f'status_before_resume_{count}.json',previous)
        w.REPORT.update(previous);w.REPORT['resume_count']=count
        w.REPORT.pop('error',None);w.REPORT.pop('traceback',None)
        shutil.copyfile(__file__,OUT/f'run_collision_state_learning.resume_{count}.py')
    else:shutil.copyfile(__file__,OUT/Path(__file__).name)
    evidence=w.read(ROOT/'results/evidence_guided_refinement_v115/ppo_collision_audit/result.json')
    if not evidence['all_replays_match'] or evidence['failures']!=41:raise RuntimeError('Incomplete source collision audit')
    protected=dict(w.read(ROOT/'results/evidence_guided_refinement_v115/status.json')['preserved_file_hashes'])
    protected[str(SOURCE/'model.pt')]=w.sha(SOURCE/'model.pt')
    cfg=w.read(SOURCE/'run_config.json')
    from parallel_evaluate import validate_course_config
    validate_course_config(cfg)
    w.status('collecting_exact_failure_state_demonstrations',source=str(SOURCE),source_sha256=w.sha(SOURCE/'model.pt'),
        protected_files=protected,source_development_collision_rate=41/300,
        source_collision_types=dict(Counter(c['type'] for r in evidence['results'] for c in r['contacts'])),
        development_seeds=[DEV,DEV+299],final_seeds=[PRIVATE,PRIVATE+99],
        retired_training_seeds=[37400,37699],no_runtime_assists=True,road_hv_rules_unchanged=True)
    execution_state(True)
    try:
        preserved();seeds=[r['seed'] for r in evidence['results']];jobs=[]
        for lead in [1,4]:
            folder=OUT/f'collision_lead_{lead}'
            if (folder/'learner_state_alignment.json').exists() and (folder/'demonstrations.npz').exists():
                jobs.append((folder,None));continue
            job=w.launch(['collect_audited_course_data.py',SOURCE,folder,'--kind','recovery','--episodes','0',
                '--seed-start','68000','--include-seeds',*map(str,seeds),'--lead-steps',str(lead),
                '--window-steps','8','--min-successful-episodes','5'],OUT/(folder.name+'.log'))
            jobs.append((folder,job))
        data=[]
        for folder,job in jobs:
            if job:w.wait([job],4*3600);data.append(check_data(folder))
            else:data.append(folder/'demonstrations.npz')
        w.status('training_verified_collision_corrections')
        settings=[('lateral',[]),('joint',['--joint'])];fits=[]
        for name,flags in settings:
            if (OUT/name/'status.json').exists() and w.read(OUT/name/'status.json')['status']=='completed':continue
            fits.append(w.launch(['train_trust_region_correction.py','--init-run',SOURCE,'--output',OUT/name,
                '--data',*data,'--rehearsal',ROOT/'results/evidence_guided_refinement_v115/rehearsal/rehearsal.npz',
                '--iterations','1600',*flags],OUT/(name+'.log')))
        for p in fits:w.wait([p],4*3600)
        for name,flags in settings:invariance(OUT/name,bool(flags))
        w.status('fresh_300_seed_comparison')
        baseline=evaluate(SOURCE,'baseline_development',300,DEV);candidates=[]
        for name,_ in settings:
            candidates.append(evaluate(OUT/name,name+'_development',300,DEV))
            w.status('fresh_300_seed_comparison',baseline_development=baseline,candidates=candidates)
        best=max([baseline]+[c for c in candidates if nonregression(c,baseline)],key=quality)
        w.status('final_100_seed_comparison',selected=best)
        paired=evaluate(SOURCE,'baseline_final',100,PRIVATE)
        final=paired if best is baseline else evaluate(Path(best['run']),'selected_final',100,PRIVATE)
        m=final['metrics'];improved=nonregression(final,paired) and m['collision_rate']<paired['metrics']['collision_rate']
        qualified=nonregression(final,paired) and m['collision_rate']<.05 and m['arrival_rate']>=.95 and stalls(m)==0 and m['red_light_rate']==0
        preserved();w.status('completed_target_verified' if qualified else 'completed_experiments_goal_not_met',
            qualified=qualified,objective_achieved=qualified,improved_on_paired_final=improved,
            paired_baseline=paired,final_evaluation=final,protected_files_unchanged=True)
        if qualified:w.save(ROOT/'results/recommended_run.json',dict(qualified=True,recommended_run=final['run'],
            actual_evaluation=m,report=str(OUT/'status.json')))
    finally:execution_state(False)


if __name__=='__main__':
    import sys
    resume='--resume' in sys.argv
    if OUT.exists() and not resume:raise SystemExit('Existing run preserved; duplicate launch refused')
    try:main(resume)
    except BaseException as error:
        OUT.mkdir(parents=True,exist_ok=True);w.OUT=OUT;w.status('failed',error=repr(error),traceback=traceback.format_exc())
        for p in w.CHILDREN:
            if p.poll() is None:p.terminate()
        raise
