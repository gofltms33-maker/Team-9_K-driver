"""Collect demonstrations with independently captured pre/post-step telemetry.

Teacher controls apply only during collection. Telemetry is offline audit data,
never additional policy observations or a runtime controller.
"""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import shutil

import numpy as np
import torch

import train_bc_ppo as collector
from env.sumo_env import traci


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def write(path,value):Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')


def validate_alignment(data, telemetry, reports, dt, accel, decel):
    captured=[]
    for report in reports:
        if not report.get('retained'):continue
        candidates=telemetry.get(int(report['seed']),[])
        count=report['samples']
        if len(candidates)<count:raise ValueError('Insufficient independently captured transitions')
        captured.extend(candidates[:count])
    if len(captured)!=len(data['episode']):raise ValueError('Telemetry/dataset row count mismatch')
    errors=[]
    for i,row in enumerate(captured):
        if int(data['episode'][i])!=row['seed']:errors.append([i,'episode'])
        if not np.allclose(data['observation'][i],row['observation'],atol=1e-6,rtol=0):errors.append([i,'observation_timestamp'])
        acceleration=(row['after_speed']-row['before_speed'])/dt
        target=float(np.clip(acceleration/(accel if acceleration>=0 else decel),-1,1))
        if abs(float(data['action'][i,0])-target)>1e-5:errors.append([i,'acceleration_timestamp'])
        if row['before_road']==row['after_road'] and data['lane_valid'][i]:
            delta=row['after_lane_index']-row['before_lane_index']
            if abs(delta)>1 or float(data['action'][i,1])!=float(np.sign(delta)):errors.append([i,'same_road_lane_label'])
        if row['speed_mode']!=31 or row['lane_mode']!=1621:errors.append([i,'teacher_mode'])
    return captured,{'rows':len(captured),'mismatch_count':len(errors),'mismatches':errors[:50],
        'passed':not errors,'checked':['pre-action observation timestamp','executed acceleration target',
        'same-road executed lateral labels','teacher modes'],
        'limitations':['Across-junction label projection additionally uses existing physical-connection traversal.',
                       'Audit consistency does not establish sufficient observation information.']}


def main(args):
    source=args.source.resolve();out=args.output.resolve();config=read(source/'run_config.json')
    seeds=list(range(args.seed_start,args.seed_start+args.episodes))+list(args.include_seeds)
    if len(seeds)!=len(set(seeds)):
        raise ValueError('Collection seeds must be distinct')
    changes={key:value for key,value in [('nav_horizon',args.nav_horizon),
              ('signal_horizon',args.signal_horizon)] if value is not None}
    if changes and args.kind!='native':
        raise ValueError('Range experiments require native collection; learner-prefix inputs must not silently change')
    for key,value in changes.items():
        if not math.isfinite(value) or value<=0:raise ValueError('Sensor horizons must be finite and positive')
        config['observation'][key]=value
    from parallel_evaluate import validate_course_config
    validate_course_config(config)
    for path,digest in config['protected_hashes'].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest,path
    for name,key in [('HPARAMS','hparams'),('OBSERVATION','observation'),('ACTION','action'),
                     ('SIMULATION','simulation'),('REWARD','reward')]:
        setattr(collector,name,copy.deepcopy(config[key]))
    torch.set_num_threads(1)
    out.mkdir(parents=True,exist_ok=False)
    saved=copy.deepcopy(config)
    saved.update(stage='Audited demonstration collection, not autonomous-model evaluation',
        args={'route_variant':'normal','route_prefix':'None'},
        collection={'kind':args.kind,'source':str(source),'seed_start':args.seed_start,
            'episodes':len(seeds),'collection_seeds':seeds,
            'teacher_modes':[31,1621],'learner_prefix_modes':[96,0],
            'lead_steps':args.lead_steps if args.kind=='recovery' else None,
            'window_steps':args.window_steps if args.kind=='recovery' else None,
            'minimum_successful_episodes':args.min_successful_episodes,
            'no_policy_input_added':True,'no_road_or_hv_edits':True})
    saved['collection']['sensor_range_experiment']=changes
    write(out/'run_config.json',saved)
    for path in ['collect_audited_course_data.py','train_bc_ppo.py','action_labels.py','physical_layout.py','env/sumo_env.py']:
        target=out/'source'/path;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,target)
    telemetry={}
    original_make=collector.make_env
    def make(*pos,**kw):
        env=original_make(*pos,**kw)
        reset=env.reset;step=env.step
        def audited_reset(*a,**k):
            seed=k.get('seed');env._audit_seed=int(seed)
            telemetry[int(seed)]=[]
            return reset(*a,**k)
        def audited_step(action):
            teacher=env.native_driver and 'ego' in traci.vehicle.getIDList()
            if teacher:
                before=dict(seed=env._audit_seed,observation=env._last_obs.copy(),
                    before_speed=traci.vehicle.getSpeed('ego'),before_road=traci.vehicle.getRoadID('ego'),
                    before_lane_index=traci.vehicle.getLaneIndex('ego'),sim_time=traci.simulation.getTime(),
                    speed_mode=traci.vehicle.getSpeedMode('ego'),lane_mode=traci.vehicle.getLaneChangeMode('ego'))
            result=step(action)
            if teacher and not result[2] and not result[3] and 'ego' in traci.vehicle.getIDList():
                before.update(after_speed=traci.vehicle.getSpeed('ego'),after_road=traci.vehicle.getRoadID('ego'),
                    after_lane_index=traci.vehicle.getLaneIndex('ego'))
                telemetry[env._audit_seed].append(before)
            return result
        env.reset=audited_reset;env.step=audited_step
        return env
    collector.make_env=make
    data=collector.collect(Path(config['base']),out,args.episodes,
        recovery_policy=source/'model.pt' if args.kind=='recovery' else None,
        failure_targeted=args.kind=='recovery',seed_start=args.seed_start,
        route_variant='normal',recovery_lead_steps=args.lead_steps if args.kind=='recovery' else None,
        recovery_window=args.window_steps if args.kind=='recovery' else None,
        failure_only=args.kind=='recovery',expert_tau=1.8,collection_seeds=seeds,
        minimum_successful_episodes=args.min_successful_episodes)
    audit=read(out/'data_audit.json')
    env=original_make(Path(config['base']))
    captured,alignment=validate_alignment(data,telemetry,audit['episodes'],env.step_length,env.max_accel,env.max_decel)
    env.close()
    audit.update(observation=config['observation'],alignment=alignment,
        collection=saved['collection'],successful_episodes=sum(r.get('retained',False) for r in audit['episodes']))
    write(out/'data_audit.json',audit)
    for key in ['before_speed','after_speed','before_lane_index','after_lane_index','sim_time',
                'before_road','after_road']:
        data['audit_'+key]=np.asarray([r[key] for r in captured])
    np.savez_compressed(out/'demonstrations.npz',**data)
    write(out/'status.json',{'status':'completed' if alignment['passed'] else 'failed',
        'alignment':alignment,'samples':len(data['episode'])})
    if not alignment['passed']:raise RuntimeError('Demonstration alignment audit failed')
    print('AUDITED_DATA',str(out),alignment,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('output',type=Path)
    p.add_argument('--kind',choices=['native','recovery'],required=True)
    p.add_argument('--episodes',type=int,required=True);p.add_argument('--seed-start',type=int,required=True)
    p.add_argument('--include-seeds',type=int,nargs='*',default=[],
        help='Additional known failure examples used for training, never claimed as unseen tests')
    p.add_argument('--lead-steps',type=int,default=120)
    p.add_argument('--window-steps',type=int,default=480)
    p.add_argument('--min-successful-episodes',type=int,default=5,
                   help='Minimum retained teacher episodes; small targeted additions still require alignment/action checks')
    p.add_argument('--nav-horizon',type=float);p.add_argument('--signal-horizon',type=float)
    args=p.parse_args()
    if args.lead_steps<1 or args.window_steps<1:p.error('Recovery lead/window must be positive')
    if args.min_successful_episodes<1:p.error('At least one successful teacher episode is required')
    main(args)
