"""Past measured sensors as policy inputs; no action history or driving helper."""
from collections import deque
import numpy as np


def pack_history(history):
    current=history[-1]
    if len(current)!=104:raise ValueError('Sensor history expects the audited104-feature layout')
    # Preserve existing current-state indices0:64 and current traffic at -40:.
    return np.concatenate([current[:64],*list(history)[:-1][::-1],current[64:]]).astype(np.float32)


class SensorHistoryEnv:
    def __init__(self,env,frames):
        if int(frames)!=frames or not 2<=frames<=4 or env.obs_dim!=104:
            raise ValueError('Only2..4frames of audited104-feature observations are supported')
        object.__setattr__(self,'_base',env)
        object.__setattr__(self,'_frames',int(frames))
        object.__setattr__(self,'_history',deque(maxlen=int(frames)))
        object.__setattr__(self,'obs_dim',104*int(frames))

    def __getattr__(self,name):return getattr(self._base,name)

    def __setattr__(self,name,value):
        if name.startswith('_') or name=='obs_dim':object.__setattr__(self,name,value)
        else:setattr(self._base,name,value)

    def reset(self,*args,**kwargs):
        obs,info=self._base.reset(*args,**kwargs)
        self._history.clear()
        self._history.extend([obs.copy() for _ in range(self._frames)])
        return pack_history(self._history),info

    def step(self,action):
        obs,reward,terminated,truncated,info=self._base.step(action)
        self._history.append(obs.copy())
        return pack_history(self._history),reward,terminated,truncated,info


def with_sensor_history(env,observation_config):
    frames=observation_config.get('sensor_history_frames',1)
    if frames==1:return env
    return SensorHistoryEnv(env,frames)


def validate_history_data(array,config):
    frames=int(config.get('sensor_history_frames',1))
    if frames==1:return
    if array.ndim!=2 or array.shape[1]!=104*frames or not np.isfinite(array).all():
        raise ValueError('Invalid sensor history shape or values')
    if config.get('redact_lane_guidance',False):
        for i in range(frames-1):
            past=array[:,64+104*i:64+104*(i+1)]
            if np.any(past[:,38:53]!=0) or np.any(past[:,54]!=0):
                raise ValueError('Past sensor frames contain detailed lane guidance')
            if not config.get('route_distance_feature',False) and np.any(past[:,53]!=0):
                raise ValueError('Past frame reserved slot53 is not enabled')


def augment_dataset(source,output,frames=4):
    import hashlib,json
    from pathlib import Path
    source=Path(source);output=Path(output)
    with np.load(source) as a:data={key:a[key].copy() for key in a.files}
    cfg=json.loads((source.parent/'run_config.json').read_text(encoding='utf-8'))
    audit=json.loads((source.parent/'data_audit.json').read_text(encoding='utf-8'))
    if cfg['observation'].get('sensor_history_frames',1)!=1 or data['observation'].shape[1]!=104:
        raise ValueError('Augment only original audited104-feature data')
    if audit['label_schema']!=3 or not audit.get('alignment',{}).get('passed'):
        raise ValueError('History requires aligned schema3 transitions')
    original=data['observation'];episodes=data['episode'];time=data['audit_sim_time']
    combined=np.empty((len(original),104*frames),dtype=np.float32)
    history=deque(maxlen=frames);offset=0
    for i,row in enumerate(original):
        new=i==0 or episodes[i]!=episodes[i-1]
        if new:
            history.clear();history.extend([row.copy() for _ in range(frames)]);offset=0
        else:
            if abs(float(time[i]-time[i-1])-cfg['simulation']['step_length'])>1e-5:
                raise ValueError('A temporal gap was found in demonstration samples')
            history.append(row);offset+=1
        combined[i]=pack_history(history)
        if audit.get('expert_suffix',False) and offset<frames-1:
            # Learner-prefix history was not stored; do not teach fabricated
            # recovery-start contexts. Subsequent teacher history is measured.
            data['accel_valid'][i]=False;data['lane_valid'][i]=False
    data['observation']=combined
    cfg['observation']['sensor_history_frames']=frames
    audit['observation']=cfg['observation'].copy()
    audit['sensor_history_transform']={'source':str(source),
        'sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'frames':frames,
        'layout':'current0:64,past raw frames newest first,current traffic40',
        'past_expert_actions_used_as_inputs':False,'recovery_first_frames_masked':frames-1}
    validate_history_data(combined,cfg['observation'])
    output.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(output/'demonstrations.npz',**data)
    for name,value in [('run_config.json',cfg),('data_audit.json',audit)]:
        (output/name).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
