"""Own measured lane state, never a desired-lane instruction or action rule."""
import copy
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import torch


def encode_own_layout(road,index,count):
    if count<1 or not 0<=index<count:raise ValueError('Invalid measured physical lane state')
    return [float(str(road).startswith(':')),min(count/8.,1.),index/max(count-1,1)]


def expand_weights(state):
    result={k:v.clone() for k,v in state.items()}
    for name in ['trunk.0.weight','accel_trunk.0.weight',
                 'lane_adapter.context.0.weight','accel_adapter.context.0.weight']:
        weight=result[name]
        result[name]=torch.cat((weight[:,:64],weight.new_zeros(weight.shape[0],3),weight[:,64:]),dim=1)
    return result


def augment_dataset(source,output):
    source=Path(source);output=Path(output)
    config=json.loads((source.parent/'run_config.json').read_text())
    if config['observation'].get('physical_layout_features'):raise ValueError('Already augmented')
    if config['observation'].get('sensor_history_frames',1)!=1:raise ValueError('Use current-frame audited data')
    audit=json.loads((source.parent/'data_audit.json').read_text())
    if not audit['alignment']['passed']:raise ValueError('Only independently aligned demonstrations')
    with np.load(source,allow_pickle=False) as archive:arrays={k:archive[k].copy() for k in archive.files}
    obs=arrays['observation']
    if obs.shape[1]!=104:raise ValueError('Expected104current features')
    net=ET.parse(Path(config['base'])/'sumo/highway.net.xml').getroot()
    counts={e.get('id'):len(e.findall('lane')) for e in net.findall('edge')}
    own=np.asarray([encode_own_layout(road,int(index),counts[str(road)]) for road,index in
        zip(arrays['audit_before_road'],arrays['audit_before_lane_index'])],dtype=np.float32)
    arrays['observation']=np.concatenate((obs[:,:64],own,obs[:,64:]),axis=1)
    config=copy.deepcopy(config);config['observation']['physical_layout_features']=True
    audit=copy.deepcopy(audit);audit['observation']=config['observation']
    audit['physical_layout_transform']=dict(source=str(source),sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        own_measured_fields=['inside_junction','physical_lane_count_divided_by8','current_lane_fraction'],
        added_columns=[64,67],labels_and_episode_split_unchanged=True,
        no_desired_lane_or_route_lane_answer=True)
    output.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(output/'demonstrations.npz',**arrays)
    for filename,value in [('run_config.json',config),('data_audit.json',audit)]:
        (output/filename).write_text(json.dumps(value,indent=2),encoding='utf-8')
    return output/'demonstrations.npz'
