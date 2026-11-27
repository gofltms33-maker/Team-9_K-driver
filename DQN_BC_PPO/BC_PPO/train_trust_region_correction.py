"""Fit genuine corrective lane demonstrations while distilling successful B rollouts."""
import argparse
import copy
import json
from pathlib import Path
import shutil
import numpy as np
import torch
import torch.nn.functional as F
from algorithms.ppo import PPO
from train_warmstart_ppo import load_demonstrations,freeze_for_lateral_ppo,censored_acceleration_error
import run_validated_imitation as w


def correction_data(paths,observation,heldout,full=False):
    parts=[]
    for path in paths:
        data=load_demonstrations([path],104,heldout=heldout,expected_observation=observation,lane_weight_power=0)
        with np.load(path) as archive:
            seeds=archive['episode'];seeds=seeds[seeds%5==4 if heldout else seeds%5!=4]
        if not len(seeds):continue
        limit=120 if path.parent.name=='route_recovery' or path.parent.name.startswith('blocked_') else 32
        indices=np.concatenate([np.flatnonzero(seeds==seed)[:limit] for seed in np.unique(seeds)])
        indices=indices[data['lane_valid'][indices].numpy()]
        if not len(indices):continue
        parts.append((data['observation'][indices],(1-data['action'][indices,1]).long(),
                      data['action'][indices,:1],data['accel_valid'][indices]))
    if not parts:raise ValueError('No valid corrective rows in requested episode split')
    result=tuple(torch.cat([v[i] for v in parts]) for i in range(4))
    return result if full else result[:2]


def corrective_loss(policy,x,y,target,valid,joint=False):
    mu,_,logits,_=policy(x)
    loss=F.cross_entropy(logits,y)
    if joint and valid.any():
        error=censored_acceleration_error(mu,target,x[:,:1])
        weight=torch.where((target<-.05)&(mu>target),4.,1.)
        loss=loss+(error.square()*weight)[valid].mean()
    return loss


def drift(policy,reference,obs):
    with torch.no_grad():
        logits=policy(obs)[2];old=reference(obs)[2]
        kl=F.kl_div(logits.log_softmax(-1),old.softmax(-1),reduction='batchmean').clamp_min(0)
        changed=(logits.argmax(-1)!=old.argmax(-1)).float().mean()
    return float(kl),float(changed)


def acceptable_drift(kl,changed):return np.isfinite(kl) and kl<=.003 and changed<=.01


def main():
    p=argparse.ArgumentParser();p.add_argument('--init-run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--data',type=Path,nargs='+',required=True)
    p.add_argument('--rehearsal',type=Path,required=True);p.add_argument('--iterations',type=int,default=1200)
    p.add_argument('--joint',action='store_true',help='Also correct the learned acceleration adapter; keep both backbones frozen')
    a=p.parse_args();torch.set_num_threads(1);torch.manual_seed(116);rng=np.random.default_rng(116)
    a.output.mkdir(parents=True,exist_ok=False)
    cfg=w.read(a.init_run/'run_config.json');cfg.pop('frozen_checkpoint',None)
    from parallel_evaluate import validate_course_config
    validate_course_config(cfg)
    agent=PPO(104,2,**{**cfg['hparams'],'device':'cpu'});agent.load(str(a.init_run/'model.pt'))
    policy=agent.policy;freeze_for_lateral_ppo(policy);reference=copy.deepcopy(policy).requires_grad_(False).eval()
    if a.joint:policy.accel_adapter.requires_grad_(True)
    cfg['stage']='Audited corrective BC plus successful-neural-rollout distillation; acceleration entirely frozen'
    cfg['initialization']=dict(run=str(a.init_run),model_sha256=w.sha(a.init_run/'model.pt'))
    cfg['correction_learning']=dict(lr=3e-6,iterations=a.iterations,anchor_kl_coefficient=10,
        max_heldout_anchor_kl=.003,max_heldout_anchor_action_change_fraction=.01,
        corrective_prefix_steps=32,route_corrective_prefix_steps=120,
        no_global_class_reweighting=True,no_runtime_rules=True,
        correction_sources=[dict(path=str(x),sha256=w.sha(x)) for x in a.data],
        rehearsal=str(a.rehearsal),rehearsal_sha256=w.sha(a.rehearsal))
    cfg['correction_learning'].update(joint_adapters=a.joint,max_heldout_acceleration_mae=.01)
    if a.joint:cfg['stage']='Audited corrective BC of both adapters with measured longitudinal and lateral distillation bounds'
    cfg['data_sources']+= [dict(path=str(x),sha256=w.sha(x)) for x in a.data]
    w.save(a.output/'run_config.json',cfg)
    for name in ['train_trust_region_correction.py','train_warmstart_ppo.py']:
        shutil.copyfile(name,a.output/name)
    train_x,train_y,train_acc,train_valid=correction_data(a.data,cfg['observation'],False,True)
    val_x,val_y,val_acc,val_valid=correction_data(a.data,cfg['observation'],True,True)
    with np.load(a.rehearsal) as archive:
        obs=archive['observation'];episode=archive['episode']
        anchor=torch.from_numpy(obs[episode%5!=4]);anchor_val=torch.from_numpy(obs[episode%5==4])
    if not all(len(v) for v in [train_x,val_x,anchor,anchor_val]):raise RuntimeError('Train/validation buffer split is empty')
    optimizer=torch.optim.Adam([v for v in policy.parameters() if v.requires_grad],lr=3e-6)
    with torch.no_grad():best=float(corrective_loss(policy,val_x,val_y,val_acc,val_valid,a.joint))
    agent.save(str(a.output/'model.pt'));accepted=copy.deepcopy(policy.state_dict());selected=0;history=[]
    for iteration in range(1,a.iterations+1):
        idx=rng.integers(0,len(train_x),128);ai=rng.integers(0,len(anchor),256)
        correction=corrective_loss(policy,train_x[idx],train_y[idx],train_acc[idx],train_valid[idx],a.joint)
        mu,_,logits,_=policy(anchor[ai])
        with torch.no_grad():old_mu,_,old,_=reference(anchor[ai])
        anchor_loss=F.kl_div(logits.log_softmax(-1),old.softmax(-1),reduction='batchmean')
        if a.joint:anchor_loss=anchor_loss+(mu-old_mu).square().mean()
        optimizer.zero_grad();(correction+10*anchor_loss).backward()
        torch.nn.utils.clip_grad_norm_([v for v in policy.parameters() if v.requires_grad],.25);optimizer.step()
        if iteration%100==0 or iteration==a.iterations:
            kl,changed=drift(policy,reference,anchor_val)
            with torch.no_grad():
                validation=float(corrective_loss(policy,val_x,val_y,val_acc,val_valid,a.joint))
                accel_drift=float((policy(anchor_val)[0]-reference(anchor_val)[0]).abs().mean())
            safe=acceptable_drift(kl,changed) and (not a.joint or accel_drift<=.01)
            if not safe:
                policy.load_state_dict(accepted);optimizer.state.clear()
                for group in optimizer.param_groups:group['lr']*=.5
            else:
                accepted=copy.deepcopy(policy.state_dict())
                if validation<best:best=validation;selected=iteration;agent.save(str(a.output/'model.pt'))
            row=dict(iteration=iteration,corrective_validation_ce=validation,anchor_kl=kl,
                anchor_action_change_fraction=changed,anchor_acceleration_mae=accel_drift,accepted=safe,selected_iteration=selected)
            history.append(row);w.save(a.output/'history.json',history);w.save(a.output/'status.json',dict(status='training',**row))
            print('CORRECTION',row,flush=True)
    agent.save(str(a.output/'model_last.pt'))
    w.save(a.output/'status.json',dict(status='completed',iterations=a.iterations,selected_iteration=selected,
        selected_validation_ce=best,training_correction_rows=len(train_x),heldout_correction_rows=len(val_x),
        rehearsal_training_rows=len(anchor),rehearsal_validation_rows=len(anchor_val)))


if __name__=='__main__':main()
