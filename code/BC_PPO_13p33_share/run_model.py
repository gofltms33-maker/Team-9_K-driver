"""Run the historical BC+PPO model using its original frozen environment."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from algorithms.ppo import PPO
from train_warmstart_ppo import saved_environment

ROOT=Path(__file__).resolve().parent

def main(a):
    cfg=json.loads((ROOT/'model/run_config.json').read_text(encoding='utf-8'))
    for p,digest in cfg['protected_hashes'].items():
        if hashlib.sha256((ROOT/p).read_bytes()).hexdigest()!=digest:
            raise RuntimeError('Frozen environment changed: '+p)
    cfg['base']=str((ROOT/cfg['base']).resolve())
    torch.set_num_threads(1)
    agent=PPO(104,2,**{**cfg['hparams'],'device':'cpu'})
    agent.load(str(ROOT/'model/model.pt'));agent.policy.eval()
    env=saved_environment(cfg,gui=not a.nogui);rows=[]
    try:
        for seed in range(a.seed,a.seed+a.episodes):
            np.random.seed(seed);obs,reset=env.reset(seed=seed)
            for step in range(1,env.max_steps+1):
                obs,_,done,truncated,info=env.step(agent.predict(obs,deterministic=True))
                if info.get('control_source')!='policy':raise RuntimeError('Non-policy controller')
                if done or truncated:break
            row=dict(seed=seed,end_reason=info['end_reason'],steps=step,initial_traffic=reset['initial_traffic'])
            rows.append(row);print(json.dumps(row),flush=True)
    finally:env.close()
    if a.output:Path(a.output).write_text(json.dumps(rows,indent=2),encoding='utf-8')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,default=1024000)
    p.add_argument('--episodes',type=int,default=1);p.add_argument('--nogui',action='store_true')
    p.add_argument('--output');main(p.parse_args())
