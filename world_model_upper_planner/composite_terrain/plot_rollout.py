"""Overlay the recorded robot/target trajectory on the actual seeded terrain."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .maps import CompositeSpec,generate_route


def main():
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);a=p.parse_args()
    m=json.loads((a.run/'metrics.json').read_text())
    difficulty=m.get('route_difficulties',[m['difficulty']])[0]
    route=generate_route(CompositeSpec(seed=m['seed']*1000+m.get('route_offset',0),difficulty=difficulty))
    d=np.load(a.run/'trajectory.npz');root=d['root'].copy();root[:,1]-=2.4
    targets=d['targets'].copy();targets[:,:,1]-=2.4
    done=np.flatnonzero(d['done']);end=int(done[0])+1 if len(done) else len(root)
    if (a.run/'completed_episodes.npz').exists():
        episodes=np.load(a.run/'completed_episodes.npz')
        first=np.flatnonzero(episodes['env_id']==0)
        if len(first):end=int(episodes['ticks'][first[0]])
    fig,ax=plt.subplots(2,1,figsize=(15,7),gridspec_kw={'height_ratios':[2,1]})
    ax[0].imshow(route['height'],origin='lower',extent=[0,route['x'][-1],-2.4,2.4],
        cmap='terrain',vmin=-.8,vmax=.3,aspect='auto')
    ax[0].plot(root[:end,0],root[:end,1],'r-',label='base first episode')
    for j in range(2):ax[0].plot(targets[:end:25,j,0],targets[:end:25,j,1],'.',label=f'foot {j} target')
    for s in route['segments']:
        ax[0].axvline(s['start_x'],c='k',alpha=.2)
        ax[0].text(s['start_x']+.05,1.7,s['kind'],fontsize=8)
    ax[0].set(xlabel='route x (m)',ylabel='y (m)',title=f"{a.run.name}: first episode {end*.02:.2f}s")
    ax[0].legend(loc='lower right');ax[1].plot(np.arange(len(root))*.02,root[:,0],label='base x')
    ax[1].plot(np.arange(len(root))*.02,root[:,2],label='base z')
    for tick in done:ax[1].axvline(tick*.02,c='r',alpha=.3)
    ax[1].set(xlabel='nominal time (s)',ylabel='m');ax[1].legend()
    fig.tight_layout();fig.savefig(a.run/'trajectory.png',dpi=130);plt.close(fig)

if __name__=='__main__':main()
