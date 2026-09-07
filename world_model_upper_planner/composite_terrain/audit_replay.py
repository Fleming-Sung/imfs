"""Audit physical replay coverage and exact decision-state sequence linkage."""
import argparse
import json
from pathlib import Path
import numpy as np


def audit(path):
    if path.is_dir():
        manifest=json.loads((path/'manifest.json').read_text())
        d={k:np.load(path/v['file'],mmap_mode='r') for k,v in manifest['arrays'].items()}
    else:
        with np.load(path) as source:
            d={k:source[k] for k in source.files}
    lookup={tuple(map(int,k)):i for i,k in enumerate(zip(
        d['env_id'],d['episode_id'],d['option_index']))}
    pairs=[(i,lookup[(e,p,o+1)]) for (e,p,o),i in lookup.items()
           if (e,p,o+1) in lookup and not d['done'][i]]
    state_error=0.;image_error=0
    # Bound temporary image memory independently of replay size. Converting
    # every paired uint8 image to int64 can consume tens of GB unnecessarily.
    for offset in range(0,len(pairs),4096):
        first,second=np.asarray(pairs[offset:offset+4096]).T
        state_error=max(state_error,float(abs(d['next_proprio'][first]-d['proprio'][second]).max()))
        image_error=max(image_error,int(abs(d['next_depth'][first].astype(np.int16)-d['depth'][second]).max()))
    finite=all(np.isfinite(v).all() for v in d.values() if v.dtype.kind in 'fc')
    kinds,counts=np.unique(d['terrain_kind'],return_counts=True)
    z,zcounts=np.unique(d['action'][:,2],return_counts=True)
    result=dict(transitions=len(d['env_id']),environments=len(np.unique(d['env_id'])),
        linked_pairs=len(pairs),next_state_max_error=state_error,next_image_max_error=image_error,
        all_finite=bool(finite),fall_fraction=float(d['fall'].mean()),
        states_with_valid_candidate=float((d['candidate_valid'].sum(-1)>0).mean()),
        terrain_counts=dict(zip(kinds.tolist(),counts.tolist())),
        normalized_dz_counts=dict(zip(z.tolist(),zcounts.tolist())))
    (path.parent/'replay_audit.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
    if not finite or state_error>1e-6 or image_error:
        raise ValueError('invalid physical sequence linkage')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('dataset',type=Path)
    audit(p.parse_args().dataset)
