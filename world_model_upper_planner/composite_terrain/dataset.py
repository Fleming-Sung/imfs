"""Linked replay with in-data TD bootstrap, avoiding offline max-Q actions."""
import numpy as np
import torch
from scripts.train_h3 import SequenceDataset as BaseSequenceDataset


class SequenceDataset(BaseSequenceDataset):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        d=self.data
        lookup={tuple(map(int,k)):i for i,k in enumerate(zip(
            d['env_id'],d['episode_id'],d['option_index']))}
        self.next_row=np.full(len(d['env_id']),-1,dtype=np.int64)
        for (e,p,o),i in lookup.items():
            if not d['done'][i]:self.next_row[i]=lookup.get((e,p,o+1),-1)

    def batch(self,indices,depth_mode='normal'):
        batch=super().batch(indices,depth_mode)
        rows=self.next_row[self.sequences[indices]]
        batch['next_action']=torch.as_tensor(self.data['action'][rows.clip(0)],
            device=self.device,dtype=torch.float32)
        batch['bootstrap_valid']=torch.as_tensor(rows>=0,device=self.device,dtype=torch.float32)
        return batch
