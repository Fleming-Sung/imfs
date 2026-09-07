"""H1 warm-up then H3 learning, bound to the exact static-map/lower replay."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from cgowm import CandidateGroundedWorldModel,ModelConfig
from .trainer import WorldModelTrainer,TrainerConfig
from scripts.train_h3 import validate
from .dataset import SequenceDataset


def main():
    p=argparse.ArgumentParser(); p.add_argument("--dataset",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--init",type=Path)
    p.add_argument("--h1_updates",type=int,default=750);p.add_argument("--h3_updates",type=int,default=750)
    p.add_argument("--batch_size",type=int,default=256);p.add_argument("--seed",type=int,default=9301)
    p.add_argument("--log_every",type=int,default=50,help='TensorBoard scalar cadence in updates')
    p.add_argument("--hidden_dim",type=int,default=256)
    p.add_argument("--geometry_dim",type=int,default=64)
    p.add_argument("--dynamics_dim",type=int,default=64)
    args=p.parse_args(); args.output.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(args.seed);np.random.seed(args.seed);torch.set_num_threads(4)
    writer=SummaryWriter(args.output/"tb")
    ds1=SequenceDataset(args.dataset,"cuda:0",1,.2,args.seed)
    ds3=SequenceDataset(args.dataset,"cuda:0",3,.2,args.seed) if args.h3_updates else None
    meta=json.loads((args.dataset.parent/"metrics.json").read_text())
    cfg=ModelConfig(action_dim=4,proprio_dim=ds1.data["proprio"].shape[-1],
                    hidden_dim=args.hidden_dim,geometry_dim=args.geometry_dim,
                    dynamics_dim=args.dynamics_dim)
    model=CandidateGroundedWorldModel(ds1.data["candidates"],cfg).cuda()
    if args.init:
        initial=torch.load(args.init,map_location='cuda:0',weights_only=False)
        if initial['lower_sha256']!=meta['lower_sha256'] or initial['model_config']!=asdict(cfg):
            raise ValueError('incompatible warm start')
        if not torch.equal(initial['candidates'].to(model.candidates),model.candidates):
            raise ValueError('changed action contract')
        model.load_state_dict(initial['model'])
    trainer=WorldModelTrainer(model,TrainerConfig(learning_rate=1e-4))
    best=float("inf");start=time.perf_counter()
    for stage,dataset,updates in [("h1",ds1,args.h1_updates),("h3",ds3,args.h3_updates)]:
        if not updates:continue
        best=float("inf")
        groups=[v for v in dataset.train_groups.values() if len(v)]
        for it in range(updates):
            pool=groups[it%len(groups)]
            indices=pool[np.random.randint(len(pool),size=args.batch_size)]
            model.train();metrics=trainer.train_step(dataset.batch(indices))
            if (it+1)%args.log_every==0:
                writer.add_scalars(f"train/{stage}",
                    {k:float(v) for k,v in metrics.items()},it+1)
            if (it+1)%250==0 or it==updates-1:
                val=validate(trainer,dataset,min(128,args.batch_size),8)
                writer.add_scalars(f"val/{stage}",
                    {k:float(v) for k,v in val.items()},it+1)
                selection_score=float(np.mean([val[f'selection_h{s}_regret']+
                    .25*(1-val[f'selection_h{s}_valid']) for s in range(1,dataset.horizon+1)]))
                writer.add_scalar(f"selection/{stage}/score",selection_score,it+1)
                record=dict(stage=stage,updates=it+1,elapsed_seconds=time.perf_counter()-start,
                            train=metrics,validation=val,selection_score=selection_score)
                with (args.output/"progress.jsonl").open("a") as stream:stream.write(json.dumps(record)+"\n")
                state=dict(model=model.state_dict(),model_config=asdict(cfg),candidates=model.candidates.cpu(),
                           lower_sha256=meta["lower_sha256"],dataset=str(args.dataset),stage=stage,updates=it+1,
                           target=trainer.target.state_dict(),optimizer=trainer.optimizer.state_dict(),
                           trainer_config=asdict(trainer.config),init=str(args.init) if args.init else None)
                if selection_score<best:
                    best=selection_score;torch.save(state,args.output/f"{stage}_best.pt")
                torch.save(state,args.output/f"{stage}_{it+1}.pt")
                print(json.dumps(dict(stage=stage,update=it+1,val_loss=val["loss_total"],
                    elapsed_seconds=record["elapsed_seconds"])),flush=True)
    writer.close()
    (args.output/"summary.json").write_text(json.dumps(dict(
        transitions=len(ds1.data["env_id"]),linked_h3_sequences=len(ds3.sequences) if ds3 else None,
        lower_sha256=meta["lower_sha256"],h1_updates=args.h1_updates,h3_updates=args.h3_updates,
        wall_seconds=time.perf_counter()-start),indent=2))

if __name__=="__main__":main()
