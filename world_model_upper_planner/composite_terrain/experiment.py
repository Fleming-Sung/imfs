"""Finite, resumable train -> H1/H3 physical tests -> videos -> report cycle."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--name',required=True);p.add_argument('--lower_checkpoint',required=True)
    p.add_argument('--updates',type=int,default=2500);p.add_argument('--batch_size',type=int,default=512)
    p.add_argument('--init');p.add_argument('--anchored',action='store_true')
    p.add_argument('--sequence_motion',action='store_true')
    p.add_argument('--local_refinement',action='store_true')
    p.add_argument('--extra_eval_seed',type=int)
    p.add_argument('--horizons',type=int,nargs='+',choices=[1,3],
                   help='explicit evaluation horizons; use 1 for the established short-model branch')
    p.add_argument('--eval_seed',type=int,default=9601);p.add_argument('--eval_difficulty',type=float,default=.25)
    a=p.parse_args()
    if (a.sequence_motion or a.local_refinement) and not a.anchored:
        raise ValueError('physical sequence/refinement experiments require --anchored')
    root=Path('experiments/composite3d')/a.name;root.mkdir(parents=True,exist_ok=True)
    model=Path('runs')/a.name
    log=root/'stages.jsonl'
    def event(stage,status,**kw):
        with log.open('a') as f:f.write(json.dumps(dict(stage=stage,status=status,time=time.time(),**kw))+'\n')
        latest={}
        for line in log.read_text().splitlines():
            record=json.loads(line);latest[record['stage']]=record
        Path(f'docs/{a.name}_training_live.md').write_text(
            f'# {a.name} 训练与验证自动进度\n\n'
            '| 阶段 | 状态 | 墙钟秒 |\n|---|---|---:|\n'+''.join(
                f"| {name} | {r['status']} | {r.get('wall_seconds','')} |\n" for name,r in latest.items())+
            f'\n原始日志：`{root}`。只有完整结束的评估才计入性能结果。\n')
    def execute(stage,module,args,completion):
        if completion.exists():event(stage,'already_complete');return
        event(stage,'running');start=time.perf_counter()
        with (root/f'{stage}.log').open('a') as stream:
            result=subprocess.run([sys.executable,'-m',module,*map(str,args)],stdout=stream,stderr=subprocess.STDOUT)
        event(stage,'complete' if result.returncode==0 else 'failed',returncode=result.returncode,
              wall_seconds=time.perf_counter()-start)
        if result.returncode:raise RuntimeError(f'{stage} failed; see {root/stage}.log')
    event('dataset','waiting')
    deadline=time.monotonic()+1800
    while not (a.dataset.exists() and (a.dataset.parent/'metrics.json').exists()):
        if time.monotonic()>deadline:raise TimeoutError('collector did not finish within 30 minutes')
        time.sleep(5)
    execute('audit','composite_terrain.audit_replay',[a.dataset],a.dataset.parent/'replay_audit.json')
    execute('train','composite_terrain.train',['--dataset',a.dataset,'--output',model,
        '--h1_updates',a.updates,'--h3_updates',0 if a.anchored else a.updates,'--batch_size',a.batch_size]
        +(['--init',a.init] if a.init else []),model/'summary.json')
    world=model/('h1_best.pt' if a.anchored else 'h3_best.pt')
    base=['--lower_checkpoint',a.lower_checkpoint,'--checkpoint',world,
          '--behavior','model','--root_geometry_guard','--difficulty',a.eval_difficulty]
    if a.anchored:
        execute('motion','composite_terrain.train_motion',['--dataset',a.dataset,
            '--world_checkpoint',world,'--output',model/'motion','--updates',3000],model/'motion/summary.json')
        motion=model/'motion/model_best.pt'
        if a.sequence_motion:
            execute('motion_sequence','composite_terrain.train_motion_sequence',[
                '--dataset',a.dataset,'--init',motion,'--output',model/'motion_sequence',
                '--updates',1000],model/'motion_sequence/summary.json')
            motion=model/'motion_sequence/model_best.pt'
        base+=['--motion_checkpoint',motion]
    if a.local_refinement:base+=['--local_refinement']
    horizons=a.horizons or ([1] if a.local_refinement else [1,3])
    if a.local_refinement and horizons!=[1]:
        raise ValueError('local refinement is H1 only')
    eval_seeds=[a.eval_seed]+([a.extra_eval_seed] if a.extra_eval_seed is not None else [])
    for eval_seed in eval_seeds:
        for horizon in horizons:
            out=root/f'h{horizon}_64_seed{eval_seed}'
            stage=f'eval_h{horizon}' if eval_seed==a.eval_seed else f'eval_h{horizon}_seed{eval_seed}'
            execute(stage,'composite_terrain.run',base+[
                '--output',out,'--num_envs',64,'--steps',3000,'--seed',eval_seed,
                '--horizon',horizon,'--headless'],out/'metrics.json')
    for seed in [a.eval_seed+1,a.eval_seed+2]:
        out=root/f'video_seed{seed}'
        execute(f'video_{seed}','composite_terrain.run',base+[
            '--output',out,'--num_envs',1,'--steps',3000,'--seed',seed,
            '--horizon',max(horizons),'--record_video'],out/'rollout.mp4')
        execute(f'plot_{seed}','composite_terrain.plot_rollout',[out],out/'trajectory.png')
    results={v.parent.name:json.loads(v.read_text()) for v in root.glob('*/metrics.json')}
    (root/'summary.json').write_text(json.dumps(results,indent=2))
    lines=[f'# {a.name} 自动实验结果','',
           '所有正式模型测试均从路线起点出发，禁用全图训练引导；保留当前支撑约束。',
           ('本轮仅运行H1，不构成H1/H3对照。' if horizons==[1] else
            'H1与H3使用相同检查点/信息，区别只是规划时域。')+'各测试尾部未结束episode作为截尾记录。','',
           '| 测试 | 完成数 | 跌倒数 | 截尾数 | 最远x均值(m) |',
           '|---|---:|---:|---:|---:|']
    for name,m in results.items():
        lines.append(f"| {name} | {m['successes']} | {m['falls']} | {m['right_censored_episodes']} | {m['mean_farthest_x_m']:.2f} |")
    lines+=['','视频不是成功证明，需结合完整测试表：','']
    for seed in [a.eval_seed+1,a.eval_seed+2]:
        prefix=f'../experiments/composite3d/{a.name}/video_seed{seed}'
        lines.append(f'- seed{seed}：[视频]({prefix}/rollout.mp4)、[轨迹]({prefix}/trajectory.png)')
    lines+=['',f'原始数据、日志和逐阶段状态：`experiments/composite3d/{a.name}/`。']
    Path(f'docs/{a.name}_results.md').write_text('\n'.join(lines)+'\n')
    event('cycle','complete')

if __name__=='__main__':main()
