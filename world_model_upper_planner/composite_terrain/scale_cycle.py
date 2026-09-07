"""One bounded 1024-new-layout exploration -> fit -> paired eval/video cycle."""
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    name='composite3d_v9_exploration'
    root=Path('experiments/composite3d')/name
    root.mkdir(parents=True,exist_ok=True)
    journal=root/'collection_stages.jsonl'
    status=Path('docs/composite3d_v9_live.md')
    history=[]
    def publish(stage,state,**extra):
        record=dict(stage=stage,status=state,time=time.time(),**extra)
        with journal.open('a') as f:f.write(json.dumps(record)+'\n')
        history.append(record)
        status.write_text('# V9 新布局探索扩量：自动进度\n\n'
            '固定V7 H1采集，12%当前支撑候选探索；不改下层。两个512布局分片、四难度、'
            '每环境6000tick；计划新增6.144M下层交互。采集到达不当作整路线成功率。\n\n'
            '| 阶段 | 状态 | 墙钟秒 |\n|---|---|---:|\n'+''.join(
                '| '+r['stage']+' | '+r['status']+' | '+str(round(r.get('wall_seconds',0),1))+' |\n'
                for r in history)+'\n原始状态：`'+str(journal)+'`。训练和评估详见同目录 `stages.jsonl`；'
            '全部完成后生成 [结果与视频](composite3d_v9_exploration_results.md)。\n')
    def run(stage,module,args,completion):
        if completion.exists():publish(stage,'already_complete');return
        publish(stage,'running');start=time.perf_counter()
        with (root/(stage+'.log')).open('a') as f:
            result=subprocess.run([sys.executable,'-m',module,*map(str,args)],stdout=f,stderr=subprocess.STDOUT)
        publish(stage,'complete' if result.returncode==0 else 'failed',
                wall_seconds=time.perf_counter()-start,returncode=result.returncode)
        if result.returncode:raise RuntimeError(stage+' failed; see '+str(root/(stage+'.log')))
    lower='runs/lower3d_v2_z04/model_7200.pt'
    world='runs/composite3d_v7_mixed/h1_best.pt'
    motion='runs/composite3d_v7_sequence/model_best.pt'
    inputs=[Path('experiments/composite3d/replay_v8_mixed_768/transitions.npz')]
    for seed in [9420,9421]:
        out=root/('replay_512_seed'+str(seed))
        run('collect_'+str(seed),'composite_terrain.run',[
            '--lower_checkpoint',lower,'--checkpoint',world,'--motion_checkpoint',motion,
            '--output',out,'--num_envs',512,'--steps',6000,'--seed',seed,
            '--difficulty_levels','.1,.3,.5,.7','--horizon',1,
            '--root_geometry_guard','--behavior','model','--collect','--record_motion',
            '--model_exploration_prob',.12,'--geodesic_guidance','--reset_region_prob',.35,
            '--headless'],out/'metrics.json')
        run('audit_'+str(seed),'composite_terrain.audit_replay',[out/'transitions.npz'],out/'replay_audit.json')
        inputs.append(out/'transitions.npz')
    dataset=root/'merged'/'arrays'
    run('merge','composite_terrain.merge_replays',['--inputs',*inputs,'--output',dataset,'--memmap'],
        dataset.parent/'metrics.json')
    run('train_evaluate_video','composite_terrain.experiment',[
        '--dataset',dataset,'--name',name,'--lower_checkpoint',lower,
        '--init','runs/composite3d_v8_refined/h1_best.pt','--updates',6000,
        '--batch_size',1024,'--anchored','--sequence_motion','--horizons',1,
        '--eval_seed',9801,'--extra_eval_seed',9901,'--eval_difficulty',.5],
        Path('docs')/(name+'_results.md'))
    publish('cycle','complete')


if __name__=='__main__':
    main()
