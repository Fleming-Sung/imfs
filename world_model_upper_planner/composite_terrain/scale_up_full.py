"""Full scale-up pipeline (dev-2026-09-07, round 2).

Retrain the XYZ lower to +/-6 cm with more iterations, then collect roughly
2x the previous data, train a larger world model from scratch with more
updates, and evaluate against the greedy baseline on hard seeds.

All stages are resumable: a completion marker makes completed stages skipped.
"""
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time


def main():
    os.chdir(Path(__file__).resolve().parents[1])
    name = 'composite3d_hard_big'
    root = Path('experiments/composite3d') / name
    root.mkdir(parents=True, exist_ok=True)
    journal = root / 'stages.jsonl'
    status = Path('docs/composite3d_hard_big_live.md')

    prev_lower = 'runs/lower3d_v2_z04/model_7200.pt'
    lower_dir = Path('runs/lower3d_v3_z06')
    lower_selected = lower_dir / 'selected_checkpoint.txt'
    difficulty_levels = '.3,.5,.7'
    shard_seeds = list(range(9410, 9418))  # 8 shards
    eval_seeds = [9801, 9901]
    eval_difficulty = '.5'
    world_updates = 30000
    motion_updates = 5000
    hidden_dim, geometry_dim, dynamics_dim = 512, 128, 128

    history = []

    def publish(stage, state, **extra):
        record = dict(stage=stage, status=state, time=time.time(), **extra)
        with journal.open('a') as f:
            f.write(json.dumps(record) + '\n')
        history.append(record)
        latest = {}
        for line in journal.read_text().splitlines():
            r = json.loads(line)
            latest[r['stage']] = r
        lines = ['# 规模扩大训练：自动进度（第二轮）', '',
                 '先重训下层到 ±6 cm，再用新下层采集 2 倍数据，训练更大容量的上层世界模型。', '',
                 '| 阶段 | 状态 | 墙钟秒 |', '|---|---|---:|']
        for r in latest.values():
            lines.append(f"| {r['stage']} | {r['status']} | {round(r.get('wall_seconds', 0), 1)} |")
        lines += ['', f'原始日志：`{root}`。TensorBoard：`runs/{name}/tb`、'
                  f'`runs/{name}/motion/tb`、下层 `runs/lower3d_v3_z06/tb`。']
        status.write_text('\n'.join(lines) + '\n')

    def run(stage, module, args, completion, cwd=None):
        if completion.exists():
            publish(stage, 'already_complete')
            return
        publish(stage, 'running')
        start = time.perf_counter()
        with (root / (stage + '.log')).open('a') as stream:
            result = subprocess.run(
                [sys.executable, '-m', module, *map(str, args)],
                stdout=stream, stderr=subprocess.STDOUT, cwd=cwd)
        publish(stage, 'complete' if result.returncode == 0 else 'failed',
                wall_seconds=time.perf_counter() - start,
                returncode=result.returncode)
        if result.returncode:
            raise RuntimeError(f'{stage} failed; see {root / (stage + ".log")}')

    # 1) Retrain the lower controller: 2048 envs, 1000 iterations, +/-6 cm.
    if not lower_selected.exists():
        lower_dir.mkdir(parents=True, exist_ok=True)
        run('train_lower', 'lower_controller_3d.train', [
            '--headless', '--num_envs', 2048, '--iterations', 1000,
            '--height_range', '0.06', '--height_ramp', 300,
            '--resume', prev_lower, '--run_name', 'lower3d_v3_z06',
            '--log_dir', lower_dir, '--save_interval', 100],
            lower_selected)
        checkpoints = [int(re.match(r'model_(\d+)\.pt$', p.name).group(1))
                       for p in lower_dir.glob('model_*.pt')]
        if not checkpoints:
            raise RuntimeError('lower training produced no checkpoints')
        lower_ckpt = lower_dir / f'model_{max(checkpoints)}.pt'
        lower_selected.write_text(str(lower_ckpt) + '\n')
        publish('train_lower', 'checkpoint_selected', checkpoint=str(lower_ckpt))
    lower = lower_selected.read_text().strip()

    # 2) Collect 8 shards with the new lower.
    inputs = []
    for seed in shard_seeds:
        out = root / f'replay_256_seed{seed}'
        run(f'collect_{seed}', 'composite_terrain.run', [
            '--lower_checkpoint', lower,
            '--output', out, '--num_envs', 256, '--steps', 8000, '--seed', seed,
            '--difficulty_levels', difficulty_levels, '--behavior', 'diverse',
            '--collect', '--record_motion', '--random_action_prob', '.12',
            '--geodesic_guidance', '--reset_region_prob', '.35', '--headless'],
            out / 'transitions.npz')
        run(f'audit_{seed}', 'composite_terrain.audit_replay',
            [out / 'transitions.npz'], out / 'replay_audit.json')
        inputs.append(out / 'transitions.npz')

    # 3) Merge + audit.
    dataset = root / 'merged' / 'arrays'
    run('merge', 'composite_terrain.merge_replays',
        ['--inputs', *inputs, '--output', dataset, '--memmap'],
        dataset.parent / 'metrics.json')
    run('audit_merged', 'composite_terrain.audit_replay', [dataset],
        dataset.parent / 'replay_audit.json')

    # 4) Larger world model, trained from scratch, more updates.
    model_dir = Path('runs') / name
    run('train_world', 'composite_terrain.train', [
        '--dataset', dataset, '--output', model_dir,
        '--h1_updates', world_updates, '--h3_updates', 0,
        '--batch_size', 1024, '--hidden_dim', hidden_dim,
        '--geometry_dim', geometry_dim, '--dynamics_dim', dynamics_dim],
        model_dir / 'summary.json')
    world = model_dir / 'h1_best.pt'

    # 5) Motion model.
    run('train_motion', 'composite_terrain.train_motion', [
        '--dataset', dataset, '--world_checkpoint', world,
        '--output', model_dir / 'motion', '--updates', motion_updates],
        model_dir / 'motion' / 'summary.json')
    motion = model_dir / 'motion' / 'model_best.pt'

    # 6) Paired evaluation on hard seeds.
    for seed in eval_seeds:
        run(f'eval_h1_seed{seed}', 'composite_terrain.run', [
            '--lower_checkpoint', lower, '--checkpoint', world,
            '--motion_checkpoint', motion, '--output',
            root / f'h1_64_seed{seed}', '--num_envs', 64, '--steps', 3000,
            '--seed', seed, '--difficulty', eval_difficulty, '--horizon', 1,
            '--behavior', 'model', '--root_geometry_guard', '--headless'],
            root / f'h1_64_seed{seed}' / 'metrics.json')
        run(f'eval_greedy_seed{seed}', 'composite_terrain.run', [
            '--lower_checkpoint', lower, '--output',
            root / f'greedy_64_seed{seed}', '--num_envs', 64, '--steps', 3000,
            '--seed', seed, '--difficulty', eval_difficulty, '--behavior',
            'greedy', '--headless'],
            root / f'greedy_64_seed{seed}' / 'metrics.json')

    # 7) Videos.
    for seed in [eval_seeds[0] + 1, eval_seeds[0] + 2]:
        out = root / f'video_seed{seed}'
        run(f'video_{seed}', 'composite_terrain.run', [
            '--lower_checkpoint', lower, '--checkpoint', world,
            '--motion_checkpoint', motion, '--output', out, '--num_envs', 1,
            '--steps', 3000, '--seed', seed, '--difficulty', eval_difficulty,
            '--horizon', 1, '--behavior', 'model', '--root_geometry_guard',
            '--record_video'], out / 'rollout.mp4')
        run(f'plot_{seed}', 'composite_terrain.plot_rollout', [out],
            out / 'trajectory.png')

    publish('cycle', 'complete', lower_checkpoint=lower)


if __name__ == '__main__':
    main()
