"""Value / multi-step round (dev-2026-09-07, round 3).

Changes under test: 2-cm dz candidates (324 grid), minimal-step fallback for
zero-valid states, learned twin-Q value in the planner score, and horizon-3
multi-step planning with sequence-calibrated motion.

Pipeline: collect -> merge -> train H1+H3 -> motion(+sequence) -> paired eval
(H3 with Q vs H1 with Q vs greedy) -> videos. Resumable via completion markers.
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    os.chdir(Path(__file__).resolve().parents[1])
    name = 'composite3d_value_h3'
    root = Path('experiments/composite3d') / name
    root.mkdir(parents=True, exist_ok=True)
    journal = root / 'stages.jsonl'
    status = Path('docs/composite3d_value_h3_live.md')

    lower = 'runs/lower3d_v3_z06/model_8201.pt'
    difficulty_levels = '.3,.5,.7'
    shard_seeds = list(range(9431, 9439))  # 8 shards
    eval_seeds = [9801, 9901]
    eval_difficulty = '.5'
    h1_updates, h3_updates = 20000, 10000
    motion_updates, sequence_updates = 5000, 1000
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
        lines = ['# 多步价值 + 细粒度 dz 训练：自动进度', '',
                 '候选 dz 2 cm（324 候选）、0 候选回退、学习 Q 价值进评分、H3 多步规划。', '',
                 '| 阶段 | 状态 | 墙钟秒 |', '|---|---|---:|']
        for r in latest.values():
            lines.append(f"| {r['stage']} | {r['status']} | {round(r.get('wall_seconds', 0), 1)} |")
        lines += ['', f'原始日志：`{root}`。TensorBoard：`runs/{name}/tb`、'
                  f'`runs/{name}/motion/tb`。']
        status.write_text('\n'.join(lines) + '\n')

    def run(stage, module, args, completion):
        if completion.exists():
            publish(stage, 'already_complete')
            return
        publish(stage, 'running')
        start = time.perf_counter()
        with (root / (stage + '.log')).open('a') as stream:
            result = subprocess.run(
                [sys.executable, '-m', module, *map(str, args)],
                stdout=stream, stderr=subprocess.STDOUT)
        publish(stage, 'complete' if result.returncode == 0 else 'failed',
                wall_seconds=time.perf_counter() - start,
                returncode=result.returncode)
        if result.returncode:
            raise RuntimeError(f'{stage} failed; see {root / (stage + ".log")}')

    # 1) Collect with the new 324-candidate grid.
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

    # 2) Merge + audit.
    dataset = root / 'merged' / 'arrays'
    run('merge', 'composite_terrain.merge_replays',
        ['--inputs', *inputs, '--output', dataset, '--memmap'],
        dataset.parent / 'metrics.json')
    run('audit_merged', 'composite_terrain.audit_replay', [dataset],
        dataset.parent / 'replay_audit.json')

    # 3) World model: H1 warm-up then H3 multi-step sequence learning.
    model_dir = Path('runs') / name
    run('train_world', 'composite_terrain.train', [
        '--dataset', dataset, '--output', model_dir,
        '--h1_updates', h1_updates, '--h3_updates', h3_updates,
        '--batch_size', 1024, '--hidden_dim', hidden_dim,
        '--geometry_dim', geometry_dim, '--dynamics_dim', dynamics_dim],
        model_dir / 'summary.json')
    world = model_dir / 'h3_best.pt' if h3_updates else model_dir / 'h1_best.pt'

    # 4) Motion model + 3-step sequence calibration.
    run('train_motion', 'composite_terrain.train_motion', [
        '--dataset', dataset, '--world_checkpoint', world,
        '--output', model_dir / 'motion', '--updates', motion_updates],
        model_dir / 'motion' / 'summary.json')
    motion = model_dir / 'motion' / 'model_best.pt'
    run('motion_sequence', 'composite_terrain.train_motion_sequence', [
        '--dataset', dataset, '--init', motion,
        '--output', model_dir / 'motion_sequence', '--updates', sequence_updates],
        model_dir / 'motion_sequence' / 'summary.json')
    motion = model_dir / 'motion_sequence' / 'model_best.pt'

    # 5) Paired evaluation: H3 with Q value, H1 with Q value, greedy.
    for seed in eval_seeds:
        for horizon, tag in [(3, 'h3'), (1, 'h1')]:
            run(f'eval_{tag}_seed{seed}', 'composite_terrain.run', [
                '--lower_checkpoint', lower, '--checkpoint', world,
                '--motion_checkpoint', motion, '--output',
                root / f'{tag}_64_seed{seed}', '--num_envs', 64, '--steps', 3000,
                '--seed', seed, '--difficulty', eval_difficulty,
                '--horizon', horizon, '--value_weight', 1.0,
                '--behavior', 'model', '--root_geometry_guard', '--headless'],
                root / f'{tag}_64_seed{seed}' / 'metrics.json')
        run(f'eval_greedy_seed{seed}', 'composite_terrain.run', [
            '--lower_checkpoint', lower, '--output',
            root / f'greedy_64_seed{seed}', '--num_envs', 64, '--steps', 3000,
            '--seed', seed, '--difficulty', eval_difficulty, '--behavior',
            'greedy', '--headless'],
            root / f'greedy_64_seed{seed}' / 'metrics.json')

    # 6) Videos (H3 with Q value).
    for seed in [eval_seeds[0] + 1, eval_seeds[0] + 2]:
        out = root / f'video_seed{seed}'
        run(f'video_{seed}', 'composite_terrain.run', [
            '--lower_checkpoint', lower, '--checkpoint', world,
            '--motion_checkpoint', motion, '--output', out, '--num_envs', 1,
            '--steps', 3000, '--seed', seed, '--difficulty', eval_difficulty,
            '--horizon', 3, '--value_weight', 1.0,
            '--behavior', 'model', '--root_geometry_guard', '--record_video'],
            out / 'rollout.mp4')
        run(f'plot_{seed}', 'composite_terrain.plot_rollout', [out],
            out / 'trajectory.png')

    publish('cycle', 'complete', world=str(world), lower=lower)


if __name__ == '__main__':
    main()
