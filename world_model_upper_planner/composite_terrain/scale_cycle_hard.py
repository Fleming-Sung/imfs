"""Hard-terrain scale cycle: collect -> merge -> train (enough updates) -> eval.

Single, resumable run on the harder composite maps introduced on dev-2026-09-07.
The frozen XYZ+yaw lower is unchanged. Collection uses the geometric greedy/
diverse policy (not a learned planner) so the world model must discover, from
mostly-failure data, which 3D foothold chains are actually safe.

Each shard is 256 environments (the atlas triangle-mesh ceiling measured on
this machine; 512+ exceeds the mesh-build timeout). Several shards are merged
to reach enough environment interactions; training runs far more updates than
the earlier 2.5-6k warm-ups.
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    # Anchor all relative paths to this project regardless of launch cwd.
    os.chdir(Path(__file__).resolve().parents[1])
    name = 'composite3d_hard_h1'
    root = Path('experiments/composite3d') / name
    root.mkdir(parents=True, exist_ok=True)
    journal = root / 'stages.jsonl'
    status = Path('docs/composite3d_hard_live.md')
    lower = 'runs/lower3d_v2_z04/model_7200.pt'
    world_init = 'runs/composite3d_v8_refined/h1_best.pt'
    difficulty_levels = '.3,.5,.7'
    shard_seeds = [9400, 9401, 9402, 9403]
    eval_seeds = [9801, 9901]
    eval_difficulty = '.5'
    world_updates = 20000
    motion_updates = 5000

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
        lines = ['# 硬地形扩量训练：自动进度', '',
                 '固定冻结 XYZ+yaw 下层；地形已加难（台阶 4-6 cm 级、梅花桩 ±4-6 cm 高差、'
                 '更窄踏面与更大间隙）。采集用几何引导贪心/多样策略，非学习规划器。', '',
                 '| 阶段 | 状态 | 墙钟秒 |', '|---|---|---:|']
        for r in latest.values():
            lines.append(f"| {r['stage']} | {r['status']} | {round(r.get('wall_seconds', 0), 1)} |")
        lines += ['', f'原始日志：`{root}`。TensorBoard：训练写入 `runs/{name}/tb` 与 '
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

    # 1) Collection shards (256 envs x 6000 ticks each).
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

    # 2) Merge into a RAM/memmap pool.
    dataset = root / 'merged' / 'arrays'
    run('merge', 'composite_terrain.merge_replays',
        ['--inputs', *inputs, '--output', dataset, '--memmap'],
        dataset.parent / 'metrics.json')
    run('audit_merged', 'composite_terrain.audit_replay', [dataset],
        dataset.parent / 'replay_audit.json')

    # 3) World model: warm-start from the easy-terrain V8 encoder, then train
    #    long on hard-terrain data. H1 only (anchored branch).
    model_dir = Path('runs') / name
    run('train_world', 'composite_terrain.train', [
        '--dataset', dataset, '--output', model_dir,
        '--init', world_init, '--h1_updates', world_updates,
        '--h3_updates', 0, '--batch_size', 1024], model_dir / 'summary.json')
    world = model_dir / 'h1_best.pt'

    # 4) Motion model (physics-anchored rollout).
    run('train_motion', 'composite_terrain.train_motion', [
        '--dataset', dataset, '--world_checkpoint', world,
        '--output', model_dir / 'motion', '--updates', motion_updates],
        model_dir / 'motion' / 'summary.json')
    motion = model_dir / 'motion' / 'model_best.pt'

    # 5) Paired closed-loop evaluation on the same hard seeds: learned H1 vs
    #    the geometric greedy baseline (no learned planner).
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

    # 6) Two videos from the learned H1 planner on hard terrain.
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

    publish('cycle', 'complete')


if __name__ == '__main__':
    main()
