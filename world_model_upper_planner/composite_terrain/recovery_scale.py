"""Scale-up round for recoverable foothold + long-term value (2026-09-08).

Motivation (from video review): the extended actions fix stones/turns reachability,
but the robot stops aiming at the goal near arrival / after hard terrain because the
value function under-learned that completion carries a large terminal reward and
stalling is otherwise free.

Changes in this round:
- reward: larger terminal arrival bonus (success=20) + per-option time cost
  (time_penalty=0.05) so posture adjustment is allowed but endless stalling costs;
- data scaled ~8-10x (6 x 256-env shards) with denser per-route difficulty
  randomization (.1-.8) on the same clearance terrain principles;
- long full training (24k updates, batch 1024) so the value function can learn the
  arrival signal; motion + 3-step sequence calibrated.
- evaluation: recovery multi-step vs recovery single-step vs the legacy-actions
  reference, plus a value-weight (2.0) variant to test goal dominance weighting.
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

_PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT not in sys.path:
    sys.path.insert(0, _PROJECT)


def main():
    os.chdir(Path(__file__).resolve().parents[1])
    name = 'recovery_scale'
    root = Path('experiments/composite3d') / name
    root.mkdir(parents=True, exist_ok=True)
    journal = root / 'stages.jsonl'
    live = Path('docs/recovery_scale_live.md')

    lower = 'runs/lower3d_v3_z06/model_8201.pt'
    reference = 'runs/composite3d_value_h3_v2/h3_best.pt'
    reference_motion = 'runs/composite3d_value_h3_v2/motion_sequence/model_best.pt'
    warm_start = 'runs/recovery_study/h3_best.pt'      # recovery-action prior model
    shard_seeds = list(range(9480, 9486))              # 6 shards
    eval_seeds = [9811, 9911]
    difficulty_levels = '.1,.2,.3,.4,.5,.6,.7,.8'
    h1_updates, h3_updates = 14000, 10000
    motion_updates, sequence_updates = 5000, 1500

    def event(stage, status, **extra):
        with journal.open('a') as f:
            f.write(json.dumps(dict(stage=stage, status=status, time=time.time(), **extra)) + '\n')
        latest = {}
        for line in journal.read_text().splitlines():
            v = json.loads(line)
            latest[v['stage']] = v
        live.write_text('# 扩量可恢复落脚 + 长期价值：自动进度\n\n'
            '新 reward（到达 +20、每步时间成本 0.05）、~8-10x 数据、更密难度随机化、充分训练。\n\n'
            '| 阶段 | 状态 | 秒 |\n|---|---|---:|\n' + ''.join(
                f"| {k} | {v['status']} | {v.get('seconds', '')} |\n"
                for k, v in latest.items()) +
            '\n结果以结束的物理测试为准。\n')

    def run(stage, module, args, completion):
        if completion.exists():
            event(stage, 'already_complete')
            return
        event(stage, 'running')
        start = time.perf_counter()
        with (root / (stage + '.log')).open('a') as f:
            result = subprocess.run([sys.executable, '-m', module, *map(str, args)],
                                    stdout=f, stderr=subprocess.STDOUT)
        event(stage, 'complete' if result.returncode == 0 else 'failed',
              seconds=round(time.perf_counter() - start, 2))
        if result.returncode:
            raise RuntimeError(f'{stage} failed; inspect {root}')

    # 1) Collect: 6 shards, extended actions, clearance terrain, new reward.
    inputs = []
    for seed in shard_seeds:
        out = root / f'replay_{seed}'
        run(f'collect_{seed}', 'composite_terrain.run', [
            '--lower_checkpoint', lower, '--output', out, '--num_envs', 256,
            '--steps', 8000, '--seed', seed, '--difficulty_levels', difficulty_levels,
            '--terrain_profile', 'clearance', '--action_profile', 'recovery',
            '--behavior', 'diverse', '--collect', '--record_motion',
            '--geodesic_guidance', '--random_action_prob', '.25',
            '--reset_region_prob', '.35', '--success_reward', '20', '--time_penalty', '.05',
            '--headless'], out / 'metrics.json')
        run(f'audit_{seed}', 'composite_terrain.audit_replay',
            [out / 'transitions.npz'], out / 'replay_audit.json')
        inputs.append(out / 'transitions.npz')

    # 2) Merge + audit.
    data = root / 'merged' / 'arrays'
    run('merge', 'composite_terrain.merge_replays',
        ['--inputs', *inputs, '--output', data, '--memmap'], data.parent / 'metrics.json')
    run('audit_merged', 'composite_terrain.audit_replay', [data], data.parent / 'replay_audit.json')

    # 3) Long world-model training (transfer recovery prior, full value learning).
    model = Path('runs') / name
    run('world', 'composite_terrain.train', [
        '--dataset', data, '--output', model, '--init', warm_start,
        '--transfer_candidates', '--hidden_dim', 512, '--geometry_dim', 128,
        '--dynamics_dim', 128, '--h1_updates', h1_updates, '--h3_updates', h3_updates,
        # batch 384: 756 candidates + H3 3-step unroll OOMs the 48GB GPU at 1024.
        '--batch_size', 384], model / 'summary.json')
    world = model / 'h3_best.pt'

    # 4) Motion + 3-step sequence.
    run('motion', 'composite_terrain.train_motion', [
        '--dataset', data, '--world_checkpoint', world,
        '--output', model / 'motion', '--updates', motion_updates], model / 'motion/summary.json')
    run('sequence', 'composite_terrain.train_motion_sequence', [
        '--dataset', data, '--init', model / 'motion/model_best.pt',
        '--output', model / 'sequence', '--updates', sequence_updates], model / 'sequence/summary.json')
    motion = model / 'sequence/model_best.pt'

    # 5) Paired evaluations (legacy reference vs recovery multi/single + weight sweep).
    base = ['--lower_checkpoint', lower, '--terrain_profile', 'clearance', '--difficulty', '.5',
            '--behavior', 'model', '--root_geometry_guard']
    variants = [
        ('original', reference, reference_motion, 'legacy', 3, 'legacy', 8, '1'),
        ('recovery', world, motion, 'recovery', 3, 'legacy', 12, '1'),
        ('recovery_vw2', world, motion, 'recovery', 3, 'legacy', 12, '2'),
        ('single_step', world, motion, 'recovery', 1, 'legacy', 12, '1'),
    ]
    for seed in eval_seeds:
        for tag, w, m, profile, horizon, objective, proposals, vw in variants:
            out = root / f'{tag}_seed{seed}'
            run(f'eval_{tag}_{seed}', 'composite_terrain.run', base + [
                '--checkpoint', w, '--motion_checkpoint', m, '--action_profile', profile,
                '--planning_objective', objective, '--horizon', horizon,
                '--proposals_per_beam', proposals, '--value_weight', vw,
                '--output', out, '--num_envs', 64, '--steps', 3000,
                '--seed', seed, '--headless'], out / 'metrics.json')
            if tag != 'original':
                from composite_terrain.compare_runs import compare
                (root / f'paired_{tag}_{seed}.json').write_text(
                    json.dumps(compare(root / f'original_seed{seed}', out), indent=2))

    # 6) Videos.
    for seed in [9812, 9912]:
        out = root / f'video_seed{seed}'
        run(f'video_{seed}', 'composite_terrain.run', base + [
            '--checkpoint', world, '--motion_checkpoint', motion,
            '--action_profile', 'recovery', '--horizon', 3, '--proposals_per_beam', 12,
            '--value_weight', '1', '--output', out, '--num_envs', 1, '--steps', 3000,
            '--seed', seed, '--record_video'], out / 'rollout.mp4')
        run(f'plot_{seed}', 'composite_terrain.plot_rollout', [out], out / 'trajectory.png')

    event('cycle', 'complete')


if __name__ == '__main__':
    main()
