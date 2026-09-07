"""Failure-mode analysis: which terrains fail, and whether the candidate set or
the lower's z-capability is the bottleneck.

Reads the round-2 evaluation episodes and the merged hard-terrain replay, then
emits charts + a JSON of statistics used by the illustrated report.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from composite_terrain.maps import CompositeSpec, generate_route, FAMILIES
EVAL_SEEDS = [9801, 9901]
REPLAY_DIR = ROOT / 'experiments/composite3d/composite3d_hard_big/merged/arrays'
OUT = ROOT / 'docs/assets/failure_analysis'
OUT.mkdir(parents=True, exist_ok=True)

FAMILY_LABELS = {
    'ramp': 'Ramp', 'stairs': 'Stairs', 'stones': 'Stones',
    'bridge': 'Bridge', 'slalom': 'Slalom', 'rough': 'Rough',
    'curved_bridge': 'Curved bridge',
}


def load_eval_episodes():
    """Map every finished episode to the terrain family where it terminated."""
    rows = []
    for seed in EVAL_SEEDS:
        base = ROOT / f'experiments/composite3d/composite3d_hard_big/h1_64_seed{seed}'
        d = np.load(base / 'completed_episodes.npz')
        for i in range(len(d['env_id'])):
            route = generate_route(CompositeSpec(
                seed=int(seed) * 1000 + int(d['env_id'][i]), difficulty=0.5))
            kind = route['segments'][int(d['terminal_region'][i])]['kind']
            rows.append(dict(
                kind=kind, fall=bool(d['fall'][i]), timeout=bool(d['timeout'][i]),
                success=bool(d['success'][i]), max_x=float(d['max_x'][i]),
                ticks=int(d['ticks'][i]), seed=seed))
    return rows


def load_replay_cols():
    def mem(name):
        return np.load(REPLAY_DIR / (name + '.npy'), mmap_mode='r')
    return dict(
        kind=mem('terrain_kind'), fall=mem('fall'), collision=mem('collision'),
        valid=mem('candidate_valid'), support=mem('support'),
        touchdown=mem('touchdown_error'), action=mem('action'),
        difficulty=mem('difficulty'))


def main():
    episodes = load_eval_episodes()
    rp = load_replay_cols()
    kind = rp['kind'].astype(str)
    report = {}

    # ---- Replay-level per-family statistics ----
    replay_stats = {}
    for fam in FAMILIES:
        m = kind == fam
        if not m.any():
            continue
        n_valid = rp['valid'][m].sum(-1)
        replay_stats[fam] = dict(
            transitions=int(m.sum()),
            fall_rate=float(rp['fall'][m].mean()),
            collision_rate=float(rp['collision'][m].mean()),
            mean_valid_candidates=float(n_valid.mean()),
            frac_states_le3_valid=float((n_valid <= 3).mean()),
            frac_states_zero_valid=float((n_valid == 0).mean()),
            mean_chosen_support=float(rp['support'][m].mean()),
            mean_touchdown_xy=float(rp['touchdown'][m].mean()),
            mean_abs_dz=float(np.abs(rp['action'][m, 2]).mean()),
        )
    report['replay_per_family'] = replay_stats

    # ---- Evaluation per-family termination statistics ----
    eval_stats = {fam: dict(fall=0, timeout=0, success=0, max_x=[]) for fam in FAMILIES}
    for e in episodes:
        s = eval_stats[e['kind']]
        if e['success']:
            s['success'] += 1
        elif e['fall']:
            s['fall'] += 1
        else:
            s['timeout'] += 1
        s['max_x'].append(e['max_x'])
    for fam in FAMILIES:
        s = eval_stats[fam]
        s['total'] = s['fall'] + s['timeout'] + s['success']
        s['mean_max_x'] = float(np.mean(s['max_x'])) if s['max_x'] else 0.0
        s.pop('max_x')
    report['eval_per_family'] = eval_stats

    # ---- Fall vs non-fall: candidate availability and dz ----
    fall_m = rp['fall'].astype(bool)
    report['fall_vs_safe'] = dict(
        fall_states=int(fall_m.sum()),
        fall_mean_valid=float(rp['valid'][fall_m].sum(-1).mean()),
        safe_mean_valid=float(rp['valid'][~fall_m].sum(-1).mean()),
        fall_mean_abs_dz=float(np.abs(rp['action'][fall_m, 2]).mean()),
        safe_mean_abs_dz=float(np.abs(rp['action'][~fall_m, 2]).mean()),
        fall_mean_support=float(rp['support'][fall_m].mean()),
        safe_mean_support=float(rp['support'][~fall_m].mean()),
    )

    # ---- Chosen dz distribution (normalized action axis 2 -> cm) ----
    dz_cm = rp['action'][:, 2] * 8.0
    dz_hist = {}
    for fam in FAMILIES:
        m = kind == fam
        if m.any():
            dz_hist[fam] = np.bincount(
                np.digitize(dz_cm[m], [-9, -5, -1, 1, 5, 9])).tolist()
    report['dz_histogram_bins'] = ['<-8', '-8..-4', '-4..0', '0..4', '4..8', '>8']
    report['dz_histogram'] = dz_hist

    # ---- Figures ----
    plt.rcParams.update({'figure.dpi': 110, 'font.size': 9})
    order = sorted(FAMILIES, key=lambda f: -replay_stats.get(f, {}).get('fall_rate', 0))

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    # (a) replay fall rate
    ax = axes[0, 0]
    vals = [replay_stats[f]['fall_rate'] * 100 for f in order]
    ax.bar([FAMILY_LABELS[f] for f in order], vals, color='#d9534f')
    ax.set_title('Replay fall rate by terrain family')
    ax.set_ylabel('fall rate (%)'); ax.tick_params(axis='x', rotation=30)
    for i, v in enumerate(vals):
        ax.text(i, v, f'{v:.1f}', ha='center', va='bottom', fontsize=8)
    # (b) eval fall count
    ax = axes[0, 1]
    vals = [eval_stats[f]['fall'] for f in order]
    ax.bar([FAMILY_LABELS[f] for f in order], vals, color='#f0ad4e')
    ax.set_title('Evaluation falls by termination family (H1, 2 seeds)')
    ax.set_ylabel('fall episodes'); ax.tick_params(axis='x', rotation=30)
    for i, v in enumerate(vals):
        ax.text(i, v, str(v), ha='center', va='bottom', fontsize=8)
    # (c) mean valid candidates
    ax = axes[1, 0]
    vals = [replay_stats[f]['mean_valid_candidates'] for f in order]
    ax.bar([FAMILY_LABELS[f] for f in order], vals, color='#5bc0de')
    ax.axhline(180, color='gray', ls='--', lw=0.8)
    ax.set_title('Mean geometrically-valid candidates (of 180)')
    ax.set_ylabel('valid candidates'); ax.tick_params(axis='x', rotation=30)
    for i, v in enumerate(vals):
        ax.text(i, v, f'{v:.0f}', ha='center', va='bottom', fontsize=8)
    # (d) zero-valid fraction
    ax = axes[1, 1]
    vals = [replay_stats[f]['frac_states_zero_valid'] * 100 for f in order]
    ax.bar([FAMILY_LABELS[f] for f in order], vals, color='#5cb85c')
    ax.set_title('States with zero valid candidates (%)')
    ax.set_ylabel('% of states'); ax.tick_params(axis='x', rotation=30)
    for i, v in enumerate(vals):
        ax.text(i, v, f'{v:.1f}', ha='center', va='bottom', fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / 'failure_by_family.png', bbox_inches='tight')
    plt.close(fig)

    # dz histogram per family (normalized rows)
    fig, ax = plt.subplots(figsize=(11, 4.5))
    bins = report['dz_histogram_bins']
    fams = ['stairs', 'stones', 'bridge', 'slalom', 'curved_bridge']
    x = np.arange(len(bins))
    width = 0.16
    for j, fam in enumerate(fams):
        h = np.array(dz_hist.get(fam, [0] * len(bins)), dtype=float)
        h = h / h.sum() if h.sum() else h
        ax.bar(x + (j - 2) * width, h * 100, width, label=FAMILY_LABELS[fam])
    ax.set_xticks(x); ax.set_xticklabels(bins)
    ax.set_xlabel('commanded target dz (cm)'); ax.set_ylabel('% of decisions')
    ax.set_title('Distribution of commanded foothold height change (dz)')
    ax.legend(ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / 'dz_distribution.png', bbox_inches='tight')
    plt.close(fig)

    (OUT / 'stats.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
