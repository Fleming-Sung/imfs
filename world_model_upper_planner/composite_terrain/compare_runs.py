"""Paired layout outcomes and first-attempt audit, without rerunning physics."""
import argparse
import json
from pathlib import Path
import numpy as np


def summarize(run):
    m = json.loads((run / 'metrics.json').read_text())
    n = m['num_envs']
    # Older fixed-difficulty runs predate per-layout difficulty metadata.
    m.setdefault('route_difficulties', [m['difficulty']] * n)
    first_success = np.zeros(n, dtype=bool)
    first_fall = np.zeros(n, dtype=bool)
    first_timeout = np.zeros(n, dtype=bool)
    ended = np.zeros(n, dtype=bool)
    times = []
    path = run / 'completed_episodes.npz'
    if path.exists():
        with np.load(path) as data:
            for i in np.flatnonzero(data['episode_id'] == 0):
                env = int(data['env_id'][i])
                ended[env] = True
                first_success[env] = data['success'][i]
                first_fall[env] = data['fall'][i]
                first_timeout[env] = data['timeout'][i]
                if first_success[env]:
                    times.append(float(data['ticks'][i]) * .02)
    result = dict(run=str(run), maps=n,
        maps_completed=int(np.count_nonzero(m['successes_per_env'])),
        completed_episodes=m['successes'], falls=m['falls'], timeouts=m['timeouts'],
        censored_episodes=m['right_censored_episodes'],
        first_attempt_success=int(first_success.sum()),
        first_attempt_fall=int(first_fall.sum()),
        first_attempt_timeout=int(first_timeout.sum()),
        first_attempt_censored=int((~ended).sum()),
        first_success_time_median_s=float(np.median(times)) if times else None)
    return m, result, first_success


def compare(reference, candidate):
    old, old_stats, old_first = summarize(reference)
    new, new_stats, new_first = summarize(candidate)
    # A changed horizon/model is the treatment, not a mismatched evaluation.
    for key in ['lower_sha256', 'seed', 'route_offset', 'num_envs', 'lower_steps',
                'route_difficulties', 'physics_contract', 'success_contract',
                'reset_region_prob']:
        if key not in old or key not in new or old[key] != new[key]:
            raise ValueError(f'not a paired evaluation: {key}')
    def paired(a, b):
        return dict(both=int((a & b).sum()), reference_only=int((a & ~b).sum()),
                    candidate_only=int((~a & b).sum()), neither=int((~a & ~b).sum()))
    return dict(reference=old_stats, candidate=new_stats,
        map_completion_pairs=paired(np.array(old['successes_per_env']) > 0,
                                    np.array(new['successes_per_env']) > 0),
        first_attempt_pairs=paired(old_first, new_first),
        interpretation='Descriptive paired counts, not an independent-seed significance claim.')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('reference', type=Path); p.add_argument('candidate', type=Path)
    p.add_argument('--output', type=Path)
    a = p.parse_args()
    result = compare(a.reference, a.candidate)
    content = json.dumps(result, indent=2)
    if a.output:
        a.output.parent.mkdir(parents=True, exist_ok=True)
        a.output.write_text(content + '\n')
    print(content)


if __name__ == '__main__':
    main()
