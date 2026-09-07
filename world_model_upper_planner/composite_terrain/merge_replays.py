"""Merge compatible composite replays; publish metadata only after completion."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--inputs', type=Path, nargs='+', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--wait_seconds', type=float, default=1800)
    p.add_argument('--memmap', action='store_true')
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    deadline = time.monotonic() + a.wait_seconds
    while not all(v.exists() and (v.parent / 'metrics.json').exists() for v in a.inputs):
        if time.monotonic() > deadline:
            raise TimeoutError('input collector did not complete')
        time.sleep(5)
    meta = [json.loads((v.parent / 'metrics.json').read_text()) for v in a.inputs]
    for m in meta:
        m.setdefault('static_geometry_contract', 'proxy_16x7cm_v1')
    contract = ['lower_sha256', 'proprio_dim', 'physics_contract',
                'transition_contract', 'success_contract', 'progress_kind',
                'static_geometry_contract']
    for key in contract:
        if any(key not in m or m[key] != meta[0][key] for m in meta):
            raise ValueError(f'incompatible replay contract: {key}')
    start = time.perf_counter()
    subprocess.run([sys.executable, '-m', 'scripts.merge_datasets', '--inputs',
                    *map(str, a.inputs), '--output', str(a.output)]
                   + (['--memmap'] if a.memmap else []), check=True)
    manifest = json.loads((a.output/'manifest.json' if a.memmap else a.output.with_suffix('.json')).read_text())
    result = {key: meta[0][key] for key in contract}
    result.update(transitions=manifest['rows'], num_envs=manifest['environments'],
        lower_steps=sum(m['lower_steps'] for m in meta), sources=manifest['sources'],
        source_collection_seconds=sum(m['source_collection_seconds'] if 'source_collection_seconds' in m
                                      else m['wall_seconds'] for m in meta),
        merge_seconds=time.perf_counter() - start,
        note='Mixed geometry-guided and learned-policy physical replay; not an evaluation.')
    (a.output.parent / 'metrics.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
