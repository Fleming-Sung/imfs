#!/usr/bin/env python3
"""Save auditable top views for every upper-planner terrain family."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from upper_planner.terrain import TerrainSpec, generate_support_layout


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "experiments" / "terrain_gallery")
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def main():
    args = arguments()
    args.output.mkdir(parents=True, exist_ok=True)
    summary = []
    items = [(kind, args.seed) for kind in (
        "straight", "s_curve", "fork", "random",
        "narrow_bridge", "hurdles", "irregular_support")]
    items.extend(("random_composite", args.seed + offset) for offset in range(8))
    for kind, terrain_seed in items:
        corridor_width = 1.60 if kind == "hurdles" else (0.60 if kind == "narrow_bridge" else 0.70)
        layout = generate_support_layout(TerrainSpec(
            kind=kind, length_m=4.5 if kind in (
                "narrow_bridge", "hurdles", "irregular_support") else 6.0,
            corridor_width_m=corridor_width, seed=terrain_seed,
            hurdle_height_min_m=0.025, hurdle_height_max_m=0.05))
        image = Image.fromarray(np.uint8(layout.support_mask) * 210, "L").convert("RGB")
        draw = ImageDraw.Draw(image)
        to_pixel = lambda point: (
            int(round(point[0] / layout.spec.resolution_m)),
            int(round((0.5 * layout.spec.width_m - point[1]) / layout.spec.resolution_m)),
        )
        sy, sx = to_pixel(layout.start_xy)
        gy, gx = to_pixel(layout.goal_xy)
        r = 3
        draw.ellipse((sy - r, sx - r, sy + r, sx + r), fill=(30, 220, 80))
        draw.ellipse((gy - r, gx - r, gy + r, gx + r), fill=(230, 50, 40))
        for cx, cy, sx_m, sy_m, height in layout.obstacle_rectangles:
            x0, y0 = to_pixel((cx - 0.5 * sx_m, cy + 0.5 * sy_m))
            x1, y1 = to_pixel((cx + 0.5 * sx_m, cy - 0.5 * sy_m))
            draw.rectangle((x0, y0, x1, y1), fill=(220, 45, 35))
        name = ("{}_seed{:03d}".format(kind, terrain_seed)
                if kind == "random_composite" else kind)
        image.resize((960, 480), Image.Resampling.NEAREST).save(args.output / f"{name}.png")
        summary.append({
            "kind": kind,
            "seed": terrain_seed,
            "shape_yx": list(layout.support_mask.shape),
            "support_fraction": float(layout.support_mask.mean()),
            "start_xy": layout.start_xy.tolist(),
            "goal_xy": layout.goal_xy.tolist(),
            "corridor_width_m": layout.spec.corridor_width_m,
            "pit_depth_m": layout.spec.pit_depth_m,
            "support_rectangles": len(layout.support_rectangles),
            "obstacle_rectangles": len(layout.obstacle_rectangles),
        })
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
