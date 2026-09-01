"""Deterministic same-height support masks for upper-planner research."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TerrainSpec:
    kind: str
    length_m: float = 6.0
    width_m: float = 3.0
    resolution_m: float = 0.05
    corridor_width_m: float = 0.70
    pit_depth_m: float = 0.40
    hurdle_height_min_m: float = 0.035
    hurdle_height_max_m: float = 0.085
    support_width_min_m: float = 0.55
    support_width_max_m: float = 1.20
    support_gap_min_m: float = 0.00
    support_gap_max_m: float = 0.10
    obstacle_probability: float = 0.45
    seed: int = 0


@dataclass(frozen=True)
class TerrainLayout:
    spec: TerrainSpec
    support_mask: np.ndarray
    height_m: np.ndarray
    x_m: np.ndarray
    y_m: np.ndarray
    centerlines: tuple
    start_xy: np.ndarray
    goal_xy: np.ndarray
    # Local-axis rectangles: support=(cx,cy,sx,sy), obstacle=(cx,cy,sx,sy,h).
    # They are optional so the original mask/mesh experiments remain unchanged.
    support_rectangles: tuple = ()
    obstacle_rectangles: tuple = ()


@dataclass(frozen=True)
class TiledHeightfield:
    height_samples: np.ndarray
    origin_xy_m: np.ndarray
    env_origins_xy_m: np.ndarray
    layouts: tuple
    horizontal_scale_m: float
    vertical_scale_m: float


@dataclass(frozen=True)
class SupportSegments:
    centers_xy_m: np.ndarray
    yaw_rad: np.ndarray
    active: np.ndarray
    segment_length_m: float
    segment_width_m: float
    reset_center_xy_m: np.ndarray
    reset_size_xy_m: np.ndarray


@dataclass(frozen=True)
class SupportTriangleMesh:
    vertices: np.ndarray
    triangles: np.ndarray


@dataclass(frozen=True)
class StaticBoxes:
    centers_xyz_m: np.ndarray
    sizes_xyz_m: np.ndarray
    active: np.ndarray
    obstacle: np.ndarray


def _random_centerline(x, rng, amplitude):
    knots_x = np.linspace(x[0], x[-1], 8)
    knots_y = rng.uniform(-amplitude, amplitude, len(knots_x))
    knots_y[0] = 0.0
    # Repeated local averaging limits abrupt direction changes without scipy.
    for _ in range(3):
        knots_y[1:-1] = 0.25 * knots_y[:-2] + 0.5 * knots_y[1:-1] + 0.25 * knots_y[2:]
    return np.interp(x, knots_x, knots_y)


def _centerlines(spec, x):
    amplitude = min(0.55, 0.5 * (spec.width_m - spec.corridor_width_m) - 0.10)
    if amplitude < 0.0:
        raise ValueError("terrain width is too small for the requested corridor")
    if spec.kind == "straight":
        return (np.zeros_like(x),)
    if spec.kind == "s_curve":
        return (amplitude * np.sin(2.0 * np.pi * x / spec.length_m),)
    if spec.kind == "random":
        return (_random_centerline(x, np.random.default_rng(spec.seed), amplitude),)
    if spec.kind == "fork":
        split = 0.35 * spec.length_m
        progress = np.clip((x - split) / (spec.length_m - split), 0.0, 1.0)
        branch = amplitude * np.sin(0.5 * np.pi * progress)
        return (branch, -branch)
    raise ValueError(f"unsupported terrain kind {spec.kind!r}")


def _rectangle_mask(x, y, rectangles):
    support = np.zeros((len(y), len(x)), dtype=bool)
    for cx, cy, sx, sy in rectangles:
        support |= ((np.abs(x[None, :] - cx) <= 0.5 * sx)
                    & (np.abs(y[:, None] - cy) <= 0.5 * sy))
    return support


def _typical_course_layout(spec, x, y):
    rng = np.random.default_rng(spec.seed)
    length = float(spec.length_m)
    if spec.kind == "narrow_bridge":
        width = float(spec.corridor_width_m)
        rectangles = (
            (0.225, 0.0, 1.05, 1.00),
            (0.5 * length, 0.0, length - 1.50, width),
            (length - 0.325, 0.0, 1.05, 1.00),
        )
        lines = (np.zeros_like(x),)
        goal_y = 0.0
        obstacles = ()
    elif spec.kind == "irregular_support":
        rectangles = [(0.225, 0.0, 1.05, 1.00)]
        pad_count = max(3, int(round((length - 1.5) / 0.5)) + 1)
        centers_x = np.linspace(1.00, length - 0.95, pad_count)
        center_y = 0.0
        line_x, line_y = [0.20], [0.0]
        for center_x in centers_x:
            center_y = float(np.clip(
                center_y + rng.uniform(-0.22, 0.22), -0.48, 0.48))
            size_x = float(rng.uniform(0.42, 0.56))
            size_y = float(rng.uniform(
                0.80 * spec.corridor_width_m, 1.10 * spec.corridor_width_m))
            rectangles.append((float(center_x), center_y, size_x, size_y))
            line_x.append(float(center_x))
            line_y.append(center_y)
        end_y = center_y
        rectangles.append((length - 0.325, end_y, 1.05, 1.00))
        lines = (np.interp(x, line_x + [length - 0.25],
                           line_y + [end_y]).astype(np.float32),)
        goal_y = end_y
        obstacles = ()
    elif spec.kind == "hurdles":
        ground_width = max(1.40, float(spec.corridor_width_m))
        rectangles = ((0.5 * length, 0.0, length + 0.60, ground_width),)
        # Roughly one fence per metre after the reset platform. Short
        # capability courses therefore contain one unambiguous crossing,
        # while the 4.5 m course retains three separated fences.
        count = max(1, int(round(length - 1.5)))
        positions = np.linspace(1.20, length - 1.10, count)
        # Start conservatively: the frozen lower policy was never trained on
        # target height, so clearance must be established empirically.
        heights = rng.uniform(
            float(spec.hurdle_height_min_m),
            float(spec.hurdle_height_max_m), count)
        obstacles = tuple((float(px), 0.0, 0.055, ground_width, float(height))
                          for px, height in zip(positions, heights))
        lines = (np.zeros_like(x),)
        goal_y = 0.0
    else:
        raise ValueError("not a typical course kind")
    support = _rectangle_mask(x, y, rectangles)
    height = np.where(support, 0.0, -float(spec.pit_depth_m)).astype(np.float32)
    return TerrainLayout(
        spec, support, height, x, y,
        tuple(np.asarray(line, dtype=np.float32) for line in lines),
        np.array([0.20, 0.0], dtype=np.float32),
        np.array([length - 0.25, goal_y], dtype=np.float32),
        tuple(rectangles), tuple(obstacles))


def _random_composite_layout(spec, x, y):
    """Build one fixed random course from visible support and obstacle boxes."""
    rng = np.random.default_rng(spec.seed)
    length = float(spec.length_m)
    width_min = float(spec.support_width_min_m)
    width_max = float(spec.support_width_max_m)
    gap_min = float(spec.support_gap_min_m)
    gap_max = float(spec.support_gap_max_m)
    if width_min <= 0.0 or width_max < width_min:
        raise ValueError("random support width range must be positive and ordered")
    if gap_min < 0.0 or gap_max < gap_min:
        raise ValueError("random support gap range must be non-negative and ordered")
    if not 0.0 <= float(spec.obstacle_probability) <= 1.0:
        raise ValueError("obstacle_probability must be in [0, 1]")

    rectangles = [(0.225, 0.0, 1.05, 1.00)]
    obstacles = []
    line_x, line_y = [0.20], [0.0]
    cursor = 0.75
    end_start = length - 0.85
    center_y = 0.0
    while cursor < end_start - 0.08:
        gap = float(rng.uniform(gap_min, gap_max))
        start = cursor + gap
        if start >= end_start - 0.08:
            break
        segment_length = min(float(rng.uniform(0.32, 0.68)), end_start - start)
        center_x = start + 0.5 * segment_length
        segment_width = float(rng.uniform(width_min, width_max))
        lateral_limit = max(
            0.0, 0.5 * float(spec.width_m) - 0.5 * segment_width - 0.12)
        center_y = float(np.clip(
            center_y + rng.uniform(-0.28, 0.28), -lateral_limit, lateral_limit))
        rectangles.append((center_x, center_y, segment_length, segment_width))
        line_x.append(center_x)
        line_y.append(center_y)

        if rng.random() < float(spec.obstacle_probability):
            obstacle_x = float(rng.uniform(
                start + 0.20 * segment_length, start + 0.80 * segment_length))
            height = float(rng.uniform(
                spec.hurdle_height_min_m, spec.hurdle_height_max_m))
            if rng.random() < 0.55:
                obstacles.append((
                    obstacle_x, center_y, float(rng.uniform(0.04, 0.075)),
                    0.90 * segment_width, height))
            else:
                block_width = min(
                    float(rng.uniform(0.18, 0.34)), 0.55 * segment_width)
                offset_limit = max(
                    0.0, 0.42 * segment_width - 0.5 * block_width)
                obstacle_y = center_y + float(
                    rng.uniform(-offset_limit, offset_limit))
                obstacles.append((
                    obstacle_x, obstacle_y, float(rng.uniform(0.16, 0.28)),
                    block_width, height))
        cursor = start + segment_length

    end_y = center_y
    rectangles.append((length - 0.325, end_y, 1.05, 1.00))
    line_x.append(length - 0.25)
    line_y.append(end_y)
    support = _rectangle_mask(x, y, rectangles)
    height = np.where(support, 0.0, -float(spec.pit_depth_m)).astype(np.float32)
    centerline = np.interp(x, line_x, line_y).astype(np.float32)
    return TerrainLayout(
        spec, support, height, x, y, (centerline,),
        np.array([0.20, 0.0], dtype=np.float32),
        np.array([length - 0.25, end_y], dtype=np.float32),
        tuple(rectangles), tuple(obstacles))


def _research_layout(spec, x, y):
    """Procedural planar research scenes beyond the original short demos."""
    rng = np.random.default_rng(spec.seed)
    length = float(spec.length_m)
    obstacles = []

    if spec.kind == "edge_cases":
        rectangles = [(0.225, 0.0, 1.05, 1.00)]
        cursor, center_y = 0.75, 0.0
        line_x, line_y = [0.20], [0.0]
        while cursor < length - 0.85:
            segment_length = min(float(rng.uniform(0.35, 0.75)), length - 0.85 - cursor)
            if segment_length <= 0.05:
                break
            width = float(rng.uniform(
                spec.support_width_min_m, spec.support_width_max_m))
            center_y = float(np.clip(
                center_y + rng.uniform(-0.32, 0.32), -0.65, 0.65))
            rectangles.append((cursor + 0.5 * segment_length, center_y,
                               segment_length, width))
            line_x.append(cursor + 0.5 * segment_length)
            line_y.append(center_y)
            cursor += segment_length + float(rng.uniform(
                spec.support_gap_min_m, spec.support_gap_max_m))
        rectangles.append((length - 0.325, center_y, 1.05, 1.00))
        line_x.append(length - 0.25)
        line_y.append(center_y)
        goal_y = center_y

    elif spec.kind == "stepping_stones":
        rectangles = [(0.225, 0.0, 1.05, 1.00)]
        center_y = 0.0
        spacing = float(rng.uniform(
            0.38 + spec.support_gap_min_m,
            0.42 + spec.support_gap_max_m))
        centers = np.arange(0.95, max(1.0, length - 0.75), spacing)
        line_x, line_y = [0.20], [0.0]
        for center_x in centers:
            center_y = float(np.clip(
                center_y + rng.uniform(-0.30, 0.30), -0.70, 0.70))
            length_min = max(0.26, spacing - spec.support_gap_max_m - 0.04)
            length_max = max(length_min + 0.04,
                             spacing - spec.support_gap_min_m + 0.08)
            rectangles.append((
                float(center_x), center_y,
                float(rng.uniform(length_min, min(0.58, length_max))),
                float(rng.uniform(
                    max(0.38, spec.support_width_min_m),
                    max(0.42, spec.support_width_max_m)))))
            line_x.append(float(center_x))
            line_y.append(center_y)
        rectangles.append((length - 0.325, center_y, 1.05, 1.00))
        line_x.append(length - 0.25)
        line_y.append(center_y)
        goal_y = center_y

    elif spec.kind == "turns":
        direction = -1.0 if rng.random() < 0.5 else 1.0
        offset = direction * float(rng.uniform(
            max(0.40, spec.support_width_min_m - 0.10),
            max(0.50, min(1.05, spec.support_width_max_m))))
        width = float(rng.uniform(
            max(0.48, spec.support_width_min_m),
            max(0.55, min(1.20, spec.support_width_max_m))))
        turn_x = float(rng.uniform(1.10, 1.55))
        rectangles = [
            (0.225, 0.0, 1.05, 1.00),
            (0.5 * turn_x, 0.0, turn_x, width),
            (turn_x, 0.5 * offset, width, abs(offset) + width),
            (0.5 * (turn_x + length), offset, length - turn_x, width),
            (length - 0.325, offset, 1.05, 1.00),
        ]
        line_x = [0.20, turn_x, length - 0.25]
        line_y = [0.0, offset, offset]
        goal_y = offset
        if rng.random() < float(spec.obstacle_probability):
            obstacles.append((turn_x + 0.35, offset - direction * 0.22,
                              0.22, 0.20, float(rng.uniform(0.30, 0.55))))

    elif spec.kind == "household":
        floor_width = float(rng.uniform(2.0, 2.8))
        rectangles = [(0.5 * length, 0.0, length + 0.50, floor_width)]
        goal_y = float(rng.uniform(-0.45, 0.45))
        line_x, line_y = [0.20, length - 0.25], [0.0, goal_y]
        # Alternating furniture blocks create doorways, slaloms and corner
        # approaches while keeping the same single downward-facing camera.
        for obstacle_x in np.linspace(1.05, length - 0.95, max(2, int(length - 1.0))):
            side = -1.0 if rng.random() < 0.5 else 1.0
            if rng.random() < 0.45:
                # Paired wall/furniture blocks leave a randomized doorway.
                opening = float(rng.uniform(0.68, 1.05))
                side_width = 0.5 * (floor_width - opening)
                for sign in (-1.0, 1.0):
                    center_y = sign * (0.5 * opening + 0.5 * side_width)
                    obstacles.append((float(obstacle_x), center_y, 0.18,
                                      side_width, float(rng.uniform(0.35, 0.70))))
            else:
                obstacles.append((float(obstacle_x), side * float(rng.uniform(0.20, 0.55)),
                                  float(rng.uniform(0.20, 0.42)),
                                  float(rng.uniform(0.18, 0.38)),
                                  float(rng.uniform(0.30, 0.65))))
                if rng.random() < 0.50:
                    # Thin chair/table leg on the opposite side.
                    obstacles.append((float(obstacle_x + rng.uniform(-0.16, 0.16)),
                                      -side * float(rng.uniform(0.25, 0.60)),
                                      0.10, 0.10, float(rng.uniform(0.35, 0.65))))
    else:
        raise ValueError("unsupported research terrain kind")

    support = _rectangle_mask(x, y, rectangles)
    height = np.where(support, 0.0, -float(spec.pit_depth_m)).astype(np.float32)
    centerline = np.interp(x, line_x, line_y).astype(np.float32)
    return TerrainLayout(
        spec, support, height, x, y, (centerline,),
        np.array([0.20, 0.0], dtype=np.float32),
        np.array([length - 0.25, goal_y], dtype=np.float32),
        tuple(rectangles), tuple(obstacles))


def generate_support_layout(spec):
    if spec.resolution_m <= 0.0 or spec.corridor_width_m <= 0.0:
        raise ValueError("resolution and corridor width must be positive")
    nx = int(round(spec.length_m / spec.resolution_m)) + 1
    ny = int(round(spec.width_m / spec.resolution_m)) + 1
    x = np.linspace(0.0, spec.length_m, nx, dtype=np.float32)
    y = np.linspace(-0.5 * spec.width_m, 0.5 * spec.width_m, ny, dtype=np.float32)
    if spec.kind == "random_composite":
        return _random_composite_layout(spec, x, y)
    if spec.kind in ("edge_cases", "stepping_stones", "turns", "household"):
        return _research_layout(spec, x, y)
    if spec.kind in ("narrow_bridge", "irregular_support", "hurdles"):
        return _typical_course_layout(spec, x, y)
    lines = _centerlines(spec, x)
    support = np.zeros((ny, nx), dtype=bool)
    for line in lines:
        support |= np.abs(y[:, None] - line[None, :]) <= 0.5 * spec.corridor_width_m
    # A generous reset patch is part of the task geometry, not a state constraint.
    support |= ((x[None, :] - 0.20) ** 2 + y[:, None] ** 2) <= 0.45 ** 2
    height = np.where(support, 0.0, -float(spec.pit_depth_m)).astype(np.float32)
    # A single-goal fork uses the first branch. Averaging branch endpoints would
    # place the goal in the unsupported gap between them.
    goal_y = float(lines[0][-1])
    return TerrainLayout(
        spec, support, height, x, y, tuple(np.asarray(line, dtype=np.float32) for line in lines),
        np.array([0.20, 0.0], dtype=np.float32),
        np.array([spec.length_m - 0.25, goal_y], dtype=np.float32),
    )


def build_static_boxes(layouts):
    """Pad per-environment support and hurdle rectangles to equal actor counts."""
    rows = []
    for layout in layouts:
        boxes = []
        depth = float(layout.spec.pit_depth_m)
        for cx, cy, sx, sy in layout.support_rectangles:
            boxes.append(((cx, cy, -0.5 * depth), (sx, sy, depth), False))
        for cx, cy, sx, sy, height in layout.obstacle_rectangles:
            boxes.append(((cx, cy, 0.5 * height), (sx, sy, height), True))
        rows.append(boxes)
    maximum = max(len(row) for row in rows)
    centers = np.zeros((len(rows), maximum, 3), dtype=np.float32)
    sizes = np.ones((len(rows), maximum, 3), dtype=np.float32) * 0.01
    active = np.zeros((len(rows), maximum), dtype=bool)
    obstacle = np.zeros((len(rows), maximum), dtype=bool)
    for env_id, row in enumerate(rows):
        for box_id, (center, size, is_obstacle) in enumerate(row):
            centers[env_id, box_id] = center
            sizes[env_id, box_id] = size
            active[env_id, box_id] = True
            obstacle[env_id, box_id] = is_obstacle
    return StaticBoxes(centers, sizes, active, obstacle)


def build_tiled_heightfield(specs, spacing_xy_m=(8.0, 4.0), vertical_scale_m=0.005,
                            margin_xy_m=(0.75, 0.50)):
    """Tile independent support layouts into one global Isaac Gym heightfield."""
    if not specs:
        raise ValueError("at least one terrain spec is required")
    layouts = tuple(generate_support_layout(spec) for spec in specs)
    resolution = layouts[0].spec.resolution_m
    if any(abs(layout.spec.resolution_m - resolution) > 1e-9 for layout in layouts):
        raise ValueError("all tiled layouts must use the same resolution")
    columns = int(np.ceil(np.sqrt(len(layouts))))
    ids = np.arange(len(layouts))
    env_origins = np.stack((
        (ids % columns) * float(spacing_xy_m[0]),
        (ids // columns) * float(spacing_xy_m[1])), axis=-1).astype(np.float32)
    global_origin = np.array([-margin_xy_m[0], -0.5 * layouts[0].spec.width_m
                              - margin_xy_m[1]], dtype=np.float32)
    max_x = max(origin[0] + layout.spec.length_m + margin_xy_m[0]
                for origin, layout in zip(env_origins, layouts))
    max_y = max(origin[1] + 0.5 * layout.spec.width_m + margin_xy_m[1]
                for origin, layout in zip(env_origins, layouts))
    nx = int(np.ceil((max_x - global_origin[0]) / resolution)) + 1
    ny = int(np.ceil((max_y - global_origin[1]) / resolution)) + 1
    pit_depth = max(layout.spec.pit_depth_m for layout in layouts)
    height_m = np.full((nx, ny), -pit_depth, dtype=np.float32)
    for origin, layout in zip(env_origins, layouts):
        ix = int(round((origin[0] - global_origin[0]) / resolution))
        iy = int(round((origin[1] - 0.5 * layout.spec.width_m
                        - global_origin[1]) / resolution))
        local = layout.height_m.T
        height_m[ix:ix + local.shape[0], iy:iy + local.shape[1]] = local
    samples = np.rint(height_m / float(vertical_scale_m)).astype(np.int16)
    return TiledHeightfield(
        samples, global_origin, env_origins, layouts, float(resolution),
        float(vertical_scale_m))


def build_support_segments(layouts, sample_stride=3, segment_length_m=0.22):
    """Approximate centerline support with overlapping, same-height flat boxes."""
    per_layout = []
    for layout in layouts:
        segments = []
        for line in layout.centerlines:
            x = layout.x_m[::sample_stride]
            y = line[::sample_stride]
            dx = np.gradient(x)
            dy = np.gradient(y)
            yaw = np.arctan2(dy, dx)
            segments.extend(zip(x.tolist(), y.tolist(), yaw.tolist()))
        per_layout.append(segments)
    maximum = max(len(segments) for segments in per_layout)
    centers = np.zeros((len(layouts), maximum, 2), dtype=np.float32)
    yaws = np.zeros((len(layouts), maximum), dtype=np.float32)
    active = np.zeros((len(layouts), maximum), dtype=bool)
    for env_id, segments in enumerate(per_layout):
        for segment_id, (x, y, yaw) in enumerate(segments):
            centers[env_id, segment_id] = (x, y)
            yaws[env_id, segment_id] = yaw
            active[env_id, segment_id] = True
    width = float(layouts[0].spec.corridor_width_m)
    if any(abs(layout.spec.corridor_width_m - width) > 1e-9 for layout in layouts):
        raise ValueError("box terrain currently requires one corridor width per batch")
    return SupportSegments(
        centers, yaws, active, float(segment_length_m), width,
        np.array([0.20, 0.0], dtype=np.float32),
        np.array([0.90, 0.90], dtype=np.float32))


def build_support_triangle_mesh(tiled):
    """Create only horizontal support tops, with no slope triangles at holes."""
    support = tiled.height_samples == 0
    cells = (support[:-1, :-1] & support[1:, :-1]
             & support[1:, 1:] & support[:-1, 1:])
    ix, iy = np.nonzero(cells)
    scale = tiled.horizontal_scale_m
    x0 = tiled.origin_xy_m[0] + ix.astype(np.float32) * scale
    y0 = tiled.origin_xy_m[1] + iy.astype(np.float32) * scale
    vertices = np.stack((
        np.stack((x0, y0, np.zeros_like(x0)), axis=-1),
        np.stack((x0 + scale, y0, np.zeros_like(x0)), axis=-1),
        np.stack((x0 + scale, y0 + scale, np.zeros_like(x0)), axis=-1),
        np.stack((x0, y0 + scale, np.zeros_like(x0)), axis=-1),
    ), axis=1).reshape(-1, 3).astype(np.float32)
    base = (4 * np.arange(ix.size, dtype=np.int32))[:, None]
    # +z winding, consistent with Isaac Gym terrain_utils' official converter.
    triangles = np.concatenate((base + np.array([[0, 1, 2]], dtype=np.int32),
                                base + np.array([[0, 2, 3]], dtype=np.int32)), axis=0)
    return SupportTriangleMesh(vertices, triangles)


def build_full_triangle_mesh(tiled):
    """Isaac terrain_utils-compatible shared-grid mesh for support and low floor."""
    heights = tiled.height_samples
    rows, columns = heights.shape
    x = tiled.origin_xy_m[0] + np.arange(rows, dtype=np.float32) * tiled.horizontal_scale_m
    y = tiled.origin_xy_m[1] + np.arange(columns, dtype=np.float32) * tiled.horizontal_scale_m
    yy, xx = np.meshgrid(y, x)
    vertices = np.zeros((rows * columns, 3), dtype=np.float32)
    vertices[:, 0] = xx.reshape(-1)
    vertices[:, 1] = yy.reshape(-1)
    vertices[:, 2] = heights.reshape(-1) * tiled.vertical_scale_m
    triangles = np.empty((2 * (rows - 1) * (columns - 1), 3), dtype=np.uint32)
    for row in range(rows - 1):
        index0 = np.arange(columns - 1, dtype=np.uint32) + row * columns
        index1 = index0 + 1
        index2 = index0 + columns
        index3 = index2 + 1
        start = 2 * row * (columns - 1)
        stop = start + 2 * (columns - 1)
        triangles[start:stop:2, 0] = index0
        triangles[start:stop:2, 1] = index3
        triangles[start:stop:2, 2] = index1
        triangles[start + 1:stop:2, 0] = index0
        triangles[start + 1:stop:2, 1] = index2
        triangles[start + 1:stop:2, 2] = index3
    return SupportTriangleMesh(vertices, triangles)
