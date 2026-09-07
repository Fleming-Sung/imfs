"""Seeded continuous terrain, with traversable transitions between regions.

All arrays use [y,x]. Metadata describes generation/diagnosis only, not planner
observations. Actual geometry includes obstacles in the same height surface.
"""
from dataclasses import dataclass, asdict
import numpy as np

FAMILIES = ("ramp", "stairs", "stones", "bridge", "slalom", "rough", "curved_bridge")


@dataclass(frozen=True)
class CompositeSpec:
    seed: int = 0
    difficulty: float = 0.3
    resolution: float = 0.05
    width: float = 4.8
    segment_length: float = 2.4
    # Permutation varies per route; every episode traverses all families.
    families: tuple = FAMILIES


def generate_route(spec):
    rng = np.random.default_rng(spec.seed)
    d = float(spec.difficulty)
    if not 0 <= d <= 1:
        raise ValueError("difficulty must be in [0,1]")
    families = list(rng.permutation(spec.families))
    length = 2.0 + len(families) * spec.segment_length
    x = np.arange(0, length + spec.resolution/2, spec.resolution, dtype=np.float32)
    y = np.arange(-spec.width/2, spec.width/2 + spec.resolution/2,
                  spec.resolution, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)
    floor = -0.8
    height = np.full(xx.shape, floor, dtype=np.float32)
    support = np.zeros(xx.shape, dtype=bool)
    region = np.full(xx.shape, -1, dtype=np.int16)
    route_y = np.zeros(len(x), dtype=np.float32)
    center = 0.0; level = 0.0
    segments = []

    def surface(mask, z):
        height[mask] = np.broadcast_to(z, height.shape)[mask]
        support[mask] = True

    surface((xx < 1.0) & (abs(yy) <= .65), 0.0)
    for i, kind in enumerate(families):
        start = 1.0 + i * spec.segment_length; end = start + spec.segment_length
        t = np.clip((x-start)/spec.segment_length,0,1)
        next_center = float(rng.uniform(-.55,.55))
        centerline = center + (next_center-center)*(3*t*t-2*t*t*t)
        if kind == "curved_bridge":
            centerline += (.12+.18*d)*np.sin(2*np.pi*t)*np.sin(np.pi*t)
        if kind == "slalom":
            centerline += (.25+.18*d)*np.sin(4*np.pi*t)*np.sin(np.pi*t)**2
        col = (x>=start)&(x<end)
        route_y[col] = centerline[col]
        along = (xx>=start)&(xx<end)
        lateral = yy-centerline[None]
        # Flat landing areas at each boundary ensure valid cross-family joins.
        gate = along & ((xx < start+.3)|(xx > end-.3)) & (abs(lateral)<.65)
        dz = 0.0
        profile = np.full(len(x), level, dtype=np.float32)
        active_t = np.clip((x-start-.35)/(spec.segment_length-.7), 0, 1)
        width = float(rng.uniform(.90,1.20))
        if kind == "ramp":
            dz = float(rng.choice([-1,1])*(.04+.12*d))
            profile += dz*active_t
        elif kind == "stairs":
            step_height = (.015+.04*d)*float(rng.choice([-1,1]))
            profile += step_height*np.floor(active_t*5 + 1e-5)
            dz = 5*step_height
        elif kind in ("bridge","curved_bridge"):
            width = float(rng.uniform(.58-.14*d,.70-.12*d))
        elif kind == "rough":
            # Low-pass continuous bumps, zero at region boundaries.
            profile += (.008+.025*d)*np.sin(6*np.pi*active_t)*np.sin(np.pi*active_t)**2
        if kind == "stones":
            # Two offset rows of variable-height supports; no hidden flat floor.
            gate = along & ((xx<start+.45)|(xx>end-.45)) & (abs(lateral)<.65)
            surface(gate, level)
            pitch = .28+.04*d; pad_x = pitch-(.02+.035*d); pad_y=.27-.045*d
            for j,cx in enumerate(np.arange(start+.40,end-.25,pitch)):
                cy=float(np.interp(cx,x,centerline))
                for side in [-1,1]:
                    z = level+float(rng.uniform(-.012-.025*d,.012+.025*d))
                    mask=along & (abs(xx-cx)<=pad_x/2) & (abs(yy-cy-side*.145)<=pad_y/2)
                    surface(mask,z)
        else:
            surface(along & (abs(lateral)<width/2),profile[None])
            surface(gate,profile[None])
        if kind == "slalom":
            for j,cx in enumerate(np.linspace(start+.6,end-.6,3)):
                cy=float(np.interp(cx,x,centerline))
                # Actual obstacles offset from a guaranteed free corridor;
                # an alternating route bend is required to pass each obstacle.
                oy=cy+(-1 if j%2 else 1)*(.42+.06*d)
                mask=along & (abs(xx-cx)<.17) & (abs(yy-oy)<.18)
                height[mask]=level+float(rng.uniform(.35,.75)); support[mask]=False
        region[along]=i
        segments.append(dict(index=i, kind=kind, start_x=start, end_x=end,
            start_y=center,end_y=next_center,start_z=level,end_z=level+dz,
            width_m=width))
        center=next_center; level+=dz
    surface((xx>=length-1)&(abs(yy-center)<.65),level)
    route_y[x>=length-1]=center
    return dict(spec=asdict(spec),height=height,support=support,region=region,x=x,y=y,
        route_y=route_y,segments=segments,start=np.array([.3,0,0],np.float32),
        goal=np.array([length-.35,center,level],np.float32))


def build_atlas(specs):
    """One global heightfield containing multiple continuous composite routes."""
    routes=[generate_route(s) for s in specs]
    if not routes: raise ValueError("empty atlas")
    r=specs[0].resolution
    shape=routes[0]["height"].shape
    if any(v["height"].shape!=shape or s.resolution!=r for s,v in zip(specs,routes)):
        raise ValueError("atlas route dimensions must agree")
    ny,nx=shape; gap=int(round(.8/r)); columns=max(1,int(np.ceil(np.sqrt(len(routes)))))
    rows=int(np.ceil(len(routes)/columns))
    atlas=np.full((rows*(ny+gap),columns*(nx+gap)),-.8,np.float32)
    origins=[]
    for i,v in enumerate(routes):
        row,col=divmod(i,columns); y0=row*(ny+gap); x0=col*(nx+gap)
        atlas[y0:y0+ny,x0:x0+nx]=v["height"]
        origins.append([x0*r,y0*r+specs[i].width/2,0])
    return dict(height=atlas,origins=np.asarray(origins,np.float32),routes=routes,resolution=r)
