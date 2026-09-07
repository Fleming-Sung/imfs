"""Write a reproducible map atlas and static research figures."""
import argparse
import json
from pathlib import Path
import numpy as np
from .maps import CompositeSpec, build_atlas


def main():
    p=argparse.ArgumentParser(); p.add_argument("--output",type=Path,required=True)
    p.add_argument("--seed",type=int,default=9100); p.add_argument("--routes",type=int,default=9)
    args=p.parse_args(); args.output.mkdir(parents=True,exist_ok=True)
    specs=[CompositeSpec(seed=args.seed+i,difficulty=.15+.7*(i%3)/2) for i in range(args.routes)]
    atlas=build_atlas(specs)
    np.savez_compressed(args.output/"atlas.npz",height=atlas["height"],origins=atlas["origins"],
                        resolution=atlas["resolution"])
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(len(specs),1,figsize=(16,2.3*len(specs)),squeeze=False)
    manifest=[]
    for ax,route in zip(axes[:,0],atlas["routes"]):
        x,y=route["x"],route["y"]
        im=ax.imshow(route["height"],origin="lower",extent=(x[0],x[-1],y[0],y[-1]),
                     aspect="equal",cmap="terrain",vmin=-.8,vmax=.8)
        ax.plot(x,route["route_y"],"r--",lw=.7,alpha=.7)
        for s in route["segments"]:
            ax.axvline(s["start_x"],color="k",lw=.4)
            ax.text((s["start_x"]+s["end_x"])/2,1.8,s["kind"],ha="center",fontsize=8)
        ax.set_title(f"seed={route['spec']['seed']} difficulty={route['spec']['difficulty']:.2f}")
        ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
        manifest.append(dict(spec=route["spec"],segments=route["segments"],
                             start=route["start"].tolist(),goal=route["goal"].tolist(),
                             height_range_m=[float(route["height"].min()),float(route["height"].max())]))
    fig.tight_layout(); fig.savefig(args.output/"routes.png",dpi=140); plt.close(fig)
    (args.output/"manifest.json").write_text(json.dumps(manifest,indent=2))
    print(json.dumps(dict(routes=len(specs),shape=atlas["height"].shape,
                         total_segments=sum(len(v["segments"]) for v in atlas["routes"]))))

if __name__=="__main__": main()
