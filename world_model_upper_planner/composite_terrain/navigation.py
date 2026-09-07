"""Training-only static-map geodesic potential; never a future physics oracle."""
import numpy as np
import torch
from scipy.ndimage import distance_transform_edt,uniform_filter
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra


def route_field(route):
    support=route['support']; h=route['height']; r=route['spec']['resolution']
    ny,nx=h.shape; indices=np.arange(ny*nx).reshape(ny,nx)
    landing=uniform_filter(support.astype(float),size=3,mode='constant')>=5/9
    clearance=distance_transform_edt(support)*r
    penalty=1+.10/(clearance+.05)
    src=[];dst=[];weights=[]
    # Landing graph includes small gaps but does not invent successful dynamics.
    for dy in range(-6,7):
        for dx in range(-6,7):
            length=np.hypot(dx,dy)*r
            if not 0<length<=.30:continue
            a=(slice(max(0,-dy),min(ny,ny-dy)),slice(max(0,-dx),min(nx,nx-dx)))
            b=(slice(max(0,dy),min(ny,ny+dy)),slice(max(0,dx),min(nx,nx+dx)))
            valid=landing[a]&landing[b]&(abs(h[a]-h[b])<=.08)
            src.append(indices[a][valid]);dst.append(indices[b][valid])
            weights.append(length*(penalty[a][valid]+penalty[b][valid])*.5)
    graph=csr_matrix((np.concatenate(weights),(np.concatenate(src),np.concatenate(dst))),
                     shape=(ny*nx,ny*nx))
    gx=round(route['goal'][0]/r);gy=round((route['goal'][1]+route['spec']['width']/2)/r)
    field=dijkstra(graph,directed=False,indices=gy*nx+gx).reshape(ny,nx)
    finite=np.isfinite(field)
    # Nearest reachable values supply a recovery direction from an edge; the
    # true support labels still reject unsupported footholds.
    nearest=distance_transform_edt(~finite,return_distances=False,return_indices=True)
    filled=field[tuple(nearest)]
    filled+=distance_transform_edt(~finite)*r*3
    gy,gx=np.gradient(filled,r,r); norm=np.hypot(gx,gy).clip(1e-6)
    return np.stack((filled,-gx/norm,-gy/norm)).astype(np.float32)


class NavigationLabels:
    def __init__(self,env):
        self.env=env
        self.field=torch.as_tensor(np.stack([route_field(v) for v in env.atlas['routes']]),
                                   device=env.device)

    def sample(self,ids,xy):
        local=xy-self.env.route_origins[ids,:2].view(len(ids),*([1]*(xy.ndim-2)),2)
        local=local.clone();local[...,1]+=self.env.atlas['routes'][0]['spec']['width']/2
        index=torch.round(local/self.env.resolution).long()
        x=index[...,0].clamp(0,self.field.shape[-1]-1)
        y=index[...,1].clamp(0,self.field.shape[-2]-1)
        rows=ids.view(len(ids),*([1]*(xy.ndim-2))).expand_as(x)
        return self.field[rows,:,y,x]
