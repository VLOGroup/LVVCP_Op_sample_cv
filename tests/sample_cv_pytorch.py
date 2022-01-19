import numpy as np
import imageio
import torch
import torch.nn.functional as F
from typing import List


def bilinear_sampler(img, coords, mode:str='bilinear'):
    """ Wrapper for grid_sample, uses pixel coordinates 
        Bilinear Sampler wrapper from RAFT 
    """
    assert coords.shape[-1] == 2
    H, W = img.shape[-2:]
    xgrid, ygrid = coords.split([1,1], dim=-1)

    xgrid = 2*xgrid/(W-1) - 1
    ygrid = 2*ygrid/(H-1) - 1

    grid = torch.cat([xgrid, ygrid], dim=-1)
    img = F.grid_sample(img, grid, align_corners=True)

    return img

def bilinear_sampler_with_mask(img, coords, mode:str='bilinear'):
    """ Wrapper for grid_sample, uses pixel coordinates
        Bilinear Sampler wrapper from RAFT
    """
    assert coords.shape[-1] == 2
    H, W = img.shape[-2:]
    xgrid, ygrid = coords.split([1,1], dim=-1)
    mask_valid = (xgrid > 0) & (ygrid > 0) & (xgrid < (W-1)) & (ygrid < (H-1))

    xgrid = 2*xgrid/(W-1) - 1
    ygrid = 2*ygrid/(H-1) - 1

    grid = torch.cat([xgrid, ygrid], dim=-1)
    img = F.grid_sample(img, grid, align_corners=True)

    return img, mask_valid.float()

def coords_grid(batch:int, ht:int, wd:int) -> torch.Tensor:
    """ Simple Meshgrid wrappter from RAFT 
    """
    coords = torch.meshgrid(torch.arange(ht), torch.arange(wd))
    coords = torch.stack(coords[::-1], dim=0).float()
    return coords[None].repeat(batch, 1, 1, 1)

def sample_with_grid_HWdydx(data:torch.Tensor, coords_ofs_xy:torch.Tensor, rx:int=8, ry:int=8, stride:List[int]=None, mask:bool=False, mode:str='bilinear'):
    """
    Get a local Cost Volume on the finest resolution
    """
    assert coords_ofs_xy.shape[1] == 2, f"coords_ofs_xy must be in N2HW format"
    assert coords_ofs_xy.shape[0] == data.shape[0]
    N, _, H1, W1 = coords_ofs_xy.shape
    C,H2,W2 = data.shape[1:4]
    device = coords_ofs_xy.device
    dW2 = 2*rx+1
    dH2 = 2*ry+1
    
    dx = torch.linspace(-rx, rx, dW2, device=device)
    dy = torch.linspace(-ry, ry, dH2, device=device)
    if stride is not None:
        dx = dx*stride
        dy = dy*stride
    mg_y, mg_x = torch.meshgrid(dy, dx)

    centroid_HW = coords_ofs_xy.permute(0, 2, 3, 1)  # [N,2,H1,W1] => [N,H1,W1,2]
    centroid_HW = centroid_HW.reshape(N,1,1,H1,W1,2) # [N,1,1,H1,W1,2] # x,y
    delta_mg_dydx = torch.stack([mg_x, mg_y],dim=-1)
    delta_mg_dydx = delta_mg_dydx.reshape(1, dH2, dW2, 1, 1,2)
    coords_dydxHW = centroid_HW + delta_mg_dydx # [N,dH2,dW2,H1,W1,2]

    data_dup = data[:,None,None,].repeat(1,dW2,dH2,1,1,1)  # [N,dH2,dW2,C,H1,W1]

    coords_dydxHW = coords_dydxHW.reshape(N*dH2*dW2,  H1,W1,2) #   [N*dH2*dW2,H1,W1,2]
    data_dup      = data_dup     .reshape(N*dH2*dW2,C,H1,W1) #   [N*dH2*dW2,H1,W1,2]
    coords_dydxHW = coords_dydxHW.contiguous()
    data_dup      = data_dup.contiguous()

    if mask:
        data_grid_dydxHWd, mask_valid_dydxHWd = bilinear_sampler_with_mask(data_dup, coords_dydxHW)  #[N*dH2*dW2,C,H1,W1] => [N*dH2*dW2,C,H1,W1]
        data_grid_dydxHWd = data_grid_dydxHWd.reshape(N, dH2, dW2, C, H1, W1)  #[N*dH2*dW2,C,H1,W1] => [N, dH2, dW2, C, H1, W1]
        mask_valid_dydxHWd = mask_valid_dydxHWd.reshape(N, dH2, dW2, 1, H1, W1) 
        data_grid_HWdydx = data_grid_dydxHWd.permute(0,3,4,5,1,2) # [N, dH2, dW2, C, H1, W1] = > [N, C, H1, W1, dH2, dW2]
        mask_valid_HWdydx = mask_valid_dydxHWd.permute(0,3,4,5,1,2) # [N, dH2, dW2, C, H1, W1] = > [N, C, H1, W1, dH2, dW2]
        return data_grid_HWdydx, mask_valid_HWdydx
    else:
        data_grid_dydxHWd = bilinear_sampler(data_dup, coords_dydxHW)  #[N*dH2*dW2,C,H1,W1] => [N*dH2*dW2,C,H1,W1]
        data_grid_dydxHWd = data_grid_dydxHWd.reshape(N, dH2, dW2, C, H1, W1)  #[N*dH2*dW2,C,H1,W1] => [N, dH2, dW2, C, H1, W1]
        data_grid_HWdydx = data_grid_dydxHWd.permute(0,3,4,5,1,2) # [N, dH2, dW2, C, H1, W1] = > [N, C, H1, W1, dH2, dW2]
        return data_grid_HWdydx


def build_sample_cv(flow:torch.Tensor, f1:torch.Tensor, f2:torch.Tensor, rx:int, ry:int, stride:List[int]=None, align_corners:bool=True, mode:str='bilinear'):
    """ A simplified pytorch version of a differential cost-volume
        Replicates f2 features over a grid and uses batch matmul to compute cosine distance
    """
    N,C,H1,W1 =  f1.shape
    c_coords0 = coords_grid(N, H1, W1).to(device=f1.device)
    c_coords1 = c_coords0 + flow    

    f2_all_cands = sample_with_grid_HWdydx(f2,  c_coords1,ry=ry,rx=rx, stride=stride, mode=mode)  # N C H W ry rx
    f2_all_cands = f2_all_cands.permute(0,2,3,4,5,1).reshape(N*H1*W1,(2*ry+1)*(2*rx+1),C) #[N*H*W, dy*dx ,C]
    f1_comp = f1.permute(0,2,3,1).reshape(N*H1*W1,C,1)                                    #[N*H*W,   C   ,1]
    cv = torch.bmm(f2_all_cands,f1_comp).reshape(N,H1, W1, (2*ry+1), (2*rx+1))           # [N,H,W,dy,dx]
    return cv # [N,H,W,dy,dx]
