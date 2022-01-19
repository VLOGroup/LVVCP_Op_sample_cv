import sys, os
c_dir = os.path.split(__file__)[0]
sys.path.append(c_dir + "/..")

import numpy as np
import imageio
from imageio import imread, imsave, get_writer
import matplotlib.pyplot as plt

import torch
import torchvision.models as models
from torch.nn import functional as F

from sample_cv_op import sample_cv, SampleCV




rx = 2
ry = 3

fx,fy = 0,0
N,C,H,W = [1,128,8,2]
I    = torch.randn([N,C,H+fy,W]).cuda()
flow = (torch.randn([N,2,H,W])*0 + torch.tensor([fx,fy])[None,:,None,None]).cuda()
flow = flow.contiguous()
I1f = I[:,:,0:H,:].contiguous()
I2f = I[:,:,2:H+2  ,:].contiguous()

I1f = torch.ones_like(I1f)
I2f = torch.zeros([N,C,H,W]).cuda().contiguous()
I2f[:,0,...] =  torch.arange(N*H*W).reshape([N,1,H,W]).cuda()



# matcher = LocalDist(rx=cv_range, ry=cv_range, NCHW_nNWHC=True)
matcher = SampleCV(rx=rx, ry=ry, NCHW_nNHWC=False)
cv = matcher.forward(I1f, I2f, flow) # [N,H,W,CVy,CVx]

print(cv[0,0,0])


print("done")
