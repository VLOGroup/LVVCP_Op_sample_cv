import os, sys
import numpy as np
from time import time
import torch
from torch.cuda import Event

c_dir = os.path.split(__file__)[0]
if c_dir not in sys.path:
    sys.path.append (c_dir)

from sample_cv_pytorch import build_sample_cv, coords_grid
from sample_cv_op import sample_cv, SampleCV

from parameterized import parameterized, parameterized_class
import unittest


def get_data(NCHW=[64,64,64,64], dtype=torch.float64, int_vals=False):
    B,C,H,W = NCHW
    # print("PyTorch: initializing Cuda...", end="")
    # tmp = torch.zeros(0).cuda()
    # print(" Done")
    x1 = torch.rand((B,C,H,W), dtype=dtype).cuda()
    x2 = torch.rand((B,C,H,W), dtype=dtype).cuda()
    flow = (torch.rand(B,2,H,W, dtype=dtype).cuda()*1000)
    if int_vals:
        flow = torch.round(flow)
    else:
        flow = torch.round(flow *100) / 100
    return x1,x2, flow

def eval_NHWC(x1,x2,flow, radius=4):
    matcher_nhwc = SampleCV(rx=radius, ry=radius, NCHW_nNHWC=False)
    return matcher_nhwc.forward(x1,x2,flow)
    
def eval_NCHW(x1,x2,flow, radius=4):
    matcher_nchw = SampleCV(rx=radius, ry=radius, NCHW_nNHWC=True)
    return matcher_nchw.forward(x1,x2,flow)

def eval_NHWC_nearest(x1,x2,flow, radius=4):
    matcher_nhwc = SampleCV(rx=radius, ry=radius, NCHW_nNHWC=False, bilinear=False)
    return matcher_nhwc.forward(x1,x2,flow)
    
def eval_NCHW_nearest(x1,x2,flow, radius=4):
    matcher_nchw = SampleCV(rx=radius, ry=radius, NCHW_nNHWC=True, bilinear=False)
    return matcher_nchw.forward(x1,x2,flow)

def eval_PyTorch(x1,x2,flow, radius=4):
    return build_sample_cv(flow, x1, x2, rx=radius, ry=radius, stride=None, align_corners=True)


matchers={
    'NHWC':eval_NHWC,
    'NCHW':eval_NCHW,
    'NHWC_nearest':eval_NHWC_nearest,
    'NCHW_nearest':eval_NCHW_nearest,
    'PyTorch':eval_PyTorch
}


tols={             # atol,  rtol
    torch.float64 : (1e-9,  1e-9),
    torch.float32 : (1e-3,  1e-7),
}

class TestCases(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        print("\nPyTorch: initializing Cuda...", end="", flush=True)
        tmp = torch.zeros(1).cuda()
        tmp[0].item()
        print(" Done\n.", end="", flush=True)
        return
        
    @parameterized.expand([
    # Op1,            Op2     , NCHW,          dtype         , radius, int_val
    ("NHWC",         "PyTorch", [2, 128, 32, 64], torch.float64,  2    ,   False    ),
    ("NCHW",         "PyTorch", [2, 128, 32, 64], torch.float64,  2    ,   False    ),
    ("NCHW",         "NHWC",    [2, 128, 32, 64], torch.float64,  2    ,   False    ),
    ("NHWC",         "PyTorch", [2, 128, 32, 64], torch.float32,  2    ,   False    ),
    ("NCHW",         "PyTorch", [2, 128, 32, 64], torch.float32,  2    ,   False    ),
    ("NHWC_nearest", "PyTorch", [2, 128, 32, 64], torch.float32,  2    ,   True     ),
    ("NCHW_nearest", "PyTorch", [2, 128, 32, 64], torch.float32,  2    ,   True     ),
    ])
    def test_compare_NHWC_pt(self, op1_str, op2_str, NCHW, dtype, radius, int_vals):
        x1,x2, flow = get_data(NCHW, dtype, int_vals=int_vals)
        op1 = matchers[op1_str]
        op2 = matchers[op2_str]
        cv_1 = op1(x1,x2,flow,radius)
        cv_2 = op2(x1,x2,flow,radius)
        # Sampling done - compare results and delta
        delta =   (cv_1-cv_2).abs()
        mean  =   0.5 * (cv_1.abs() + cv_2.abs())
        rdelta = delta/torch.clamp_min(mean, 1e-3)
        rdelta_max = rdelta.max().item()
        delta_max  = delta.max().item() 
        delta_mean = delta.mean().item()
        atol, rtol = tols[dtype] 
        isOk = torch.allclose(cv_1,cv_2,rtol=rtol,atol=atol)
        msg = f"{op1_str}<->{op2_str} NCHW{NCHW}, dtype{dtype}  rdelta_max(max):{rdelta_max:.1e}  delta(max): {delta_max:.1e}   delta(mean) {delta_mean:.1e}     isOk:{isOk}"
        print(msg )

        # DEBUG=False
        # DEBUG=True
        # if DEBUG:
        #     import matplotlib.pyplot as plt
        #     fig,axes = plt.subplots(2,1,num=1,clear=True)
        #     axes[0].imshow( delta.max(-1)[0].max(-1)[0].max(0)[0].cpu().numpy())
        #     axes[0].imshow( delta.max(0)[0].max(0)[0].max(0).cpu().numpy())
        #     plt.show()

        self.assertTrue(isOk, msg)


if __name__ == "__main__":
    unittest.main()


    print("done")