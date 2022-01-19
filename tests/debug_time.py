# This file is a simple debugging environment for timing analysis
# In general the timing of the different implementation variants depends on the flow offsets
# The timing largely depends on image size and also on variety of flow results
# In cases with mainly identical and low flow values NCHW versions seem to be faster
# In cases with large and varying flow offsets NHWC version 


import subprocess as sp
import numpy as np
from time import time
import torch

from torch.cuda import Event

from sample_cv_op import sample_cv, SampleCV

def get_GPU_name():
    cmd = "nvidia-smi --query-gpu=gpu_name --format=csv"
    return  sp.check_output(cmd.split()).decode('ascii').split('\n')[1]



class TimingAnalyzer():
    def __init__(self, size, dtype=torch.float32, max_flow=10):
        self.initialize_torch_cuda()
        self.setup_data(size, dtype=torch.float32, max_flow=10)


    def setup_data(self, size, radius=4, dtype=torch.float32, max_flow=10):
        print(f" Setting up data with: size={size}, radius={radius}, dtype={dtype}, max_flow={max_flow}", end="", flush=True)

        self._is_setup = True
        [B,C,H,W] = size
        self.x1 = torch.randn((B,C,H,W), dtype=dtype).cuda()
        self.x2 = torch.randn((B,C,H,W), dtype=dtype).cuda()
        self.flow = torch.rand(B,2,H,W, dtype=dtype).cuda()*max_flow

        self._matchers = {}
        self._matchers["NHWC"] = SampleCV(rx=radius, ry=radius, NCHW_nNHWC=False), (self.x1, self.x2, self.flow)
        self._matchers["NCHW"] = SampleCV(rx=radius, ry=radius, NCHW_nNHWC=True),  (self.x1, self.x2, self.flow)
        self._matchers["NHWC_nearest"] = SampleCV(rx=radius, ry=radius, NCHW_nNHWC=False, bilinear=False), (self.x1, self.x2, self.flow)
        self._matchers["NCHW_nearest"] = SampleCV(rx=radius, ry=radius, NCHW_nNHWC=True, bilinear=False), (self.x1, self.x2, self.flow)

        print("  done", flush=True)
        
    def get_matcher_names(self):
        return self._matchers.keys()

    def call_matcher(self, name):
        matcher = self._matchers[name][0]
        values = self._matchers[name][1]
        return matcher(*values)

    def initialize_torch_cuda(self):
        print("PyTorch: initializing Cuda...", end="", flush=True)
        tmp = torch.zeros(0).cuda()
        print(" Done", flush=True)

    def run_timing(self, name, itr=20):
        # matcher = self.get_matcher_call(name)
        print(f" - initial run of matcher ", end="", flush=True)
        self.call_matcher(name) # initial run to avoid setup timing differences
        print(f" done", flush=True)
        
        # use cuda events for accurate timing measurements
        start = Event(enable_timing=True)
        stop = Event(enable_timing=True)
        print(f" - repetitive runs of matcher ", end="", flush=True)
        start.record()
        for i in range(itr):
            self.call_matcher(name) 
        print(f" done", flush=True)
        stop.record()
        torch.cuda.synchronize()
        dt = start.elapsed_time(stop) / itr
        # print(f"{name} dt: {dt}")
        return dt


if __name__ == "__main__":
    size = [64, 64, 64, 64]
    # size = [1, 512, 480, 854]

    dtype = torch.float32

    print("----")
    print(get_GPU_name())

    timing_analyzer = TimingAnalyzer( size, dtype)
    print("----")
    print(f"Evaluationg size={size}", flush=True)
    for matcher_name in timing_analyzer.get_matcher_names():
        print(f"Starting timing analysis of {matcher_name} :")
        dt = timing_analyzer.run_timing(matcher_name)
        print(f"   dt={dt:10.1f} ms / call  for {matcher_name}")
    itr=20   # repetitions for the timing test

    print("done")


################################
# ----
# NVIDIA TITAN RTX
# PyTorch: initializing Cuda... Done
#  Setting up data with: size=[64, 64, 64, 64], radius=4, dtype=torch.float32, max_flow=10  done
# ----
# Evaluationg size=[64, 64, 64, 64]
# Starting timing analysis of NHWC :
#  - initial run of matcher  done
#  - repetitive runs of matcher  done
#    dt=    310.31   for NHWC
# Starting timing analysis of NCHW :
#  - initial run of matcher  done
#  - repetitive runs of matcher  done
#    dt=    294.42   for NCHW
# Starting timing analysis of NHWC_nearest :
#  - initial run of matcher  done
#  - repetitive runs of matcher  done
#    dt=    188.45   for NHWC_nearest
# Starting timing analysis of NCHW_nearest :
#  - initial run of matcher  done
#  - repetitive runs of matcher  done
#    dt=     97.81   for NCHW_nearest
# done


################################
# ----
# NVIDIA TITAN RTX
# PyTorch: initializing Cuda... Done
#  Setting up data with: size=[1, 512, 480, 854], radius=4, dtype=torch.float32, max_flow=10  done
# ----
# Evaluationg size=[1, 512, 480, 854]
# Starting timing analysis of NHWC :
#  - initial run of matcher  done
#  - repetitive runs of matcher  done
#    dt=     195.5 ms / call  for NHWC
# Starting timing analysis of NCHW :
#  - initial run of matcher  done
#  - repetitive runs of matcher  done
#    dt=     468.5 ms / call  for NCHW
# Starting timing analysis of NHWC_nearest :
#  - initial run of matcher  done
#  - repetitive runs of matcher  done
#    dt=     107.9 ms / call  for NHWC_nearest
# Starting timing analysis of NCHW_nearest :
#  - initial run of matcher  done
#  - repetitive runs of matcher  done
#    dt=     749.9 ms / call  for NCHW_nearest
# done