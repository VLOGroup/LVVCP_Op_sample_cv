import sys

# import torch.autograd as autograd
# from torch.autograd.function import once_differentiable
# from torch.nn import Module
import torch
from os.path import join, split, isfile

from typing import Tuple,List


# Torch Script compatible way: load via load_library into torch.ops.add_one_op.add_one_forward
# Will work inside the installed package only
def get_lib_path():
    """ returns the path to the binary library file"""
    if sys.platform.startswith("win"):
        lib_name = 'samplecv_op.dll'
    elif sys.platform.startswith("linux"):
        lib_name = 'libsamplecv_op.so'

    lib_path = join(get_module_path(), lib_name)
    return lib_path

def get_include_path():
    """ returns the path to where the module is installed"""
    include_path = join(get_module_path(), "include")
    return include_path

def get_module_path():
    """ returns the path to where the module is installed"""
    module_path = split(__file__)[0]
    return module_path

lib_path = get_lib_path()
if not isfile(lib_path):
    if isfile( join(split(__file__)[0], 'source_dir_indicator.txt')  ):
        raise RuntimeError(f"Could not find {lib_path}\nThe library cannot be imported from within git repository directory! Change the directory!")
    else:
        raise RuntimeError(f"Could not find {lib_path}")
torch.ops.load_library(lib_path)
    
############################################################################################################################
############################################################################################################################  

def sample_cv(f1:torch.Tensor, f2:torch.Tensor, ofs:torch.Tensor, rx:int=4, ry:int=4, NCHW_nNHWC:bool=True, bilinear:bool=True):
    """
    sample_cv (f1:torch.Tensor, f2:torch.Tensor, ofs:torch.Tensor, rx:int=4, ry:int=4, NCHW_nNHWC:bool=True):
    ---

    Custom cost volume sampling function
    Ofs is an integer tensor specifying an offset of the search grid of size rx,ry for each pixel

    NCHW_nNHWC: (T/F) specifies if NCHW or NHWC format is used internally - however input data is always expected to be NCHW!
    bilinear: (T/F) specifices if bilinear or nearest neighboor sampling is used
    """

    return torch.ops.sample_cv_op.sample_cv_forward(f1=f1, f2=f2, ofs=ofs, rx=rx, ry=ry, NCHW_nNHWC=NCHW_nNHWC, bilinear=bilinear)

class SampleCV(torch.nn.Module):
    def __init__(self, rx:int=4, ry:int=4, NCHW_nNHWC:bool=True, bilinear:bool=True):
        """
        Custom cost volume sampling function
        
        NCHW_nNHWC: (T/F) specifies if NCHW or NHWC format is used internally - however input data is always expected to be NCHW!
        bilinear: (T/F) specifices if bilinear or nearest neighboor sampling is used
        """
        super(SampleCV, self).__init__()
        self.rx: int = rx
        self.ry: int = ry
        self.NCHW_nNHWC: bool = NCHW_nNHWC
        self.bilinear: bool = bilinear

    def forward(self, f1:torch.Tensor, f2:torch.Tensor, ofs:torch.Tensor):
        """ Compute CV
            Ofs is an integer tensor specifying an offset of the search grid of size rx,ry for each pixel
        """

        return sample_cv(f1=f1, f2=f2, ofs=ofs, rx=self.rx, ry=self.ry, NCHW_nNHWC=self.NCHW_nNHWC, bilinear=self.bilinear)

    def extra_repr(self):
        return f"rx={self.rx},ry={self.ry},NCHW_nNHWC='{self.NCHW_nNHWC}',bilinear='{self.bilinear}'"
