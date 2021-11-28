import torch
import random
import numpy as np
import torch.nn as nn
import time
import sys



np.random.seed(42069)
random.seed(42069)
torch.manual_seed(42069)

inp_h = int(sys.argv[1])
inp_w = int(sys.argv[2])
inp_c = int(sys.argv[3])
c_out = int(sys.argv[4])



inp = torch.rand((1,inp_c,inp_h,inp_w))
s = time.time()
inp = inp.cuda()
c1 = nn.Conv2d(inp_c,c_out//4,3,padding = 1,dilation = 1)
c1.cuda()
c2 = nn.Conv2d(inp_c,c_out//4,3,padding = 2,dilation = 2)
c2.cuda()
c3 = nn.Conv2d(inp_c,c_out//4,3,padding = 3,dilation = 3)
c3.cuda()
c4 = nn.Conv2d(inp_c,c_out//4,3,padding = 4,dilation = 4)
c4.cuda()

res = torch.cat((c1(inp),c2(inp),c3(inp),c4(inp)))
print(time.time() - s)
res += 1
if(res):
  c = 99
  """ holding reference count of res to save from GC"""
