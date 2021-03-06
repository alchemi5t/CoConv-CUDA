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
f = open("in_{}_{}_{}_{}".format(inp_h,inp_w,inp_c,c_out),"w")


for i in inp:
    for j in i:
        for k in j:
            for l in k:
                f.write(str(float(l)) + " ")



f.close()

f = open("weights_{}_{}_{}_{}".format(inp_h,inp_w,inp_c,c_out),"w")
def coconv(maxd):
        out_channels=c_out//4
        convs = []
        for i in range(maxd):
                convs.append(nn.Conv2d(inp_c,out_channels,3,bias=False,dilation=i+1,padding = i+1))

        return convs

convs = coconv(4)


inp = inp.cuda()

for c1 in convs:
        for name, param in c1.named_parameters():
            for j in param:
                for k in j:
                    for l in k:
                        for m in l:
                                f.write(str(float(m)) + " ")
f.close()


f = open("op_{}_{}_{}_{}".format(inp_h,inp_w,inp_c,c_out),"w")
for c1 in convs:
        c1.cuda()
        for i in (c1(inp)):
                for j in i:
                        for k in j:
                                for l in k:
                                        f.write(str(round(float(l),6)) + "\n")

