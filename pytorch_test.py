import torch
import random
import numpy as np
import torch.nn as nn
import time





np.random.seed(42069)
random.seed(42069)
torch.manual_seed(42069)

inp = torch.rand((1,100,100,100))
torch.rand((1)).cuda()
s = time.time()
#inp = inp.cuda()
c1 = nn.Conv2d(100,100,3,padding = 1,dilation = 1)
#c1.cuda()
c2 = nn.Conv2d(100,100,3,padding = 2,dilation = 2)
#c2.cuda()
c3 = nn.Conv2d(100,100,3,padding = 3,dilation = 3)
#c3.cuda()
c4 = nn.Conv2d(100,100,3,padding = 4,dilation = 4)
#c4.cuda()

#res = torch.cat((c1(inp),c2(inp),c3(inp),c4(inp)))
print(time.time() - s)
