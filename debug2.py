import torch
import  torch.nn.functional as F
from my_pvt20_2 import mypvt20_2_small
from my_pvt2520_4 import mypvt2520_4_small

model0 = mypvt20_2_small()
pre_dict = torch.load('work_dirs/my20_s2/my20_300.pth')['model']
model0.load_state_dict(pre_dict)

model1 = mypvt2520_4_small()
pre_dict = torch.load('work_dirs/my20_s2/my20_300_pre.pth')['model']
model1.load_state_dict(pre_dict)

for i in range(10):
    x1 = torch.rand([1, 3, 448, 448])
    x0 = F.avg_pool2d(x1, kernel_size=2)

    x0 = model0.forward_features(x0)
    x1 = model1.forward_features(x1)

    err = x0 - x1
    tmp = err.abs().max()