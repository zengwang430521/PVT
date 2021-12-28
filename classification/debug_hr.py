from tc_module.hr_tcformer import hrtcformer_w32
import torch

device = torch.device('cuda')
x = torch.rand([1, 3, 224, 224]).to(device)

model = hrtcformer_w32().to(device)

y = model(x)

l = y.sum()
l.backward()

