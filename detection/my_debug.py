import torch
from pvt import pvt_small_f4
input = torch.zeros([1, 3, 224, 224])
model = pvt_small_f4()
output = model(input)
print(output)

