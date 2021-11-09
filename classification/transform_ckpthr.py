


import  torch
from myhrpvt import myhrpvt_win_32

# src_file = '/home/wzeng/mycodes/PVT2/classification/work_dirs/pvt_v2_b2.pth'
# tar_file = '/home/wzeng/mycodes/PVT2/classification/work_dirs/tran_pvt_v2_b2.pth'
# model = mypvt3h2_density0f_small()
src_file = '/home/wzeng/mycodes/PVT2/classification/work_dirs/hrt_small.pth'
tar_file = '/home/wzeng/mycodes/PVT2/classification/work_dirs/tran_hrt_small.pth'
model = myhrpvt_win_32()

device = torch.device('cuda')
model = model.to(device)
import time
input = torch.rand(2,3,224,224).to(device)
for i in range(5):
    out = model(input)
    del out

t1 = time.time()
for i in range(10):
    out = model(input)
    del out
t2 = time.time()
print((t2-t1) / 10)


# src_dict = torch.load(src_file)
# src_dict = src_dict['model']
# tar_dict = model.state_dict()
#
# share = []
# own = []
# for key in tar_dict.keys():
#     if key in src_dict.keys():
#         tar_dict[key] = src_dict[key]
#         share.append(key)
#     elif 'skip' in key:
#         tar_dict[key] = tar_dict[key] * 0
#     else:
#         own.append(key)
#         # print(key)
#
# model.load_state_dict(tar_dict)
#
#
# torch.save(tar_dict, tar_file)
#
#
#
# t = 0
