


import  torch
from myhrpvt import myhrpvt_win_32, myhrpvt_32

# src_file = '/home/wzeng/mycodes/PVT2/classification/work_dirs/pvt_v2_b2.pth'
# tar_file = '/home/wzeng/mycodes/PVT2/classification/work_dirs/tran_pvt_v2_b2.pth'
# model = mypvt3h2_density0f_small()
src_file = '/home/wzeng/mycodes/PVT2/classification/work_dirs/hrt_small.pth'
# tar_file = '/home/wzeng/mycodes/PVT2/classification/work_dirs/tran_hrt_small.pth'
# model = myhrpvt_win_32()
tar_file = '/home/wzeng/mycodes/PVT2/classification/work_dirs/tran_hrpvt_small.pth'
model = myhrpvt_32()

device = torch.device('cuda')
model = model.to(device)


# import time
# input = torch.rand(2,3,112,112).to(device)
# out = model(input)
# loss = out.sum()
# loss.backward()
#
# for tmp in model.named_parameters():
#     name = tmp[0]
#     para = tmp[1]
#     if para.grad is None:
#         print(name)
#
# print('finish')

# for i in range(5):
#     out = model(input)
#     del out
#
# t1 = time.time()
# for i in range(10):
#     out = model(input)
#     del out
# t2 = time.time()
# print((t2-t1) / 10)


src_dict = torch.load(src_file)
src_dict = src_dict['model']
tar_dict = model.state_dict()

own_keys = []
for key in tar_dict.keys():
    if 'transition1' in key:
        own_keys.append(key)
    elif key in src_dict.keys():
        if 'fc' in key and 'weight' in key:
            tar_dict[key] = src_dict[key][:, :, 0, 0]
        else:
            tar_dict[key] = src_dict[key]
    elif 'attn' in key and 'stage' in key:
        if '.q.' in key:
            src_key = key.replace('.attn.q', '.attn.attn.q_proj')
            tar_dict[key] = src_dict[src_key]
        elif '.kv' in key:
            k_key = key.replace('.attn.kv', '.attn.attn.k_proj')
            v_key = key.replace('.attn.kv', '.attn.attn.v_proj')
            k = src_dict[k_key]
            v = src_dict[v_key]
            kv = torch.cat([k, v], dim=0)
            tar_dict[key] = kv
        elif 'proj' in key:
            src_key = key.replace('.attn.proj', '.attn.attn.out_proj')
            tar_dict[key] = src_dict[src_key]
        else:
            own_keys.append(key)
    elif 'dwconv' in key and 'skip' not in key and 'transition' not in key:
        src_key = key.replace('.dwconv.dwconv.', '.dw3x3.')
        tar_dict[key] = src_dict[src_key]
    else:
        own_keys.append(key)

print(len(tar_dict.keys()))
print(len(own_keys))
model.load_state_dict(tar_dict)
torch.save(tar_dict, tar_file)



'''hrpvt_swin'''
def tran_swin(tar_dict, src_dict):
    share = []
    own = []
    for key in tar_dict.keys():
        if key in src_dict.keys():
            tar_dict[key] = src_dict[key]
            share.append(key)
        elif 'skip' in key:
            tar_dict[key] = tar_dict[key] * 0
        else:
            own.append(key)
            # print(key)

    model.load_state_dict(tar_dict)


    torch.save(tar_dict, tar_file)



t = 0
