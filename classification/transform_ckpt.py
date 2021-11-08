def transpvt(tar_dict, src_dict):
    for key in tar_dict.keys():
        # stage 1
        if 'skip' in key:
            tar_dict[key] = tar_dict[key] * 0

        elif 'conf' in key:
            tar_dict[key] = tar_dict[key] * 0

        elif key.startswith('norm') or key.startswith('patch_embed1') or key.startswith('block1') or key.startswith(
                'head'):
            tar_dict[key] = src_dict[key]

        elif key.startswith('block2') or key.startswith('block3') or key.startswith('block4'):
            tmp = key.split('.')
            num = int(tmp[1])
            tmp[1] = f'{num+1}'
            src_key = ''
            for t in tmp:
                src_key = src_key + '.' + t
            src_key = src_key[1:]

            tar_dict[key] = src_dict[src_key]

        elif key.startswith('down_layers'):
            d_stage = int(key[11])
            if 'block' in key:
                src_key = key.replace(f'down_layers{d_stage}.block', f'block{d_stage+1}.{0}')
                tar_dict[key] = src_dict[src_key]
            elif 'conv.' in key:
                tmp = key.split('.')
                src_key = f'patch_embed{d_stage+1}.proj.{tmp[-1]}'
                tar_dict[key] = src_dict[src_key]

            elif 'norm' in key:
                tmp = key.split('.')
                src_key = f'patch_embed{d_stage+1}.norm.{tmp[-1]}'
                tar_dict[key] = src_dict[src_key]
        else:
            print(key)
    return tar_dict


import  torch
from pvt_v2_3h2_density_f import mypvt3h2_density0f_large

# src_file = '/home/wzeng/mycodes/PVT2/classification/work_dirs/pvt_v2_b2.pth'
# tar_file = '/home/wzeng/mycodes/PVT2/classification/work_dirs/tran_pvt_v2_b2.pth'
# model = mypvt3h2_density0f_small()
src_file = '/home/wzeng/mycodes/PVT2/classification/work_dirs/pvt_v2_b4.pth'
tar_file = '/home/wzeng/mycodes/PVT2/classification/work_dirs/tran_pvt_v2_b4.pth'
model = mypvt3h2_density0f_large()


src_dict = torch.load(src_file)

tar_dict = model.state_dict()

tar_dict = transpvt(tar_dict, src_dict)

model.load_state_dict(tar_dict)


torch.save(tar_dict, tar_file)



t = 0