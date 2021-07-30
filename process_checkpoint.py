import torch
src_file = '/home/SENSETIME/zengwang/codes/PVT/work_dirs/my20_s2/my20_300.pth'
tar_file = '/home/SENSETIME/zengwang/codes/PVT/work_dirs/my20_s2/my20_300_pre.pth'

checkpoint = torch.load(src_file)
model = checkpoint['model']
new_model = {}
keys = model.keys()
for key in keys:
    key_new = key
    if 'block2.' in key or 'block3.' in key or 'block4.' in key:
        key_new = key[:7] + str(int(key[7])+1) + key[8:]

    if 'down_layers1.block' in key:
        key_new = key.replace('down_layers1.block', 'block2.0')
    if 'down_layers1.norm' in key:
        key_new = key.replace('down_layers1.norm', 'block2.0.conf_norm')
    if 'down_layers1.conv' in key:
        key_new = key.replace('down_layers1.conv', 'block2.0.pre_conv')
    if 'down_layers1.conf' in key:
        key_new = key.replace('down_layers1.conf', 'block2.0.conf')

    if 'down_layers2.block' in key:
        key_new = key.replace('down_layers2.block', 'block3.0')
    if 'down_layers2.norm' in key:
        key_new = key.replace('down_layers2.norm', 'block3.0.conf_norm')
    if 'down_layers2.conv' in key:
        key_new = key.replace('down_layers2.conv', 'block3.0.pre_conv')
    if 'down_layers2.conf' in key:
        key_new = key.replace('down_layers2.conf', 'block3.0.conf')

    if 'down_layers3.block' in key:
        key_new = key.replace('down_layers3.block', 'block4.0')
    if 'down_layers3.norm' in key:
        key_new = key.replace('down_layers3.norm', 'block4.0.conf_norm')
    if 'down_layers3.conv' in key:
        key_new = key.replace('down_layers3.conv', 'block4.0.pre_conv')
    if 'down_layers3.conf' in key:
        key_new = key.replace('down_layers3.conf', 'block4.0.conf')

    new_model[key_new] = model[key]

del new_model['down_layers1.T']
del new_model['down_layers2.T']
del new_model['down_layers3.T']

new_ck = {}
new_ck['model'] = new_model
torch.save(new_ck, tar_file)


####
# src_file = '/home/SENSETIME/zengwang/codes/PVT/work_dirs/my20_s2/my20_300_pre.pth'
# tar_file = '/home/SENSETIME/zengwang/codes/PVT/work_dirs/my20_s2/my20_300_pre.pth'
#
# checkpoint = torch.load(src_file)
# model = checkpoint['model']
# for key in model.keys():
#     if 'down_layer' in key:
#         del model[key]
#
# new_ck = {}
# new_ck['model'] = model
# torch.save(new_ck, tar_file)