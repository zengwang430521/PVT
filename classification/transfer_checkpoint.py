import torch
import copy
from tc_module.tcformer import tcformer_small
# src_file = 'work_dirs/my3h2_density0/checkpoint.pth'
# out_file = 'work_dirs/my3h2_density0/checkpoint_tcformer.pth'
src_file = 'work_dirs/my3h2_density0f_tiny_16/checkpoint.pth'
out_file = 'work_dirs/my3h2_density0f_tiny_16/checkpoint_tcformer.pth'


model = tcformer_small()
model_dict = model.state_dict()
src_dict = torch.load(src_file, map_location='cpu')
src_dict = src_dict['model']

src_left = copy.copy(src_dict)
model_left = copy.copy(model_dict)

out_dict = {}
for src_key in src_dict:
    if src_key in model_dict.keys():
        model_key = src_key
    elif 'down_layers' in src_key:
        model_key = src_key.replace('down_layers', 'ctm')
    else:
        model_key = src_key

    if model_key in model_dict.keys():
        out_dict[model_key] = src_dict[src_key]
        src_left.pop(src_key)
        model_left.pop(model_key)

tmp = model.load_state_dict(out_dict)
print(tmp)

torch.save(out_dict, out_file)