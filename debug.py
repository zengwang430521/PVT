import torch
import cv2
from my_pvt23 import map2token, token2map, get_loc
# from my_pvt20 import map2token, token2map, get_loc

import matplotlib.pyplot as plt




img_file = '/home/SENSETIME/zengwang/codes/mmpose/tests/data/coco/000000196141.jpg'
img = cv2.imread(img_file)[:, :, ::-1].copy()
l, c, n = 1, 3, 1

img = torch.tensor(img).float() / 255.0
img = img.permute(2, 0, 1).unsqueeze(0)
plt.subplot(l, c, n); n += 1
plt.imshow(img[0].permute(1, 2, 0))

x = img
B, C, H, W = x.shape
x = x.flatten(2).permute(0, 2, 1)
x, loc, N_grid = get_loc(x, H, W, grid_stride=1000)
x_map = token2map(x, loc, [H, W], 9, 2)
plt.subplot(l, c, n); n += 1
plt.imshow(x_map[0].permute(1, 2, 0))

x2 = map2token(x_map, loc)
loc2 = loc

# mask = torch.rand(x2.shape[1]) > 0.5
# pos = torch.arange(x.shape[1], dtype=torch.long, device=x.device)
# pos2 = torch.masked_select(pos, mask)
# x2 = torch.index_select(x, 1, pos2)
# loc2 = torch.index_select(loc, 1, pos2)

x2_o = x2
for i in range(100):
    err = x2 - x2_o
    x2_map = token2map(x2, loc2, [H*2, W*2], 3, 2)
    plt.subplot(l, c, n)
    plt.imshow(x2_map[0].permute(1, 2, 0))
    x2 = map2token(x2_map, loc2)

print('finish')