import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
from pvt import Mlp, Attention, PatchEmbed, Block, PyramidVisionTransformer
from torchvision.models.resnet import resnet50


class PVTCNN(PyramidVisionTransformer):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dims=[64, 128, 256, 512],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1]):

        super().__init__(img_size, patch_size, in_chans, num_classes, embed_dims,
                     num_heads, mlp_ratios, qkv_bias, qk_scale, drop_rate,
                     attn_drop_rate, drop_path_rate, norm_layer,
                     depths, sr_ratios)

        del self.patch_embed1, self.block1, self.pos_embed1
        resnet = resnet50(pretrained=False)
        self.cnn_layers = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            nn.Conv2d(256, embed_dims[0], kernel_size=1)
        )

    def reset_drop_path(self, drop_path_rate):
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(self.depths))]
        cur = 0
        for i in range(self.depths[0]):
            # self.block1[i].drop_path.drop_prob = dpr[cur + i]
            pass

        cur += self.depths[0]
        for i in range(self.depths[1]):
            self.block2[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[1]
        for i in range(self.depths[2]):
            self.block3[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[2]
        for i in range(self.depths[3]):
            self.block4[i].drop_path.drop_prob = dpr[cur + i]

    def forward_features(self, x):
        B = x.shape[0]

        # NO stage 1
        x = self.cnn_layers(x)
        _, _, H, W = x.shape

        # stage 2
        x, (H, W) = self.patch_embed2(x)
        x = x + self.pos_embed2
        x = self.pos_drop2(x)
        for blk in self.block2:
            x = blk(x, H, W)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        # stage 3
        x, (H, W) = self.patch_embed3(x)
        x = x + self.pos_embed3
        x = self.pos_drop3(x)
        for blk in self.block3:
            x = blk(x, H, W)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        # stage 4
        x, (H, W) = self.patch_embed4(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed4
        x = self.pos_drop4(x)
        for blk in self.block4:
            x = blk(x, H, W)

        x = self.norm(x)

        return x[:, 0]


@register_model
def pvtcnn_small(pretrained=False, **kwargs):
    model = PVTCNN(
        patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1], **kwargs)
    model.default_cfg = _cfg()

    return model


# For test
if __name__ == '__main__':
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = pvtcnn_small().to(device)
    empty_input = torch.rand([2, 3, 224, 224], device=device)
    output = model(empty_input)
    print('Finish')

