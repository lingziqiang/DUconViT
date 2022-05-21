from .mctrans.models.builder import build_network
import ml_collections
from .mctrans.models.encoders.vgg import VGG
from .mctrans.models.centers.mctrans import MCTrans
from .mctrans.models.decoders.unet_decoder import UNetDecoder
# from .mctrans.models.heads import basic_seg_head
# from .mctrans.models.heads import mctrans_aux_head
import torch.nn as nn

import torch
from torch.nn.init import normal_
from mmcv.cnn import ConvModule
from .mctrans.models.builder import CENTERS
from .mctrans.models.ops.modules import MSDeformAttn
from .mctrans.models.trans.transformer import DSALayer, DSA
from .mctrans.models.trans.utils import build_position_encoding, NestedTensor

from .mctrans.models.builder import DECODERS

class MCTrans_all(nn.Module):
    def __init__(self, config, num_classes=2, kernel_size=1):
        super(MCTrans_all, self).__init__()
        config['encoder'].pop('type')
        config['center'].pop('type')
        config['decoder'].pop('type')

        self.encoder = VGG(**config['encoder'])
        self.center = MCTrans(**config['center'])
        self.decoder = UNetDecoder(**config['decoder'])
        self.seg_head = nn.Conv2d(in_channels=config['decoder']['in_channels'][0], out_channels=num_classes, kernel_size=1)

    def forward(self, x):
        o1 = self.encoder(x)
        o2 = self.center(o1)
        o3 = self.decoder(o2)
        out = self.seg_head(o3)

        return out








# def conv3x3(in_planes, out_planes, dilation=1):
#     return nn.Conv2d(
#         in_planes,
#         out_planes,
#         kernel_size=3,
#         padding=dilation,
#         dilation=dilation)
#
#
# def conv_bn_relu(in_channels, out_channels, kernel_size=3, padding=1, stride=1):
#     return nn.Sequential(nn.Conv2d(in_channels,
#                                    out_channels,
#                                    kernel_size=kernel_size,
#                                    padding=padding,
#                                    stride=stride),
#                          nn.BatchNorm2d(out_channels),
#                          nn.ReLU(inplace=True))
#
#
# def deconv_bn_relu(in_channels, out_channels, kernel_size=3, padding=1, stride=2):
#     return nn.Sequential(nn.ConvTranspose2d(in_channels,
#                                             out_channels,
#                                             kernel_size=kernel_size,
#                                             padding=padding,
#                                             stride=stride,
#                                             output_padding=1),
#                          nn.BatchNorm2d(out_channels),
#                          nn.ReLU(inplace=True))
#
#
# def make_vgg_layer(inplanes,
#                    planes,
#                    num_blocks,
#                    dilation=1,
#                    with_bn=False,
#                    down_sample=False,
#                    ceil_mode=False):
#     layers = []
#     if down_sample:
#         layers.append(nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=ceil_mode))
#     for _ in range(num_blocks):
#         layers.append(conv3x3(inplanes, planes, dilation))
#         if with_bn:
#             layers.append(nn.BatchNorm2d(planes))
#         layers.append(nn.ReLU(inplace=True))
#         inplanes = planes
#     return nn.Sequential(*layers)
#
# class VGG(nn.Module):
#     def __init__(self, in_channel=1, depth=5, init_channels=16, num_blocks=2):
#         super(VGG, self).__init__()
#         filters = [(2 ** i) * init_channels for i in range(depth)]
#         self.out_channels = filters.copy()
#
#         filters.insert(0, in_channel)
#         self.stages = nn.ModuleList()
#
#         for idx in range(depth):
#             down_sample = False if idx == 0 else True
#             self.stages.append(make_vgg_layer(inplanes=filters[idx],
#                                               planes=filters[idx + 1],
#                                               num_blocks=num_blocks,
#                                               with_bn=True,
#                                               down_sample=down_sample))
#
#     def forward(self, x):
#
#         features = []
#         for stage in self.stages:
#             x = stage(x)
#             features.append(x)
#         return features
#
#     def init_weights(self, pretrained=None):
#         pass
#         # for m in self.modules():
#         #     if isinstance(m, nn.Conv2d):
#         #         kaiming_init(m)
#         #     elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
#         #         constant_init(m, 1)
#
# class MCTrans(nn.Module):
#     def __init__(self,
#                  d_model=240,
#                  nhead=8,
#                  d_ffn=1024,
#                  dropout=0.1,
#                  act="relu",
#                  n_points=4,
#                  n_levels=3,
#                  n_sa_layers=6,
#                  in_channles=[64, 64, 128, 256, 512],
#                  proj_idxs=(2, 3, 4),
#
#                  ):
#         super().__init__()
#         self.nhead = nhead
#         self.d_model = d_model
#         self.n_levels = n_levels
#
#         self.proj_idxs = proj_idxs
#         self.projs = nn.ModuleList()
#         for idx in self.proj_idxs:
#             self.projs.append(ConvModule(in_channles[idx],
#                                          d_model,
#                                          kernel_size=3,
#                                          padding=1,
#                                          conv_cfg=dict(type="Conv"),
#                                          norm_cfg=dict(type='BN'),
#                                          act_cfg=dict(type='ReLU')
#                                          ))
#
#         dsa_layer = DSALayer(d_model=d_model,
#                              d_ffn=d_ffn,
#                              dropout=dropout,
#                              activation=act,
#                              n_levels=n_levels,
#                              n_heads=nhead,
#                              n_points=n_points)
#
#         self.dsa = DSA(att_layer=dsa_layer,
#                        n_layers=n_sa_layers)
#
#         self.level_embed = nn.Parameter(torch.Tensor(n_levels, d_model))
#         self.position_embedding = build_position_encoding(position_embedding="sine", hidden_dim=d_model)
#         self._reset_parameters()
#
#     def _reset_parameters(self):
#         for p in self.parameters():
#             if p.dim() > 1:
#                 nn.init.xavier_uniform_(p)
#         for m in self.modules():
#             if isinstance(m, MSDeformAttn):
#                 m._reset_parameters()
#         normal_(self.level_embed)
#
#     def get_valid_ratio(self, mask):
#         _, H, W = mask.shape
#         valid_H = torch.sum(~mask[:, :, 0], 1)
#         valid_W = torch.sum(~mask[:, 0, :], 1)
#         valid_ratio_h = valid_H.float() / H
#         valid_ratio_w = valid_W.float() / W
#         valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
#         return valid_ratio
#
#     def projection(self, feats):
#         pos = []
#         masks = []
#         cnn_feats = []
#         tran_feats = []
#
#         for idx, feats in enumerate(feats):
#             if idx not in self.proj_idxs:
#                 cnn_feats.append(feats)
#             else:
#                 n, c, h, w = feats.shape
#                 mask = torch.zeros((n, h, w)).to(torch.bool).to(feats.device)
#                 nested_feats = NestedTensor(feats, mask)
#                 masks.append(mask)
#                 pos.append(self.position_embedding(nested_feats).to(nested_feats.tensors.dtype))
#                 tran_feats.append(feats)
#
#         for idx, proj in enumerate(self.projs):
#             tran_feats[idx] = proj(tran_feats[idx])
#
#         return cnn_feats, tran_feats, pos, masks
#
#     def forward(self, x):
#         # project and prepare for the input
#         cnn_feats, trans_feats, pos_embs, masks = self.projection(x)
#         # dsa
#         features_flatten = []
#         mask_flatten = []
#         lvl_pos_embed_flatten = []
#         feature_shapes = []
#         spatial_shapes = []
#         for lvl, (feature, mask, pos_embed) in enumerate(zip(trans_feats, masks, pos_embs)):
#             bs, c, h, w = feature.shape
#             spatial_shapes.append((h, w))
#             feature_shapes.append(feature.shape)
#
#             feature = feature.flatten(2).transpose(1, 2)
#             mask = mask.flatten(1)
#             pos_embed = pos_embed.flatten(2).transpose(1, 2)
#             lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
#             lvl_pos_embed_flatten.append(lvl_pos_embed)
#
#             features_flatten.append(feature)
#             mask_flatten.append(mask)
#
#         features_flatten = torch.cat(features_flatten, 1)
#         mask_flatten = torch.cat(mask_flatten, 1)
#         lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
#         spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=features_flatten.device)
#         level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
#         valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)
#
#         # self att
#         feats = self.dsa(features_flatten,
#                          spatial_shapes,
#                          level_start_index,
#                          valid_ratios,
#                          lvl_pos_embed_flatten,
#                          mask_flatten)
#         # recover
#         out = []
#         features = feats.split(spatial_shapes.prod(1).tolist(), dim=1)
#         for idx, (feats, ori_shape) in enumerate(zip(features, spatial_shapes)):
#             out.append(feats.transpose(1, 2).reshape(feature_shapes[idx]))
#
#         cnn_feats.extend(out)
#         return cnn_feats
#
# class AttBlock(nn.Module):
#     def __init__(self, F_g, F_l, F_int):
#         super(AttBlock, self).__init__()
#         self.W_g = nn.Sequential(
#             nn.Conv2d(in_channels=F_g,
#                       out_channels=F_int,
#                       kernel_size=1,
#                       stride=1,
#                       padding=0,
#                       bias=True),
#             nn.BatchNorm2d(F_int)
#         )
#
#         self.W_x = nn.Sequential(
#             nn.Conv2d(in_channels=F_l,
#                       out_channels=F_int,
#                       kernel_size=1,
#                       stride=1,
#                       padding=0,
#                       bias=True),
#             nn.BatchNorm2d(F_int)
#         )
#
#         self.psi = nn.Sequential(
#             nn.Conv2d(in_channels=F_int,
#                       out_channels=1,
#                       kernel_size=1,
#                       stride=1,
#                       padding=0,
#                       bias=True),
#             nn.BatchNorm2d(1),
#             nn.Sigmoid()
#         )
#
#         self.relu = nn.ReLU(inplace=True)
#
#     def forward(self, g, x):
#         g1 = self.W_g(g)
#         x1 = self.W_x(x)
#         psi = self.relu(g1 + x1)
#         psi = self.psi(psi)
#
#         return x * psi
#
#
# class DecBlock(nn.Module):
#     def __init__(
#             self,
#             in_channels,
#             skip_channels,
#             out_channels,
#             attention=False
#     ):
#         super().__init__()
#         self.conv1 = conv_bn_relu(in_channels=in_channels + skip_channels,
#                                   out_channels=out_channels)
#
#         self.conv2 = conv_bn_relu(in_channels=out_channels,
#                                   out_channels=out_channels)
#
#         self.up = nn.Upsample(scale_factor=2,
#                               mode='bilinear',
#                               align_corners=True)
#
#         if attention:
#             self.att = AttBlock(F_g=in_channels, F_l=skip_channels, F_int=in_channels)
#
#     def forward(self, x, skip=None):
#         x = self.up(x)
#         if skip is not None:
#             if hasattr(self, "att"):
#                 skip = self.att(g=x, x=skip)
#             x = torch.cat([x, skip], dim=1)
#         x = self.conv1(x)
#         x = self.conv2(x)
#         return x
#
# class UNetDecoder(nn.Module):
#     def __init__(
#             self,
#             in_channels,
#             att=False
#     ):
#         super().__init__()
#         self.decoders = nn.ModuleList()
#         in_channels = in_channels[::-1]
#         skip_channels = in_channels[1:]
#         for in_c, skip_c in zip(in_channels, skip_channels):
#             self.decoders.append(DecBlock(in_c, skip_c, skip_c, att))
#
#     def forward(self, features):
#         features = features[::-1]
#         x = features[0]
#         skips = features[1:]
#
#         for i, layer in enumerate(self.decoders):
#             x = layer(x, skips[i])
#
#         return x
#
#     def init_weights(self):
#         pass



