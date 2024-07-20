from torch import Tensor
import torch
import torch.nn as nn
from torch import nn, einsum
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
import math
# from utils import create_mask
import torchvision.transforms.functional as F
import torchvision
from torch.nn.utils.rnn import pad_sequence
#import pytorchvideo.models.x3d as x3d
import utils as utils

""" PyTorch MBART model."""
from transformers import MBartForConditionalGeneration, MBartPreTrainedModel, MBartModel, MBartConfig
from transformers.modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    Seq2SeqLMOutput,
    Seq2SeqModelOutput,
    Seq2SeqQuestionAnsweringModelOutput,
    Seq2SeqSequenceClassifierOutput,
)
from transformers.models.mbart.modeling_mbart import shift_tokens_right

from transformers.models.mbart.modeling_mbart import MBartLearnedPositionalEmbedding, MBartEncoderLayer, _expand_mask

from collections import OrderedDict


import copy
import math
import random
from typing import List, Optional, Tuple, Union
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import numpy as np

# global definition
from definition import *

from hpman.m import _
from pathlib import Path

class PositionalEncoding(nn.Module):
    def __init__(self,
                 emb_size: int,
                 dropout: float,
                 maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2)* math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: Tensor):
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])


def make_resnet(name='resnet18'):
    if name == 'resnet18':
        model = torchvision.models.resnet18(pretrained=True)
    elif name == 'resnet34':
        model = torchvision.models.resnet34(pretrained=True)
    elif name == 'resnet50':
        model = torchvision.models.resnet50(pretrained=True)
    elif name == 'resnet101':
        model = torchvision.models.resnet101(pretrained=True)
    else:
        raise Exception('There are no supported resnet model {}.'.format(_('resnet')))

    inchannel = model.fc.in_features
    model.fc = nn.Identity()
    return model

class resnet(nn.Module):
    def __init__(self):
        super(resnet, self).__init__()
        self.resnet = make_resnet(name='resnet18')

    def forward(self, x, lengths):
        x = self.resnet(x)
        x_batch = []
        start = 0
        for length in lengths:
            end = start + length
            x_batch.append(x[start:end])
            start = end
        x = pad_sequence(x_batch,padding_value=PAD_IDX,batch_first=True)
        return x
# 测试代码
if __name__ == "__main__":
    # 创建模型
    model = resnet()

    # 创建一个随机输入张量 (batch_size, channels, height, width)
    input_tensor = torch.randn(6, 3, 224, 224)  # 假设 batch_size = 6，图像尺寸为 224x224

    # 定义每个样本的长度
    lengths = [1, 1, 1, 1, 1, 1]  # 假设每个图像都对应一个特征

    # 执行前向传播
    output = model(input_tensor, lengths)

    # 打印输出形状
    print(output.shape)


class TemporalConv(nn.Module):
    def __init__(self, input_size, hidden_size, conv_type=2):
        super(TemporalConv, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.conv_type = conv_type

        if self.conv_type == 0:
            self.kernel_size = ['K3']
        elif self.conv_type == 1:
            self.kernel_size = ['K5', "P2"]
        elif self.conv_type == 2:
            self.kernel_size = ['K5', "P2", 'K5', "P2"]

        modules = []
        for layer_idx, ks in enumerate(self.kernel_size):
            input_sz = self.input_size if layer_idx == 0 else self.hidden_size
            if ks[0] == 'P':
                modules.append(nn.MaxPool1d(kernel_size=int(ks[1]), ceil_mode=False))
            elif ks[0] == 'K':
                modules.append(
                    nn.Conv1d(input_sz, self.hidden_size, kernel_size=int(ks[1]), stride=1, padding=0)
                )
                modules.append(nn.BatchNorm1d(self.hidden_size))
                modules.append(nn.ReLU(inplace=True))
        self.temporal_conv = nn.Sequential(*modules)

    def forward(self, x):
        print(f"Input shape (before permute): {x.shape}")
        x = x.permute(0, 2, 1)
        print(f"Shape after permute: {x.shape}")
        x = self.temporal_conv(x)
        print(f"Output shape (after conv): {x.shape}")
        x = x.permute(0, 2, 1)
        print(f"Final output shape (after permute back): {x.shape}")
        return x


# input_size = 16
# hidden_size = 32
# conv_type = 2
#
# model = TemporalConv(input_size, hidden_size, conv_type)
# print(model)
#
# x = torch.randn(10, 50, input_size)  # (batch_size, sequence_length, input_size)
# output = model.forward(x)
# print(output.shape)

def make_head(inplanes, planes, head_type):
    if head_type == 'linear':
        return nn.Linear(inplanes, planes, bias=False)
    else:
        return nn.Identity()
class TextCLIP(nn.Module):
    def __init__(self, config=None, inplanes=1024, planes=1024, head_type='identy'):
        super(TextCLIP, self).__init__()

        self.model_txt = MBartForConditionalGeneration.from_pretrained(config['model']['transformer']).get_encoder()

        self.lm_head = make_head(inplanes, planes, head_type)

    def forward(self, tgt_input):
        txt_logits = self.model_txt(input_ids=tgt_input['input_ids'].cuda(), attention_mask=tgt_input['attention_mask'].cuda())[0]
        output = txt_logits[torch.arange(txt_logits.shape[0]), tgt_input['input_ids'].argmax(dim=-1)]
        return self.lm_head(output), txt_logits


class Text_Decoder(nn.Module):
    def __init__(self, config):
        super(Text_Decoder, self).__init__()
        self.text_decoder = MBartForConditionalGeneration.from_pretrained(
            config['model']['visual_encoder']).get_decoder()
        self.lm_head = MBartForConditionalGeneration.from_pretrained(
            config['model']['visual_encoder']).get_output_embeddings()
        self.register_buffer("final_logits_bias", torch.zeros((1, MBartForConditionalGeneration.from_pretrained(
            config['model']['visual_encoder']).model.shared.num_embeddings)))

    def forward(self, tgt_input, masked_tgt_input, model_txt):
        with torch.no_grad():
            _, encoder_hidden_states = model_txt(masked_tgt_input)

        decoder_input_ids = shift_tokens_right(tgt_input['input_ids'].cuda(), self.text_decoder.config.pad_token_id)
        decoder_out = self.text_decoder(
            input_ids=decoder_input_ids,
            attention_mask=tgt_input['attention_mask'].cuda(),
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=masked_tgt_input['attention_mask'].cuda(),
            return_dict=True,
        )
        lm_logits = self.lm_head(decoder_out[0]) + self.final_logits_bias

        return lm_logits



class SLRCLIP(nn.Module):
    def __init__(self, config, embed_dim=1024) :
        super(SLRCLIP, self).__init__()
        self.model_txt = TextCLIP(config, inplanes=embed_dim, planes=embed_dim)
        self.model_images = ImageCLIP(config, inplanes=embed_dim, planes=embed_dim)
        self.rgb_fusion = rgb_fusion(frozen=False)

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def get_model_txt(self):
        return self.model_txt

    def get_encoder_hidden_states(self):
        return self.encoder_hidden_states
    def detect_keypoints(self,x):
        keypoints = []
        non_zero = (x!=0).any(dim=1)
        batch_indices,rows,cols = torch.nonzero(non_zero,as_tuple=True)
        keypoints = list(zip(batch_indices.tolist(),rows.tolist(),cols.tolist()))
        return keypoints
    def forward(self, src_input1,src_input2,tgt_input):
        # 两个input
        keypoints = self.detect_keypoints(src_input2)
        fused_img = self.rgb_fusion(src_input1,src_input2)
        expanded_img = self.rgb_fusion.expand_keypoints(fused_img,keypoints,radius=5)
        image_features = self.model_images(expanded_img)
        text_features,self.encoder_hidden_states = self.model_txt(tgt_input)

        image_features = image_features/image_features.norm(dim=-1, keepdim=True)
        text_features = text_features/text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale*image_features @text_features.t()
        logits_per_text = logit_scale * text_features @ image_features.t()

        ground_truth = torch.eye(logits_per_image.shape[0],device=logits_per_text.device,dtype=logits_per_image.dtype,requires_grad=False)

        return logits_per_image, logits_per_text, ground_truth
class ImageCLIP(nn.Module):
    def __init__(self, config, embed_dim=1024) :
        super(ImageCLIP, self).__init__()
        self.config = config

        self.model = FeatureExtracter()

        return

class rgb_fusion(nn.Module):
    def __init__(self,frozen = False):
        super(rgb_fusion, self).__init__()
        self.alpha = nn.Parameter(torch.tensor(255))
        if frozen:
            self.alpha.requires_grad =False
    def forward(self, x1,x2):
        out = x1 + self.alpha * x2
        return out
    def expand_keypoints(self,x,keypoints,radius=5):
        # 半径可以设置和更改试试，后续可以改成可以训练的值
        for (batch_idx, row, col) in keypoints:
            # 确保索引不会越界
            row_start = max(0, row - radius)
            row_end = min(x.size(2), row + radius + 1)
            col_start = max(0, col - radius)
            col_end = min(x.size(3), col + radius + 1)
            # 取关键点的值
            value = x[batch_idx, :, row, col].unsqueeze(-1).unsqueeze(-1)
            # 将值广播到指定的区域
            x[batch_idx, :, row_start:row_end, col_start:col_end] = value
        return x
    # 用boardcast 广播直接一步扩散，而不是用循环



class FeatureExtracter(nn.Module):
    def __init__(self, frozen=False):
        super(FeatureExtracter, self).__init__()
        self.conv_2d = resnet() # InceptionI3d()
        self.conv_1d = TemporalConv(input_size=512, hidden_size=1024, conv_type=2)

        if frozen:
            for param in self.conv_2d.parameters():
                param.requires_grad = False
    def forward(self,
                src:Tensor,
                src_length_batch):
        src = self.conv_2d(src,src_length_batch)
        src = self.conv_1d(src)

        return src

class V_encoder(nn.Module):
    def __init__(self, emb_size,feature_size,config):
        super(V_encoder, self).__init__()
        self.config = config
        self.src_emb = nn.Linear(feature_size,emb_size)
        modules = []
        modules.append(nn.BatchNorm1d(emb_size))
        modules.append(nn.ReLU(inplace=True))
        self.bn_ac = nn.Sequential(*modules)

        for m in self.modules():
            if isinstance(m, (nn.Conv1d,nn.Linear)):
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    def forward(self, src:Tensor):
        src = self.src_emb(src)
        src = self.bn_ac(src.permute(0,2,1)).permute(0,2,1)
        return src


def config_decoder(config):
    from transformers import AutoConfig

    decoder_type = _('decoder_type', 'LD', choices=['LD', 'LLMD'])
    if decoder_type == 'LD':

        return MBartForConditionalGeneration.from_pretrained(config['model']['visual_encoder'],
                                                             ignore_mismatched_sizes=True,
                                                             config=AutoConfig.from_pretrained(Path(
                                                                 config['model']['visual_encoder']) / 'config.json'))
    elif decoder_type == 'LLMD':
        return MBartForConditionalGeneration.from_pretrained(config['model']['transformer'],
                                                             ignore_mismatched_sizes=True,
                                                             config=AutoConfig.from_pretrained(Path(
                                                                 config['model']['transformer']) / 'LLMD_config.json'))


class VK_model(nn.Module):
    def __init__(self, config,args,embed_dim=1024):
        super(VK_model, self).__init__()
        self.args = args
        self.config = config

        self.backbone = RGBFeatureFusion(frozen=config.get('freeze_backbone', False))
        self.mbart = config_decoder(config)

        if config['model']['sign_proj']:
            self.sign_emb = V_encoder(emb_size=embed_dim,feature_size=embed_dim, config = config)
            self.embed_scale = math.sqrt(embed_dim) if config['training']['scale_embedding'] else 1.0
        else:
            self.sign_emb = nn.Identity()
            self.embed_scale = 1.0

    # def share_forward(self,src_input):
    #     images = src_input['img_ids'].cuda()  # 修改这里
    #     keypoints = src_input['kp_ids'].cuda()  # 确认这里的键也是正确的
    #     src_length_batch = src_input['src_length_batch']
    #
    #     print(f"Images shape: {images.shape}")
    #     print(f"Keypoints shape: {keypoints.shape}")
    #
    #
    #     frames_feature = self.backbone(images, keypoints, src_length_batch)
    #     print(f"Frames features shape after backbone: {frames_feature.shape}")
    #     attention_mask = src_input['attention_mask']
    #
    #     inputs_embeds = self.sign_emb(frames_feature)
    #     inputs_embeds = self.embed_scale * inputs_embeds
    #
    #
    #     return inputs_embeds, attention_mask
    def share_forward(self, src_input):
        images = src_input['img_ids'].cuda()  # 修改这里
        keypoints = src_input['kp_ids'].cuda()  # 确认这里的键也是正确的
        src_length_batch = src_input['src_length_batch']

        print(f"Images shape: {images.shape}")
        print(f"Keypoints shape: {keypoints.shape}")

        frames_feature = self.backbone(images, keypoints, src_length_batch)
        print(f"Frames features shape after backbone: {frames_feature.shape}")
        attention_mask = src_input['attention_mask']

        inputs_embeds = self.sign_emb(frames_feature)
        inputs_embeds = self.embed_scale * inputs_embeds

        # 检查inputs_embeds的范围
        if inputs_embeds.max() > 1e4 or inputs_embeds.min() < -1e4:
            raise ValueError(f"inputs_embeds中的值超出了合理范围，"
                             f"最大值：{inputs_embeds.max()}，最小值：{inputs_embeds.min()}")

        return inputs_embeds, attention_mask

    # def forward(self,src_input,tgt_input):
    #
    #     inputs_embeds, attention_mask = self.share_forward(src_input)
    #
    #     out = self.mbart(inputs_embeds=inputs_embeds,
    #                      attention_mask=attention_mask.cuda(),
    #                      # decoder_input_ids = tgt_input['input_ids'].cuda(),
    #                      labels=tgt_input['input_ids'].cuda(),
    #                      decoder_attention_mask=tgt_input['attention_mask'].cuda(),
    #                      return_dict=True,
    #                      )
    #     return out['logits']
    def forward(self, src_input, tgt_input):
        inputs_embeds, attention_mask = self.share_forward(src_input)

        # 获取词汇表大小
        vocab_size = self.mbart.model.shared.num_embeddings

        # 检查目标输入的范围
        if tgt_input['input_ids'].max() >= vocab_size or tgt_input['input_ids'].min() < 0:
            raise ValueError(f"tgt_input['input_ids']中的值超出了词汇表的范围，"
                             f"最大值：{tgt_input['input_ids'].max()}，最小值：{tgt_input['input_ids'].min()}，"
                             f"词汇表大小：{vocab_size}")

        out = self.mbart(inputs_embeds=inputs_embeds,
                         attention_mask=attention_mask.cuda(),
                         labels=tgt_input['input_ids'].cuda(),
                         decoder_attention_mask=tgt_input['attention_mask'].cuda(),
                         return_dict=True,
                         )
        return out['logits']

    def generate(self,src_input,max_new_tokens,num_beams,decoder_start_token_id):
        inputs_embeds, attention_mask = self.share_forward(src_input)

        out = self.mbart.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask.cuda(),
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            decoder_start_token_id=decoder_start_token_id,

        )
        return out


class RGBFeatureFusion(nn.Module):
    def __init__(self,frozen = False):
        super(RGBFeatureFusion,self).__init__()
        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.conv_2d = resnet()
        self.conv_1d = TemporalConv(input_size=512, hidden_size=1024, conv_type=2)


        if frozen:
            for param in self.conv_2d.parameters():
                param.requires_grad = False
            self.alpha.requires_grad =False

    def forward(self, x1, x2, src_length_batch, radius=5):
        # 检测关键点
        keypoints = self.detect_keypoints(x2)
        print(len(keypoints))
        if keypoints and keypoints[0][0] == 0:  # 检查是否有关键点，且第一个关键点属于第一个批次\

            for i, (batch_idx, row, col) in enumerate(keypoints):
                if batch_idx == 0:  # 只考虑第一个批次
                    if i == 0:  # 只打印第一个关键点的信息，表示第一张图片的第一个关键点
                        print(f"Keypoint {i}: Batch {batch_idx}, Position ({row}, {col}), Value {x2[batch_idx, :, row, col]}")



        # 扩展关键点
        x2 = self.expand_keypoints(x2, keypoints, radius)
        print(f"x2 shape after expand_keypoints: {x2.shape}")

        # 加权求和
        out = x1 + self.alpha * x2
        print(f"out shape after alpha * x2: {out.shape}")
        out = F.resize(out,(256,256))
        print("输出维度大小",out.shape)
        # 通过 2D 卷积
        out = self.conv_2d(out, src_length_batch)
        print(f"out shape after conv_2d: {out.shape}")

        # 通过 1D 卷积
        out = self.conv_1d(out)
        print(f"out shape after conv_1d: {out.shape}")

        return out

    def expand_keypoints(self, x, keypoints, radius=2):
        # 对关键点周围的区域进行扩展
        for (batch_idx, row, col) in keypoints:
            row_start = max(0, row - radius)
            row_end = min(x.size(2), row + radius + 1)
            col_start = max(0, col - radius)
            col_end = min(x.size(3), col + radius + 1)
            value = x[batch_idx, :, row, col].unsqueeze(-1).unsqueeze(-1)
            x[batch_idx, :, row_start:row_end, col_start:col_end] = value
            print(f"Expanding at ({row}, {col}) with radius {radius}: row {row_start} to {row_end}, col {col_start} to {col_end}")
        return x
    def detect_keypoints(self, x):
        # 检测关键点
        keypoints = []
        non_zero = (x != 0).any(dim=1)
        batch_indices, rows, cols = torch.nonzero(non_zero, as_tuple=True)
        keypoints = list(zip(batch_indices.tolist(), rows.tolist(), cols.tolist()))
        return keypoints

