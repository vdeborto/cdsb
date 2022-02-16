from abc import abstractmethod

import math

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from .fp16_util import convert_module_to_f16, convert_module_to_f32
from .layers import *


class UNetModel(nn.Module):
    """
    The full UNet model with attention and timestep embedding.
    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    :param attention_resolutions: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention
        will be used.
    :param dropout: the dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet.
    :param conv_resample: if True, use learned convolutions for upsampling and
        downsampling.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param num_classes: if specified (as an int), then this model will be
        class-conditional with `num_classes` classes.
    :param use_checkpoint: use gradient checkpointing to reduce memory usage.
    :param num_heads: the number of attention heads in each attention layer.
    :param use_scale_shift_norm: use a FiLM-like conditioning mechanism.
    :param resblock_updown: use residual blocks for up/downsampling.
    """

    def __init__(
        self,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        num_classes=None,
        use_checkpoint=False,
        num_heads=1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        temb_max_period=10000
    ):
        super().__init__()

        self.locals = [ in_channels,
                        model_channels,
                        out_channels,
                        num_res_blocks,
                        attention_resolutions,
                        dropout,
                        channel_mult,
                        conv_resample,
                        dims,
                        num_classes,
                        use_checkpoint,
                        num_heads,
                        use_scale_shift_norm,
                        resblock_updown,
                        temb_max_period
                    ]
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.num_heads = num_heads
        self.temb_max_period = temb_max_period

        self.input_ch = int(channel_mult[0] * model_channels)
        ch = self.input_ch

        time_embed_dim = self.input_ch * 4
        self.time_embed = nn.Sequential(
            linear(self.input_ch, time_embed_dim),
            nn.SiLU(inplace=True),
            linear(time_embed_dim, time_embed_dim),
        )

        if self.num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_embed_dim)

        self.input_blocks = nn.ModuleList(
            [TimestepEmbedSequential(conv_nd(dims, in_channels, ch, 3, padding=1))]
        )
        self._feature_size = ch
        input_block_chans = [ch]
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=int(mult * model_channels),
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = int(mult * model_channels)
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch, use_checkpoint=use_checkpoint, num_heads=num_heads
                        )
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(ch, use_checkpoint=use_checkpoint, num_heads=num_heads),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self._feature_size += ch

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(
                        ch + ich,
                        time_embed_dim,
                        dropout,
                        out_channels=int(model_channels * mult),
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = int(model_channels * mult)
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads,
                        )
                    )
                if level and i == num_res_blocks:
                    out_ch = ch
                    layers.append(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,
                        )
                        if resblock_updown
                        else Upsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch

        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(inplace=True),
            zero_module(conv_nd(dims, self.input_ch, out_channels, 3, padding=1)),
        )


    def convert_to_fp16(self):
        """
        Convert the torso of the model to float16.
        """
        self.input_blocks.apply(convert_module_to_f16)
        self.middle_block.apply(convert_module_to_f16)
        self.output_blocks.apply(convert_module_to_f16)

    def convert_to_fp32(self):
        """
        Convert the torso of the model to float32.
        """
        self.input_blocks.apply(convert_module_to_f32)
        self.middle_block.apply(convert_module_to_f32)
        self.output_blocks.apply(convert_module_to_f32)


    def forward(self, x, y, timesteps):

        """
        Apply the model to an input batch.
        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """
        timesteps = timesteps.squeeze()
        assert (y is not None) == (
            self.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"

        hs = []
        emb = self.time_embed(timestep_embedding(timesteps, self.input_ch, self.temb_max_period))

        if self.num_classes is not None:
            assert y.shape == (x.shape[0],)
            emb = emb + self.label_emb(y)

        h = x  # .type(self.dtype)
        for module in self.input_blocks:
            h = module(h, emb)
            hs.append(h)
        h = self.middle_block(h, emb)
        for module in self.output_blocks:
            h = th.cat([h, hs.pop()], dim=1)
            h = module(h, emb)
        h = h.type(x.dtype)
        return self.out(h)

    # def get_feature_vectors(self, x, y, timesteps):
    #     """
    #     Apply the model and return all of the intermediate tensors.
    #     :param x: an [N x C x ...] Tensor of inputs.
    #     :param timesteps: a 1-D batch of timesteps.
    #     :param y: an [N] Tensor of labels, if class-conditional.
    #     :return: a dict with the following keys:
    #              - 'down': a list of hidden state tensors from downsampling.
    #              - 'middle': the tensor of the output of the lowest-resolution
    #                          block in the model.
    #              - 'up': a list of hidden state tensors from upsampling.
    #     """
    #     hs = []
    #     emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))
    #     if self.num_classes is not None:
    #         assert y.shape == (x.shape[0],)
    #         emb = emb + self.label_emb(y)
    #     result = dict(down=[], up=[])
    #     h = x#.type(self.inner_dtype)
    #     for module in self.input_blocks:
    #         h = module(h, emb)
    #         hs.append(h)
    #         result["down"].append(h.type(x.dtype))
    #     h = self.middle_block(h, emb)
    #     result["middle"] = h.type(x.dtype)
    #     for module in self.output_blocks:
    #         cat_in = th.cat([h, hs.pop()], dim=1)
    #         h = module(cat_in, emb)
    #         result["up"].append(h.type(x.dtype))
    #     return result


class SuperResModel(UNetModel):
    """
    A UNetModel that performs super-resolution.
    Expects an extra kwarg `low_res` to condition on a low-resolution image.
    """

    def __init__(self, in_channels, *args, **kwargs):
        super().__init__(in_channels*2, *args, **kwargs)
        self.locals[0] = in_channels

    def forward(self, x, low_res, timesteps, **kwargs):
        _, _, new_height, new_width = x.shape
        upsampled = F.interpolate(low_res, (new_height, new_width), mode="bilinear")
        x = th.cat([x, upsampled], dim=1)
        return super().forward(x, None, timesteps, **kwargs)

    # def get_feature_vectors(self, x, low_res, timesteps, **kwargs):
    #     _, new_height, new_width, _ = x.shape
    #     upsampled = F.interpolate(low_res, (new_height, new_width), mode="bilinear")
    #     x = th.cat([x, upsampled], dim=1)
    #     return super().get_feature_vectors(x, None, timesteps, **kwargs)


class BasicUNetModel(nn.Module):
    """
    The full UNet model with attention.
    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    :param attention_resolutions: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention
        will be used.
    :param dropout: the dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet.
    :param conv_resample: if True, use learned convolutions for upsampling and
        downsampling.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param use_checkpoint: use gradient checkpointing to reduce memory usage.
    :param num_heads: the number of attention heads in each attention layer.
    :param resblock_updown: use residual blocks for up/downsampling.
    """
    def __init__(
        self,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        use_checkpoint=False,
        num_heads=1,
        resblock_updown=False
    ):
        super().__init__()

        self.locals = [ in_channels,
                        model_channels,
                        out_channels,
                        num_res_blocks,
                        attention_resolutions,
                        dropout,
                        channel_mult,
                        conv_resample,
                        dims,
                        use_checkpoint,
                        num_heads,
                        resblock_updown
                    ]
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.use_checkpoint = use_checkpoint
        self.num_heads = num_heads

        self.input_ch = int(channel_mult[0] * model_channels)
        ch = self.input_ch

        self.input_blocks = nn.ModuleList(
            [conv_nd(dims, in_channels, ch, 3, padding=1)]
        )
        self._feature_size = ch
        input_block_chans = [ch]
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    BasicResBlock(
                        ch,
                        dropout,
                        out_channels=int(mult * model_channels),
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                    )
                ]
                ch = int(mult * model_channels)
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch, use_checkpoint=use_checkpoint, num_heads=num_heads
                        )
                    )
                self.input_blocks.append(nn.Sequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    nn.Sequential(
                        BasicResBlock(
                            ch,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        self.middle_block = nn.Sequential(
            BasicResBlock(
                ch,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint
            ),
            AttentionBlock(ch, use_checkpoint=use_checkpoint, num_heads=num_heads),
            BasicResBlock(
                ch,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint
            ),
        )
        self._feature_size += ch

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [
                    BasicResBlock(
                        ch + ich,
                        dropout,
                        out_channels=int(model_channels * mult),
                        dims=dims,
                        use_checkpoint=use_checkpoint
                    )
                ]
                ch = int(model_channels * mult)
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads,
                        )
                    )
                if level and i == num_res_blocks:
                    out_ch = ch
                    layers.append(
                        BasicResBlock(
                            ch,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            up=True
                        )
                        if resblock_updown
                        else Upsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                    ds //= 2
                self.output_blocks.append(nn.Sequential(*layers))
                self._feature_size += ch

        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(inplace=True),
            zero_module(conv_nd(dims, self.input_ch, out_channels, 3, padding=1)),
        )

    # def convert_to_fp16(self):
    #     """
    #     Convert the torso of the model to float16.
    #     """
    #     self.input_blocks.apply(convert_module_to_f16)
    #     self.middle_block.apply(convert_module_to_f16)
    #     self.output_blocks.apply(convert_module_to_f16)
    #
    # def convert_to_fp32(self):
    #     """
    #     Convert the torso of the model to float32.
    #     """
    #     self.input_blocks.apply(convert_module_to_f32)
    #     self.middle_block.apply(convert_module_to_f32)
    #     self.output_blocks.apply(convert_module_to_f32)

    def forward(self, y):
        hs = []

        h = y  # .type(self.dtype)
        for module in self.input_blocks:
            h = module(h)
            hs.append(h)
        h = self.middle_block(h)
        for module in self.output_blocks:
            h = th.cat([h, hs.pop()], dim=1)
            h = module(h)
        h = h.type(y.dtype)
        return self.out(h)
