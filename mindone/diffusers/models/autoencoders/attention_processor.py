# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Callable, Optional, Union

import mindspore as ms
from mindspore import nn, ops

from ..utils import logging
from .normalization import GroupNorm, LayerNorm

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class Attention(nn.Cell):
    r"""
    A cross attention layer.

    Parameters:
        query_dim (`int`):
            The number of channels in the query.
        cross_attention_dim (`int`, *optional*):
            The number of channels in the encoder_hidden_states. If not given, defaults to `query_dim`.
        heads (`int`,  *optional*, defaults to 8):
            The number of heads to use for multi-head attention.
        dim_head (`int`,  *optional*, defaults to 64):
            The number of channels in each head.
        dropout (`float`, *optional*, defaults to 0.0):
            The dropout probability to use.
        bias (`bool`, *optional*, defaults to False):
            Set to `True` for the query, key, and value linear layers to contain a bias parameter.
        upcast_attention (`bool`, *optional*, defaults to False):
            Set to `True` to upcast the attention computation to `float32`.
        upcast_softmax (`bool`, *optional*, defaults to False):
            Set to `True` to upcast the softmax computation to `float32`.
        cross_attention_norm (`str`, *optional*, defaults to `None`):
            The type of normalization to use for the cross attention. Can be `None`, `layer_norm`, or `group_norm`.
        cross_attention_norm_num_groups (`int`, *optional*, defaults to 32):
            The number of groups to use for the group norm in the cross attention.
        added_kv_proj_dim (`int`, *optional*, defaults to `None`):
            The number of channels to use for the added key and value projections. If `None`, no projection is used.
        norm_num_groups (`int`, *optional*, defaults to `None`):
            The number of groups to use for the group norm in the attention.
        spatial_norm_dim (`int`, *optional*, defaults to `None`):
            The number of channels to use for the spatial normalization.
        out_bias (`bool`, *optional*, defaults to `True`):
            Set to `True` to use a bias in the output linear layer.
        scale_qk (`bool`, *optional*, defaults to `True`):
            Set to `True` to scale the query and key by `1 / sqrt(dim_head)`.
        only_cross_attention (`bool`, *optional*, defaults to `False`):
            Set to `True` to only use cross attention and not added_kv_proj_dim. Can only be set to `True` if
            `added_kv_proj_dim` is not `None`.
        eps (`float`, *optional*, defaults to 1e-5):
            An additional value added to the denominator in group normalization that is used for numerical stability.
        rescale_output_factor (`float`, *optional*, defaults to 1.0):
            A factor to rescale the output by dividing it with this value.
        residual_connection (`bool`, *optional*, defaults to `False`):
            Set to `True` to add the residual connection to the output.
        _from_deprecated_attn_block (`bool`, *optional*, defaults to `False`):
            Set to `True` if the attention block is loaded from a deprecated state dict.
        processor (`AttnProcessor`, *optional*, defaults to `None`):
            The attention processor to use. If `None`, defaults to `AttnProcessor2_0` if `torch 2.x` is used and
            `AttnProcessor` otherwise.
    """

    def __init__(
        self,
        query_dim: int,
        cross_attention_dim: Optional[int] = None,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        bias: bool = False,
        upcast_attention: bool = False,
        upcast_softmax: bool = False,
        cross_attention_norm: Optional[str] = None,
        cross_attention_norm_num_groups: int = 32,
        added_kv_proj_dim: Optional[int] = None,
        norm_num_groups: Optional[int] = None,
        spatial_norm_dim: Optional[int] = None,
        out_bias: bool = True,
        scale_qk: bool = True,
        only_cross_attention: bool = False,
        eps: float = 1e-5,
        rescale_output_factor: float = 1.0,
        residual_connection: bool = False,
        _from_deprecated_attn_block: bool = False,
        processor: Optional["AttnProcessor"] = None,
        out_dim: int = None,
    ):
        super().__init__()
        self.inner_dim = out_dim if out_dim is not None else dim_head * heads
        self.query_dim = query_dim
        self.use_bias = bias
        self.is_cross_attention = cross_attention_dim is not None
        self.cross_attention_dim = cross_attention_dim if cross_attention_dim is not None else query_dim
        self.upcast_attention = upcast_attention
        self.upcast_softmax = upcast_softmax
        self.rescale_output_factor = rescale_output_factor
        self.residual_connection = residual_connection
        self.dropout = dropout
        self.fused_projections = False
        self.out_dim = out_dim if out_dim is not None else query_dim

        # we make use of this private variable to know whether this class is loaded
        # with an deprecated state dict so that we can convert it on the fly
        self._from_deprecated_attn_block = _from_deprecated_attn_block

        self.scale_qk = scale_qk
        self.scale = dim_head**-0.5 if self.scale_qk else 1.0

        self.heads = out_dim // dim_head if out_dim is not None else heads
        # for slice_size > 0 the attention score computation
        # is split across the batch axis to save memory
        # You can set slice_size with `set_attention_slice`
        self.sliceable_head_dim = heads

        self.added_kv_proj_dim = added_kv_proj_dim
        self.only_cross_attention = only_cross_attention

        if self.added_kv_proj_dim is None and self.only_cross_attention:
            raise ValueError(
                "`only_cross_attention` can only be set to True if `added_kv_proj_dim` is not None. Make sure to set either `only_cross_attention=False` or define `added_kv_proj_dim`."  # noqa: E501
            )

        if norm_num_groups is not None:
            self.group_norm = GroupNorm(num_channels=query_dim, num_groups=norm_num_groups, eps=eps, affine=True)
        else:
            self.group_norm = None

        if spatial_norm_dim is not None:
            raise NotImplementedError("SpatialNorm is not implemented.")
        else:
            self.spatial_norm = None

        self.cross_attention_norm = cross_attention_norm
        if cross_attention_norm is None:
            self.norm_cross = None
        elif cross_attention_norm == "layer_norm":
            self.norm_cross = LayerNorm(self.cross_attention_dim)
        elif cross_attention_norm == "group_norm":
            if self.added_kv_proj_dim is not None:
                # The given `encoder_hidden_states` are initially of shape
                # (batch_size, seq_len, added_kv_proj_dim) before being projected
                # to (batch_size, seq_len, cross_attention_dim). The norm is applied
                # before the projection, so we need to use `added_kv_proj_dim` as
                # the number of channels for the group norm.
                norm_cross_num_channels = added_kv_proj_dim
            else:
                norm_cross_num_channels = self.cross_attention_dim

            self.norm_cross = GroupNorm(
                num_channels=norm_cross_num_channels, num_groups=cross_attention_norm_num_groups, eps=1e-5, affine=True
            )
        else:
            raise ValueError(
                f"unknown cross_attention_norm: {cross_attention_norm}. Should be None, 'layer_norm' or 'group_norm'"
            )

        linear_cls = nn.Dense

        self.linear_cls = linear_cls
        self.to_q = linear_cls(query_dim, self.inner_dim, has_bias=bias)

        if not self.only_cross_attention:
            # only relevant for the `AddedKVProcessor` classes
            self.to_k = linear_cls(self.cross_attention_dim, self.inner_dim, has_bias=bias)
            self.to_v = linear_cls(self.cross_attention_dim, self.inner_dim, has_bias=bias)
        else:
            self.to_k = None
            self.to_v = None

        if self.added_kv_proj_dim is not None:
            self.add_k_proj = linear_cls(added_kv_proj_dim, self.inner_dim)
            self.add_v_proj = linear_cls(added_kv_proj_dim, self.inner_dim)

        self.to_out = nn.CellList([linear_cls(self.inner_dim, self.out_dim, has_bias=out_bias), nn.Dropout(p=dropout)])

        # set attention processor
        # We use the AttnProcessor2_0 by default when torch 2.x is used which uses
        # torch.nn.functional.scaled_dot_product_attention for native Flash/memory_efficient_attention
        # but only if it has the default `scale` argument. TODO remove scale_qk check when we move to torch 2.1
        if processor is None:
            processor = AttnProcessor()
        self.processor = processor

    def set_use_memory_efficient_attention_xformers(
        self, use_memory_efficient_attention_xformers: bool, attention_op: Optional[Callable] = None
    ) -> None:
        r"""
        Set whether to use memory efficient attention from `xformers` or not.

        Args:
            use_memory_efficient_attention_xformers (`bool`):
                Whether to use memory efficient attention from `xformers` or not.
            attention_op (`Callable`, *optional*):
                Not supported for now.
        """
        if use_memory_efficient_attention_xformers:
            if not hasattr(ops.operations.nn_ops, "FlashAttentionScore"):
                raise ModuleNotFoundError(
                    f"Memory efficient attention on mindspore uses flash attention under the hoods. "
                    f"The implementation of flash attention is `FlashAttentionScore`, "
                    f"which should be available in `mindspore.ops.operations.nn_ops`. "
                    f"However, we cannot find it in current environment(mindspore version: {ms.__version__})."
                )
            elif ms.get_context("device_target") != "Ascend":
                raise ValueError(
                    f"Memory efficient attention is only available for Ascend, "
                    f"but got current device: {ms.get_context('device_target')}"
                )
            else:
                try:
                    # Make sure we can run the memory efficient attention
                    flash_attn = ops.operations.nn_ops.FlashAttentionScore(1, input_layout="BSH")
                    _ = flash_attn(
                        ops.randn(1, 16, 64, dtype=ms.float16),
                        ops.randn(1, 16, 64, dtype=ms.float16),
                        ops.randn(1, 16, 64, dtype=ms.float16),
                    )
                except Exception as e:
                    raise e

            # The following lines is a patch for flash attn, which calculates implicit padding on head_dim.
            # TODO: Remove it if flash attention has better supports.
            import bisect

            self.flash_attn_valid_head_dims = [64, 80, 96, 120, 128, 256]
            self.head_dim = self.inner_dim // self.heads
            if self.head_dim in self.flash_attn_valid_head_dims:
                self.head_dim_padding = 0
            else:
                minimum_larger_index = bisect.bisect_right(self.flash_attn_valid_head_dims, self.head_dim)
                if minimum_larger_index >= len(self.flash_attn_valid_head_dims):
                    self.head_dim_padding = -1  # head_dim is bigger than the largest one, we cannot do padding
                else:
                    self.head_dim_padding = self.flash_attn_valid_head_dims[minimum_larger_index] - self.head_dim

            if self.head_dim_padding == 0:
                logger.info(
                    f"The head dimension of '{self.to_q.weight.name[:-12]}' is {self.head_dim}. "
                    f"Successfully set to use the flash attention."
                )
                processor = XFormersAttnProcessor(attention_op=attention_op)
            elif self.head_dim_padding > 0:
                logger.warning(
                    f"Flash attention requires that the head dimension must be one of "
                    f"{self.flash_attn_valid_head_dims}, but got {self.head_dim} in '{self.to_q.weight.name[:-12]}'. "
                    f"We will implicitly pad the head dimension to {self.head_dim + self.head_dim_padding}."
                )
                processor = XFormersAttnProcessor(attention_op=attention_op)
            else:
                logger.warning(
                    f"Flash attention requires that the head dimension must be one of "
                    f"{self.flash_attn_valid_head_dims}, but got {self.head_dim} in '{self.to_q.weight.name[:-12]}'. "
                    f"Fallback to the vanilla implementation of attention."
                )
                processor = AttnProcessor()

            # # The following lines is a patch for flash attn, which fallbacks to vanilla attn if head_dim is invalid.
            # # TODO: Remove it if flash attention has better supports.
            # self.flash_attn_valid_head_dims = [64, 80, 96, 120, 128, 256]
            # self.head_dim = self.inner_dim // self.heads
            # self.head_dim_padding = 0
            # if self.head_dim in self.flash_attn_valid_head_dims:
            #     logger.info(
            #         f"The head dimension of '{self.to_q.weight.name[:-12]}' is {self.head_dim}. "
            #         f"Successfully set to use the flash attention."
            #     )
            #     processor = XFormersAttnProcessor(attention_op=attention_op)
            # else:
            #     logger.warning(
            #         f"Flash attention requires that the head dimension must be one of "
            #         f"{self.flash_attn_valid_head_dims}, but got {self.head_dim} in '{self.to_q.weight.name[:-12]}'. "
            #         f"Fallback to the vanilla implementation of attention."
            #     )
            #     processor = AttnProcessor()
        else:
            # set attention processor
            # We use the AttnProcessor2_0 by default when torch 2.x is used which uses
            # torch.nn.functional.scaled_dot_product_attention for native Flash/memory_efficient_attention
            # but only if it has the default `scale` argument. TODO remove scale_qk check when we move to torch 2.1
            processor = AttnProcessor()

        self.set_processor(processor)

    def set_processor(self, processor: "AttnProcessor") -> None:
        r"""
        Set the attention processor to use.

        Args:
            processor (`AttnProcessor`):
                The attention processor to use.
        """
        # if current processor is in `self._modules` and if passed `processor` is not, we need to
        # pop `processor` from `self._modules`
        if hasattr(self, "processor") and isinstance(self.processor, nn.Cell) and not isinstance(processor, nn.Cell):
            logger.info(f"You are removing possibly trained weights of {self.processor} with {processor}")
            self._cells.pop("processor")

        self.processor = processor

    def get_processor(self) -> "AttentionProcessor":
        r"""
        Get the attention processor in use.

        Returns:
            "AttentionProcessor": The attention processor in use.
        """
        return self.processor

    def construct(
        self,
        hidden_states: ms.Tensor,
        encoder_hidden_states: Optional[ms.Tensor] = None,
        attention_mask: Optional[ms.Tensor] = None,
        **cross_attention_kwargs,
    ) -> ms.Tensor:
        r"""
        The forward method of the `Attention` class.

        Args:
            hidden_states (`ms.Tensor`):
                The hidden states of the query.
            encoder_hidden_states (`ms.Tensor`, *optional*):
                The hidden states of the encoder.
            attention_mask (`ms.Tensor`, *optional*):
                The attention mask to use. If `None`, no mask is applied.
            **cross_attention_kwargs:
                Additional keyword arguments to pass along to the cross attention.

        Returns:
            `ms.Tensor`: The output of the attention layer.
        """
        # The `Attention` class can call different attention processors / attention functions
        # here we simply pass along all tensors to the selected processor class
        # For standard processors that are defined here, `**cross_attention_kwargs` is empty
        return self.processor(
            self,
            hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
            **cross_attention_kwargs,
        )

    def batch_to_head_dim(self, tensor: ms.Tensor) -> ms.Tensor:
        r"""
        Reshape the tensor from `[batch_size, seq_len, dim]` to `[batch_size // heads, seq_len, dim * heads]`. `heads`
        is the number of heads initialized while constructing the `Attention` class.

        Args:
            tensor (`ms.Tensor`): The tensor to reshape.

        Returns:
            `ms.Tensor`: The reshaped tensor.
        """
        head_size = self.heads
        batch_size, seq_len, dim = tensor.shape
        tensor = tensor.reshape(batch_size // head_size, head_size, seq_len, dim)
        tensor = tensor.permute(0, 2, 1, 3).reshape(batch_size // head_size, seq_len, dim * head_size)
        return tensor

    def head_to_batch_dim(self, tensor: ms.Tensor, out_dim: int = 3) -> ms.Tensor:
        r"""
        Reshape the tensor from `[batch_size, seq_len, dim]` to `[batch_size, seq_len, heads, dim // heads]` `heads` is
        the number of heads initialized while constructing the `Attention` class.

        Args:
            tensor (`ms.Tensor`): The tensor to reshape.
            out_dim (`int`, *optional*, defaults to `3`): The output dimension of the tensor. If `3`, the tensor is
                reshaped to `[batch_size * heads, seq_len, dim // heads]`.

        Returns:
            `ms.Tensor`: The reshaped tensor.
        """
        head_size = self.heads
        if tensor.ndim == 3:
            batch_size, seq_len, dim = tensor.shape
            extra_dim = 1
        else:
            batch_size, extra_dim, seq_len, dim = tensor.shape
        tensor = tensor.reshape(batch_size, seq_len * extra_dim, head_size, dim // head_size)
        tensor = tensor.permute(0, 2, 1, 3)

        if out_dim == 3:
            tensor = tensor.reshape(batch_size * head_size, seq_len * extra_dim, dim // head_size)

        return tensor

    def get_attention_scores(self, query: ms.Tensor, key: ms.Tensor, attention_mask: ms.Tensor = None) -> ms.Tensor:
        r"""
        Compute the attention scores.

        Args:
            query (`ms.Tensor`): The query tensor.
            key (`ms.Tensor`): The key tensor.
            attention_mask (`ms.Tensor`, *optional*): The attention mask to use. If `None`, no mask is applied.

        Returns:
            `ms.Tensor`: The attention probabilities/scores.
        """
        dtype = query.dtype
        if self.upcast_attention:
            query = query.float()
            key = key.float()

        if attention_mask is None:
            attention_scores = self.scale * ops.bmm(
                query,
                key.swapaxes(-1, -2),
            )
        else:
            attention_scores = ops.baddbmm(
                attention_mask,
                query,
                key.swapaxes(-1, -2),
                beta=1,
                alpha=self.scale,
            )

        if self.upcast_softmax:
            attention_scores = attention_scores.float()

        attention_probs = ops.softmax(attention_scores, axis=-1)

        attention_probs = attention_probs.to(dtype)

        return attention_probs

    def prepare_attention_mask(
        self, attention_mask: ms.Tensor, target_length: int, batch_size: int, out_dim: int = 3
    ) -> Optional[ms.Tensor]:
        r"""
        Prepare the attention mask for the attention computation.

        Args:
            attention_mask (`ms.Tensor`):
                The attention mask to prepare.
            target_length (`int`):
                The target length of the attention mask. This is the length of the attention mask after padding.
            batch_size (`int`):
                The batch size, which is used to repeat the attention mask.
            out_dim (`int`, *optional*, defaults to `3`):
                The output dimension of the attention mask. Can be either `3` or `4`.

        Returns:
            `ms.Tensor`: The prepared attention mask.
        """
        head_size = self.heads
        if attention_mask is None:
            return attention_mask

        current_length = attention_mask.shape[-1]
        if current_length != target_length:
            # TODO: for pipelines such as stable-diffusion, padding cross-attn mask:
            #       we want to instead pad by (0, remaining_length), where remaining_length is:
            #       remaining_length: int = target_length - current_length
            # TODO: re-enable tests/models/test_models_unet_2d_condition.py#test_model_xattn_padding
            attention_mask = ops.pad(attention_mask, (0, target_length), value=0.0)

        if out_dim == 3:
            if attention_mask.shape[0] < batch_size * head_size:
                attention_mask = attention_mask.repeat_interleave(head_size, dim=0)
        elif out_dim == 4:
            attention_mask = attention_mask.unsqueeze(1)
            attention_mask = attention_mask.repeat_interleave(head_size, dim=1)

        return attention_mask

    def norm_encoder_hidden_states(self, encoder_hidden_states: ms.Tensor) -> ms.Tensor:
        r"""
        Normalize the encoder hidden states. Requires `self.norm_cross` to be specified when constructing the
        `Attention` class.

        Args:
            encoder_hidden_states (`ms.Tensor`): Hidden states of the encoder.

        Returns:
            `ms.Tensor`: The normalized encoder hidden states.
        """
        assert self.norm_cross is not None, "self.norm_cross must be defined to call self.norm_encoder_hidden_states"

        if self.cross_attention_norm == "layer_norm":
            encoder_hidden_states = self.norm_cross(encoder_hidden_states)
        elif self.cross_attention_norm == "group_norm":
            # Group norm norms along the channels dimension and expects
            # input to be in the shape of (N, C, *). In this case, we want
            # to norm along the hidden dimension, so we need to move
            # (batch_size, sequence_length, hidden_size) ->
            # (batch_size, hidden_size, sequence_length)
            encoder_hidden_states = encoder_hidden_states.swapaxes(1, 2)
            encoder_hidden_states = self.norm_cross(encoder_hidden_states)
            encoder_hidden_states = encoder_hidden_states.swapaxes(1, 2)
        else:
            assert False

        return encoder_hidden_states

    def fuse_projections(self, fuse=True):
        dtype = self.to_q.weight.dtype

        if not self.is_cross_attention:
            # fetch weight matrices.
            concatenated_weights = ops.cat([self.to_q.weight, self.to_k.weight, self.to_v.weight])
            in_features = concatenated_weights.shape[1]
            out_features = concatenated_weights.shape[0]

            # create a new single projection layer and copy over the weights.
            self.to_qkv = self.linear_cls(in_features, out_features, has_bias=self.use_bias, dtype=dtype)
            self.to_qkv.weight.set_data(concatenated_weights)
            if self.use_bias:
                concatenated_bias = ops.cat([self.to_q.bias, self.to_k.bias, self.to_v.bias])
                self.to_qkv.bias.set_data(concatenated_bias)

        else:
            concatenated_weights = ops.cat([self.to_k.weight, self.to_v.weight])
            in_features = concatenated_weights.shape[1]
            out_features = concatenated_weights.shape[0]

            self.to_kv = self.linear_cls(in_features, out_features, has_bias=self.use_bias, dtype=dtype)
            self.to_kv.weight.set_data(concatenated_weights)
            if self.use_bias:
                concatenated_bias = ops.cat([self.to_k.bias, self.to_v.bias])
                self.to_kv.bias.set_data(concatenated_bias)

        self.fused_projections = fuse


@ms.jit_class
class AttnProcessor:
    r"""
    Default processor for performing attention-related computations.
    """

    def __call__(
        self,
        attn: Attention,
        hidden_states: ms.Tensor,
        encoder_hidden_states: Optional[ms.Tensor] = None,
        attention_mask: Optional[ms.Tensor] = None,
        temb: Optional[ms.Tensor] = None,
    ) -> ms.Tensor:
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).swapaxes(1, 2)
        else:
            batch_size, channel, height, width = None, None, None, None

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.swapaxes(1, 2)).swapaxes(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = ops.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.swapaxes(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


@ms.jit_class
class XFormersAttnProcessor:
    r"""
    Processor for implementing memory efficient attention using xFormers-like interface.

    Args:
        attention_op (`Callable`, *optional*, defaults to `None`):
            The base
            [operator](https://facebookresearch.github.io/xformers/components/ops.html#xformers.ops.AttentionOpBase) to
            use as the attention operator. It is recommended to set to `None`, and allow xFormers to choose the best
            operator.
    """

    def __init__(self, attention_op: Optional[Callable] = None):
        assert attention_op is None, (
            "Memory efficient attention on mindspore uses flash attention under the hoods. "
            "There is no other implementation for now. Please do not set `attention_op`."
        )
        self.attention_op = attention_op

    def __call__(
        self,
        attn: Attention,
        hidden_states: ms.Tensor,
        encoder_hidden_states: Optional[ms.Tensor] = None,
        attention_mask: Optional[ms.Tensor] = None,
        temb: Optional[ms.Tensor] = None,
    ) -> ms.Tensor:
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).swapaxes(1, 2)
        else:
            batch_size, channel, height, width = None, None, None, None

        batch_size, key_tokens, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        attention_mask = attn.prepare_attention_mask(attention_mask, key_tokens, batch_size)
        if attention_mask is not None:
            # expand our mask's singleton query_tokens dimension:
            #   [batch*heads,            1, key_tokens] ->
            #   [batch*heads, query_tokens, key_tokens]
            # so that it can be added as a bias onto the attention scores that xformers computes:
            #   [batch*heads, query_tokens, key_tokens]
            # we do this explicitly because xformers doesn't broadcast the singleton dimension for us.
            _, query_tokens, _ = hidden_states.shape
            attention_mask = attention_mask.tile((1, query_tokens, 1))

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.swapaxes(1, 2)).swapaxes(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        # Memory efficient attention on mindspore uses flash attention under the hoods.
        # Flash attention implementation is called `FlashAttentionScore`
        # which is an experimental api with the following limitations:
        # 1. Sequence length of query must be divisible by 16 and in range of [1, 32768].
        # 2. Head dimensions must be one of [64, 80, 96, 120, 128, 256].
        # 3. The input dtype must be float16 or bfloat16.
        # Sequence length of query must be checked in runtime.
        _, query_tokens, _ = query.shape
        assert query_tokens % 16 == 0, f"Sequence length of query must be divisible by 16, but got {query_tokens=}."
        # Head dimension is checked in Attention.set_use_memory_efficient_attention_xformers. We maybe pad on head_dim.
        if attn.head_dim_padding > 0:
            query_padded = ops.pad(query, (0, attn.head_dim_padding), mode="constant", value=0.0)
            key_padded = ops.pad(key, (0, attn.head_dim_padding), mode="constant", value=0.0)
            value_padded = ops.pad(value, (0, attn.head_dim_padding), mode="constant", value=0.0)
        else:
            query_padded, key_padded, value_padded = query, key, value
        flash_attn = ops.operations.nn_ops.FlashAttentionScore(1, scale_value=attn.scale)
        hidden_states_padded = flash_attn(query_padded, key_padded, value_padded, None, None, None, attention_mask)[3]
        # If we did padding before calculate attention, undo it!
        if attn.head_dim_padding > 0:
            hidden_states = hidden_states_padded[..., : attn.head_dim]
        else:
            hidden_states = hidden_states_padded

        hidden_states = hidden_states.to(query.dtype)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.swapaxes(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


CROSS_ATTENTION_PROCESSORS = (
    AttnProcessor,
    XFormersAttnProcessor,
)

AttentionProcessor = Union[
    AttnProcessor,
    XFormersAttnProcessor,
]