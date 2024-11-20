from functools import partial
from typing import Any
from typing import Tuple, List
from typing import Optional, TypeVar, Union

import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
import equinox as eqx
from models.dymn.utils import make_divisible, cnn_out_size
import inspect
from jax import numpy as jnp
import jax 
from jaxtyping import Array, PRNGKeyArray

class DynamicInvertedResidualConfig:
    def __init__(
            self,
            input_channels: int,
            kernel: int,
            expanded_channels: int,
            out_channels: int,
            use_dy_block: bool,
            activation: str,
            stride: int,
            dilation: int,
            width_mult: float,
    ):
        self.save_hyperparameters()
        self.input_channels = self.adjust_channels(input_channels, width_mult)
        self.kernel = kernel
        self.expanded_channels = self.adjust_channels(expanded_channels, width_mult)
        self.out_channels = self.adjust_channels(out_channels, width_mult)
        self.use_dy_block = use_dy_block
        self.use_hs = activation == "HS"
        self.use_se = False
        self.stride = stride
        self.dilation = dilation
        self.width_mult = width_mult

    @staticmethod
    def adjust_channels(channels: int, width_mult: float):
        return make_divisible(channels * width_mult, 8)

    def out_size(self, in_size):
        padding = (self.kernel - 1) // 2 * self.dilation
        return cnn_out_size(in_size, padding, self.dilation, self.kernel, self.stride)
        
        
class DynamicConv(eqx.Module):
    in_channels: int = eqx.field(static=True)
    out_channels: int = eqx.field(static=True)
    context_dim: int = eqx.field(static=True)
    kernel_size: int = eqx.field(static=True)
    stride: int = eqx.field(static=True)
    dilation: int = eqx.field(static=True)
    padding: int = eqx.field(static=True)
    groups: int = eqx.field(static=True)
    att_groups: int = eqx.field(static=True)
    k: int = eqx.field(static=True)
    temp_schedule: Tuple[int, int, int, float] = eqx.field(static=True)
    
    weight: Array
    bias: Optional[Array]
    residuals: eqx.nn.Sequential
    
    def __init__(self,
                 key,
                 in_channels,
                 out_channels,
                 context_dim,
                 kernel_size,
                 stride=1,
                 dilation=1,
                 padding=0,
                 groups=1,
                 att_groups=1,
                 bias=False,
                 k=4,
                 temp_schedule=(30, 1, 1, 0.05),
                 
                 ):
        super(DynamicConv, self).__init__()
        assert in_channels % groups == 0
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.k = k
        self.T_max, self.T_min, self.T0_slope, self.T1_slope = temp_schedule
        self.temperature = self.T_max
        
        self.att_groups = att_groups
        
        key, key_residual, key_w, key_b, key_random = jax.random.split(key, 5)
        
        self.residuals = eqx.nn.Sequential([
                eqx.nn.Linear(context_dim, k * self.att_groups, key=key_residual)
        ]
        )
        
        # k sets of weights for convolution
        weight = jax.random.normal(key_w, (out_channels, in_channels // groups, kernel_size, kernel_size))

        if bias:
            self.bias = jnp.zeros(k, out_channels)
        else:
            self.bias = None

        self._initialize_weights(weight, self.bias, key=key_random)

        weight = weight.view(1, k, att_groups, out_channels,
                             in_channels // groups, kernel_size, kernel_size)

        weight = weight.transpose(1, 2).view(1, self.att_groups, self.k, -1)
        self.weight = weight
        
    
    def _initialize_weights(self, weight, bias, key):
        init_func = partial(jax.nn.initializers.variance_scaling, factor=2.0, mode='fan_out', distribution='truncated_normal')
        for i in range(self.k):
            key, subkey, subkey_b = jax.random.split(key, 3)
            weight = init_func()(weight[i], key=subkey)
            if bias is not None:
                bias = jax.nn.initializers.zeros()(bias[i], key=subkey_b)
        return weight, bias
    

