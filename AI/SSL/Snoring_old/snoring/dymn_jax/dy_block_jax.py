# %% 
import jax
from jax import numpy as jnp
from jax.nn import softmax, hard_swish, sigmoid
import flax.linen as nn
from functools import partial
from pprint import pprint
from jax import lax
from collections.abc import Sequence
from typing import NamedTuple, Union
from jax._src.typing import Array, DTypeLike
from flax.linen import initializers
from flax.core import meta
from flax.linen import initializers
from flax.linen.dtypes import promote_dtype
from flax.linen.module import Module, compact
from typing import (
  Any,
  Callable,
  Iterable,
  List,
  Optional,
  Sequence,
  Tuple,
  Union,
)
import numpy as np




PRNGKey = Any
Shape = Tuple[int, ...]
Dtype = Any
Array = Any
PrecisionLike = Union[None, str, lax.Precision, Tuple[str, str], Tuple[lax.Precision, lax.Precision]]
DotGeneralT = Callable[..., Array]
ConvGeneralDilatedT = Callable[..., Array]

default_kernel_init = initializers.lecun_normal()

show_shape = partial(jax.tree_map, lambda x: x.shape)

def _normalize_axes(axes: Tuple[int, ...], ndim: int) -> Tuple[int, ...]:
  # A tuple by convention. len(axes_tuple) then also gives the rank efficiently.
  return tuple(sorted(ax if ax >= 0 else ndim + ax for ax in axes))


def _canonicalize_tuple(x: Union[Sequence[int], int]) -> Tuple[int, ...]:
  if isinstance(x, Iterable):
    return tuple(x)
  else:
    return (x,)

def _conv_dimension_numbers(input_shape):
  """Computes the dimension numbers based on the input shape."""
  ndim = len(input_shape)
  lhs_spec = (0, ndim - 1) + tuple(range(1, ndim - 1))
  rhs_spec = (ndim - 1, ndim - 2) + tuple(range(0, ndim - 2))
  out_spec = lhs_spec
  return lax.ConvDimensionNumbers(lhs_spec, rhs_spec, out_spec)

PaddingLike = Union[str, int, Sequence[Union[int, Tuple[int, int]]]]
LaxPadding = Union[str, Sequence[Tuple[int, int]]]

def canonicalize_padding(padding: PaddingLike, rank: int) -> LaxPadding:
  """ "Canonicalizes conv padding to a jax.lax supported format."""
  if isinstance(padding, str):
    return padding
  if isinstance(padding, int):
    return [(padding, padding)] * rank
  if isinstance(padding, Sequence) and len(padding) == rank:
    new_pad = []
    for p in padding:
      if isinstance(p, int):
        new_pad.append((p, p))
      elif isinstance(p, tuple) and len(p) == 2:
        new_pad.append(p)
      else:
        break
    if len(new_pad) == rank:
      return new_pad
  raise ValueError(
    f'Invalid padding format: {padding}, should be str, int,'
    f' or a sequence of len {rank} where each element is an'
    ' int or pair of ints.'
  )
  


class MyConv(nn.Module):

  features: int
  kernel_size: Union[int, Sequence[int]]
  strides: Union[None, int, Sequence[int]] = 1
  padding: PaddingLike = 'VALID'
  input_dilation: Union[None, int, Sequence[int]] = 1
  kernel_dilation: Union[None, int, Sequence[int]] = 1
  feature_group_count: int = 1
  use_bias: bool = True
  mask: Optional[Array] = None
  dtype: Optional[Dtype] = None
  param_dtype: Dtype = jnp.float32
  precision: PrecisionLike = None
  kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = default_kernel_init
  bias_init: Callable[
    [PRNGKey, Shape, Dtype], Array
  ] = initializers.zeros_init()
  # Deprecated. Will be removed.
  conv_general_dilated: Optional[ConvGeneralDilatedT] = None
  conv_general_dilated_cls: Any = None

  @property
  def shared_weights(self) -> bool:
    return True

  @nn.compact
  def __call__(self, inputs: Array) -> Array:

    kernel_size: Sequence[int]
    if isinstance(self.kernel_size, int):
      kernel_size = (self.kernel_size,)
    else:
      kernel_size = tuple(self.kernel_size)

    def maybe_broadcast(
      x: Optional[Union[int, Sequence[int]]],
    ) -> Tuple[int, ...]:
      if x is None:
        # backward compatibility with using None as sentinel for
        # broadcast 1
        x = 1
      if isinstance(x, int):
        return (x,) * len(kernel_size)
      return tuple(x)

    # Combine all input batch dimensions into a single leading batch axis.
    num_batch_dimensions = inputs.ndim - (len(kernel_size) + 1)
    if num_batch_dimensions != 1:
      input_batch_shape = inputs.shape[:num_batch_dimensions]
      total_batch_size = int(np.prod(input_batch_shape))
      flat_input_shape = (total_batch_size,) + inputs.shape[
        num_batch_dimensions:
      ]
      inputs = jnp.reshape(inputs, flat_input_shape)

    # self.strides or (1,) * (inputs.ndim - 2)
    strides = maybe_broadcast(self.strides)
    input_dilation = maybe_broadcast(self.input_dilation)
    kernel_dilation = maybe_broadcast(self.kernel_dilation)

    padding_lax = canonicalize_padding(self.padding, len(kernel_size))
    if padding_lax == 'CIRCULAR':
      kernel_size_dilated = [
        (k - 1) * d + 1 for k, d in zip(kernel_size, kernel_dilation)
      ]
      zero_pad: List[Tuple[int, int]] = [(0, 0)]
      pads = (
        zero_pad
        + [((k - 1) // 2, k // 2) for k in kernel_size_dilated]
        + [(0, 0)]
      )
      inputs = jnp.pad(inputs, pads, mode='wrap')
      padding_lax = 'VALID'
    elif padding_lax == 'CAUSAL':
      if len(kernel_size) != 1:
        raise ValueError(
          'Causal padding is only implemented for 1D convolutions.'
        )
      left_pad = kernel_dilation[0] * (kernel_size[0] - 1)
      pads = [(0, 0), (left_pad, 0), (0, 0)]
      inputs = jnp.pad(inputs, pads)
      padding_lax = 'VALID'

    dimension_numbers = _conv_dimension_numbers(inputs.shape)
    in_features = jnp.shape(inputs)[1]
    if self.shared_weights:
      # One shared convolutional kernel for all pixels in the output.
        assert in_features % self.feature_group_count == 0
    #   kernel_shape = kernel_size + (
    #     in_features // self.feature_group_count,
    #     self.features,
    #   )
        kernel_shape =  (
            self.features,
            in_features // self.feature_group_count
        ) + kernel_size


    if self.mask is not None and self.mask.shape != kernel_shape:
      raise ValueError(
        'Mask needs to have the same shape as weights. '
        f'Shapes are: {self.mask.shape}, {kernel_shape}'
      )

    kernel = self.param(
      'kernel', self.kernel_init, kernel_shape, self.param_dtype
    )

    if self.mask is not None:
      kernel *= self.mask

    if self.use_bias:
      if self.shared_weights:
        # One bias weight per output channel, shared between pixels.
        bias_shape = (self.features,)

      bias = self.param('bias', self.bias_init, bias_shape, self.param_dtype)
    else:
      bias = None

    inputs, kernel, bias = promote_dtype(inputs, kernel, bias, dtype=self.dtype)
    if self.shared_weights:
      if self.conv_general_dilated_cls is not None:
        conv_general_dilated = self.conv_general_dilated_cls()
      elif self.conv_general_dilated is not None:
        conv_general_dilated = self.conv_general_dilated
      else:
        conv_general_dilated = lax.conv_general_dilated
      y = conv_general_dilated(
        inputs,
        kernel,
        strides,
        padding_lax,
        lhs_dilation=input_dilation,
        rhs_dilation=kernel_dilation,
        dimension_numbers=dimension_numbers,
        feature_group_count=self.feature_group_count,
        precision=self.precision,
      )

    if self.use_bias:
    #   print((1,) * (y.ndim - bias.ndim) + bias.shape)
      bias = bias.reshape((1,) + bias.shape + (1,) * (y.ndim - bias.ndim - 1))
      y += bias

    if num_batch_dimensions != 1:
      output_shape = input_batch_shape + y.shape[1:]
      y = jnp.reshape(y, output_shape)
    return y

def my_conv_general_dilated(
  lhs, rhs: Array, window_strides: Sequence[int],
  padding: str | Sequence[tuple[int, int]],
  lhs_dilation: Sequence[int] | None = None,
  rhs_dilation: Sequence[int] | None = None,
  dimension_numbers = None,
  *args, **kwargs
  ):
    
    return jax.lax.conv_general_dilated(
        lhs,
        rhs,
        window_strides,
        padding,
        lhs_dilation=lhs_dilation,
        rhs_dilation=rhs_dilation,
        dimension_numbers=('NCHW', 'OIHW', 'NCHW'),
        *args, **kwargs
    )
 
def pool(inputs, init, reduce_fn, window_shape, strides, padding):

    num_batch_dims = inputs.ndim - (len(window_shape) + 1)
    strides = strides or (1,) * len(window_shape)
    assert len(window_shape) == len(
        strides
    ), f'len({window_shape}) must equal len({strides})'
    strides = (1,) * num_batch_dims + (1,) + strides
    dims = (1,) * num_batch_dims + (1,) + window_shape
  

    is_single_input = False
    if num_batch_dims == 0:
        # add singleton batch dimension because lax.reduce_window always
        # needs a batch dimension.
        inputs = inputs[None]
        strides = (1,) + strides
        dims = (1,) + dims
        is_single_input = True

    assert inputs.ndim == len(dims), f'len({inputs.shape}) != len({dims})'
    if not isinstance(padding, str):
        padding = tuple(map(tuple, padding))
        assert len(padding) == len(window_shape), (
        f'padding {padding} must specify pads for same number of dims as '
        f'window_shape {window_shape}'
        )
        assert all(
        [len(x) == 2 for x in padding]
        ), f'each entry in padding {padding} must be length 2'
        padding = ((0, 0),) + ((0, 0),) + padding

    y = lax.reduce_window(inputs, init, reduce_fn, dims, strides, padding)
    if is_single_input:
        y = jnp.squeeze(y, axis=0)
    return y
        
def avg_pool(
    inputs, window_shape, strides=None, padding='VALID', count_include_pad=True
    ):
    # print(inputs.shape)
    y = pool(inputs, 0.0, lax.add, window_shape, strides, padding)
    if count_include_pad:
        y = y / np.prod(window_shape)
    return y
  
  
    
class DynamicConv(nn.Module):
    in_channels: int
    out_channels: int
    context_dim: int
    kernel_size: int
    stride: int = 1
    dilation: int = 1
    padding: int = 0
    groups: int = 1
    att_groups: int = 1
    use_bias: bool = False
    k: int = 4
    temp_schedule: tuple = (30, 1, 1, 0.05)

    T_max, T_min, T0_slope, T1_slope = temp_schedule
    T_max, T_min, T0_slope, T1_slope = map(jnp.array, (T_max, T_min, T0_slope, T1_slope))
    # temperature = T_max

    @staticmethod
    def _initialize_weights(key, shape, dtype=jnp.float32):
        init_func = nn.initializers.kaiming_normal()
        return init_func(key, shape, dtype)

    def setup(self):
        
        self.residuals = nn.Sequential([
            nn.Dense(self.k * self.att_groups, kernel_init=self._initialize_weights)
        ])
        
        self.temperature = self.variable("immutable", "temperature", jnp.array, self.T_max).value

        weight_shape = (self.k, self.out_channels, self.in_channels // self.groups, self.kernel_size, self.kernel_size)
        temp = jnp.ones(weight_shape)
        temp = temp.swapaxes(1, 2).reshape(1, self.att_groups, self.k, -1)
        self.weight = self.param('weight', self._initialize_weights, temp.shape)
        # print(self.weight.swapaxes(1, 2).shape)
        # self.weight = self.weight.swapaxes(1, 2).reshape(1, self.att_groups, self.k, -1)
        
        if self.use_bias:
            self.bias = self.param('bias', nn.initializers.zeros, (self.k, self.out_channels))
        else:
            self.bias = None

    def __call__(self, x, g=None):
        b, c, f, t = x.shape
        # print(x.shape)
        g_c = g[0].reshape((b, -1))
        residuals = self.residuals(g_c).reshape((b, self.att_groups, 1, -1))
        attention = jax.nn.softmax(residuals / self.temperature, axis=-1)

        aggregate_weight = (attention @ self.weight).swapaxes(1, 2).reshape((b, self.out_channels,
                                                                                 self.in_channels // self.groups,
                                                                                 self.kernel_size, self.kernel_size))

        aggregate_weight = aggregate_weight.reshape((b * self.out_channels, self.in_channels // self.groups,
                                                     self.kernel_size, self.kernel_size))

        x = x.reshape((1, -1, f, t))
        
        output = jax.lax.conv_general_dilated(x, rhs=aggregate_weight, window_strides=(self.stride, self.stride),
                                  padding=[(self.padding, self.padding), (self.padding, self.padding)],
                                  dimension_numbers=('NCHW', 'OIHW', 'NCHW'), feature_group_count=b*self.groups)
        
        if self.bias is not None:
            aggregate_bias = jnp.matmul(attention, self.bias).reshape(-1)
            output = output + aggregate_bias

        output = output.reshape((b, self.out_channels, output.shape[-2], output.shape[-1]))
        return output

    # def update_params(self, epoch):
    #     t0 = self.T_max - self.T0_slope * epoch
    #     t1 = 1 + self.T1_slope * (self.T_max - 1) / self.T0_slope - self.T1_slope * epoch
    #     self.temperature = max(t0, t1, self.T_min)
    #     print(f"Setting temperature for attention over kernels to {self.temperature}")


class DyReLU(nn.Module):
    channels: int
    context_dim: int
    M: int = 2

    def setup(self):
        # Define parameters with buffers
        # self.coef_net = nn.Dense(features=2 * self.M)
        self.lambdas = jnp.array([1.] * self.M + [0.5] * self.M, dtype=jnp.float32)
        self.init_v = jnp.array([1.] + [0.] * (2 * self.M - 1), dtype=jnp.float32)

    def get_relu_coefs(self, x):
        theta = self.coef_net(x)
        sigmoid = lambda x: 1 / (1 + jnp.exp(-x))  # Sigmoid function
        theta = 2 * sigmoid(theta) - 1
        return theta

    def __call__(self, x, g):
        raise NotImplementedError("DyReLU should not be called directly, please use DyReLUB instead.")


class DyReLUB(DyReLU):
    channels: int
    context_dim: int
    M: int = 2

    def setup(self):
        super().setup()
        # Redefine the linear layer for coef_net
        self.coef_net = nn.Sequential([
            nn.Dense(features=2 * self.M * self.channels)
        ])

    def __call__(self, x, g):
        assert x.shape[1] == self.channels
        assert g is not None
        b, c, f, t = x.shape
        h_c = g[0].reshape((b, -1))

        theta = self.get_relu_coefs(h_c)

        relu_coefs = theta.reshape((-1, self.channels, 1, 1, 2 * self.M)) * self.lambdas + self.init_v
        x_mapped = x[..., jnp.newaxis] * relu_coefs[..., :self.M] + relu_coefs[..., self.M:]

        if self.M == 2:
            result = jnp.maximum(x_mapped[:, :, :, :, 0], x_mapped[:, :, :, :, 1])
        else:
            result = jnp.max(x_mapped, axis=-1)

        return result

class CoordAtt(nn.Module):
    @nn.compact
    def __call__(self, x, g):
        g_cf, g_ct = g[1], g[2]
        a_f = nn.sigmoid(g_cf)
        a_t = nn.sigmoid(g_ct)
        # Recalibration with channel-frequency and channel-time weights
        out = x * a_f * a_t
        return out


class Identity(nn.Module):
    @nn.compact
    def __call__(self, x):
        return x
    
class ContextGen(nn.Module):
    context_dim: int
    in_ch: int
    exp_ch: int
    norm_layer: nn.Module  # You would define this based on the expected norm layer or pass it as a module callable.
    stride: int = 1
    trainning: bool = True

    def setup(self):
        # You need to instantiate your submodules (layers) here in `setup` when not using `nn.compact`
        self.joint_conv = MyConv(features=self.context_dim, kernel_size=(1, 1), strides=(1, 1), padding='VALID', use_bias=False, conv_general_dilated=my_conv_general_dilated)
        self.conv_f = MyConv(features=self.exp_ch, kernel_size=(1, 1), strides=(1, 1), padding='VALID', conv_general_dilated=my_conv_general_dilated)
        self.conv_t = MyConv(features=self.exp_ch, kernel_size=(1, 1), strides=(1, 1), padding='VALID', conv_general_dilated=my_conv_general_dilated)
        self.joint_norm = self.norm_layer()
        # We no longer declare pooling here. Depending on the stride, apply it in the __call__ method.

        if self.stride > 1:
            # sequence pooling for Coordinate Attention
            
            self.pool_f = lambda x: avg_pool(x, window_shape=(3, 1), strides=(self.stride, 1), padding=((1,1), (0,0)))
            self.pool_t = lambda x: avg_pool(x, window_shape=(1, 3), strides=(1, self.stride), padding=((0,0), (1,1)))
        else:
            self.pool_f = Identity()
            self.pool_t = Identity()
            
    def __call__(self, x, g, train):
        cf = jnp.mean(x, axis=3, keepdims=True)
        ct = jnp.mean(x, axis=2, keepdims=True).transpose((0, 1, 3, 2))

        # if self.stride > 1:
        #     cf = nn.avg_pool(cf, kernel_size=(3, 1), strides=(self.stride, 1), padding='SAME')
        #     ct = nn.avg_pool(ct, kernel_size=(1, 3), strides=(1, self.stride), padding='SAME')
        
        g_cat = jnp.concatenate([cf, ct], axis=2)
        g_cat = self.joint_norm(self.joint_conv(g_cat), not train)
        g_cat = hard_swish(g_cat)

        f, t = cf.shape[2], ct.shape[2]
        h_cf, h_ct = jnp.split(g_cat, indices_or_sections=[f], axis=2)
        h_ct = jnp.transpose(h_ct, (0, 1, 3, 2))
        h_c = jnp.mean(g_cat, axis=2, keepdims=True)

        g_cf = self.conv_f(self.pool_f(h_cf))
        g_ct = self.conv_t(self.pool_t(h_ct))

        return (h_c, g_cf, g_ct)
    
# if __name__ == "__main__":
#     conv = ContextGen(10,3, 20, norm_layer = partial(nn.BatchNorm, False, momentum=0.01, axis=1))
#     params = conv.init(jax.random.PRNGKey(0), jnp.ones([2,3,10,10]), None)

#     from pprint import pprint
#     pprint(show_shape(params))


import math
from typing import Optional


def make_divisible(v: float, divisor: int, min_value: Optional[int] = None) -> int:
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def cnn_out_size(in_size, padding, dilation, kernel, stride):
    s = in_size + 2 * padding - dilation * (kernel - 1) - 1
    return math.floor(s / stride + 1)

class DynamicWrapper(nn.Module):
    module: nn.Module
    def __call__(self, x, g):
        return self.module(x)

class Wrapper(nn.Module):
    module: nn.Module
    def __call__(self, x):
        return self.module(x)

class WrapperModule(nn.Module):
    module: nn.Module
    def __call__(self, *args):
        return self.module(*args)

    
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
    

class DY_Block(nn.Module):
    # Define attributes here, such as:
    cnf: DynamicInvertedResidualConfig
    context_ratio: int = 4
    max_context_size: int = 128
    min_context_size: int = 32
    temp_schedule: tuple = (30, 1, 1, 0.05)
    dyrelu_k: int = 2
    dyconv_k: int = 4
    no_dyrelu: bool = False
    no_dyconv: bool = False
    no_ca: bool = False
    # training: bool = False

    def setup(self):
        cnf = self.cnf
        if not (1 <= cnf.stride <= 2):
            raise ValueError("illegal stride value")

        self.use_res_connect = cnf.stride == 1 and cnf.input_channels == cnf.out_channels
        # Here, self.context_dim will be a static value unlike dynamic value in PyTorch
        self.context_dim = jnp.clip(
            make_divisible(self.cnf.expanded_channels // self.context_ratio, 8),
            make_divisible(self.min_context_size * self.cnf.width_mult, 8),
            make_divisible(self.max_context_size * self.cnf.width_mult, 8)
        )
        # print(self.context_dim)
        
        activation_layer =nn.activation.hard_swish if self.cnf.use_hs else nn.relu

        norm_layer = partial(nn.BatchNorm, momentum=1-0.01, epsilon=0.001, axis=1)

        # Expand
        # print(self.cnf.expanded_channels, self.cnf.input_channels)
        if self.cnf.expanded_channels != self.cnf.input_channels:
            if self.no_dyconv:
                self.exp_conv = DynamicWrapper(
                    MyConv(
                        self.cnf.expanded_channels,
                        kernel_size=(1, 1),
                        strides=1,
                        padding=0,
                        # kernel_init=nn.initializers.xavier_uniform(),
                        bias=False
                    )
                )
            else:
                self.exp_conv = DynamicConv(
                    cnf.input_channels,
                    cnf.expanded_channels,
                    self.context_dim,
                    kernel_size=1,
                    k=self.dyconv_k,
                    temp_schedule=self.temp_schedule,
                    stride=1,
                    dilation=1,
                    padding=0,
                    use_bias=False
                )

            self.exp_norm = norm_layer()
            self.exp_act = DynamicWrapper(activation_layer)
        else:
            self.exp_conv = DynamicWrapper(lambda x: x)
            self.exp_norm = WrapperModule(lambda x, train: x)
            self.exp_act = DynamicWrapper(lambda x: x)

        # Depthwise
        stride = 1 if self.cnf.dilation > 1 else self.cnf.stride
        padding = (self.cnf.kernel - 1) // 2 * self.cnf.dilation
        # print(cnf.expanded_channels, self.context_dim)
        if self.no_dyconv:
            # print("OK")
            self.depth_conv = DynamicWrapper(
                MyConv(
                    self.cnf.expanded_channels,
                    kernel_size=(self.cnf.kernel, self.cnf.kernel),
                    stride=stride,
                    dilation=(self.cnf.dilation, self.cnf.dilation),
                    padding=padding,
                    groups=self.cnf.expanded_channels,
                    # kernel_init=nn.initializers.xavier_uniform(),
                    bias=False
                )
            )
        else:
            self.depth_conv = DynamicConv(
                cnf.expanded_channels,
                cnf.expanded_channels,
                self.context_dim,
                kernel_size=cnf.kernel,
                k=self.dyconv_k,
                temp_schedule=self.temp_schedule,
                groups=cnf.expanded_channels,
                stride=stride,
                dilation=cnf.dilation,
                padding=padding,
                use_bias=False
            )


        self.depth_norm = norm_layer()
        self.depth_act = DynamicWrapper(activation_layer) if self.no_dyrelu \
            else DyReLUB(self.cnf.expanded_channels, self.context_dim, M=self.dyrelu_k)

        self.ca = DynamicWrapper(nn.Identity()) if self.no_ca else CoordAtt()

        # Project
        if self.no_dyconv:
            self.proj_conv = DynamicWrapper(
                MyConv(
                    self.cnf.out_channels,
                    kernel_size=(1, 1),
                    strides=(1, 1),
                    padding=0,
                    # kernel_init=nn.initializers.xavier_uniform(),
                    bias=False
                )
            )
        else:
            self.proj_conv = DynamicConv(
                cnf.expanded_channels,
                cnf.out_channels,
                self.context_dim,
                kernel_size=1,
                k=self.dyconv_k,
                temp_schedule=self.temp_schedule,
                stride=1,
                dilation=1,
                padding=0,
                use_bias=False,
            )
            

        self.proj_norm = norm_layer()
        self.context_gen = ContextGen(self.context_dim, self.cnf.input_channels, self.cnf.expanded_channels,
                                      norm_layer=norm_layer, stride=stride)
        

        # Continue setting up other layers like depthwise conv, activations, and so on
        # ...

    def __call__(self, x, g=None, train=False):
        # x: CNN feature map (C x F x T)
        inp = x

        g = self.context_gen(x, g, train)
        x = self.exp_conv(x, g)
        x = self.exp_norm(x, not train)
        x = self.exp_act(x, g)
        x = self.depth_conv(x, g)
        x = self.depth_norm(x, not train)
        x = self.depth_act(x, g)
        x = self.ca(x, g)

        x = self.proj_conv(x, g)
        x = self.proj_norm(x, not train)

        if self.use_res_connect:
            x += inp
        return x

    # Here, you would define ContextGen and any other necessary submodules.
    

if __name__ == "__main__":
# if False:
    batch_size = 1
    input_channels = 8
    feature_size = 64
    time_frames = 64
    
    dy_block = DY_Block(
        cnf=DynamicInvertedResidualConfig(
            activation='HS', 
            dilation=1, 
            expanded_channels=16, 
            input_channels=16, 
            kernel=3, 
            out_channels=16, 
            stride=2, 
            use_dy_block=True, 
            width_mult=0.4
        ),
        context_ratio=4, 
        dyconv_k=4, 
        dyrelu_k=2,  
        max_context_size=128, 
        min_context_size=32, 
        no_ca=False, 
        no_dyconv=False, 
        no_dyrelu=False, 
        temp_schedule=(1.0, 1, 1.0, 0.02)

    )
    
    key = jax.random.PRNGKey(0)
    params = dy_block.init(key, jnp.ones((batch_size, input_channels, feature_size, time_frames)))

    # uniform(key, shape=(1000,))
    input_data = 1-jax.random.uniform(jax.random.PRNGKey(758493), shape=((batch_size, input_channels, feature_size, time_frames)))
    output = dy_block.apply(params, input_data, mutable=['batch_stats'])

    # pprint(show_shape(output))

def parse_to_tree(ordered_dict):
    tree = {}
    for key, value in ordered_dict.items():
        keys = key.split('.')
        current_node = tree
        for k in keys[:-1]:
            current_node = current_node.setdefault(k, {})
        current_node[keys[-1]] = value
    return tree

def transfer_torch2jax_llayer(params, state):
    # print(show_shape(state))
    keys_up = params.keys()
    for key in keys_up:
        #conv
        if key == "kernel":
            key_ = "weight"
        #batchnorm
        elif key == "scale":
            key_ = "weight"
        elif key == "mean":
            key_ = "running_mean"
        elif key == "var":
            key_ = "running_var"
        # sequence
        elif key[:len("layers_")]=="layers_":
            key_ = key[len("layers_"):]
        
        else:
            key_ = key
        # new = params[key]
        # new_= state[key_]
        if not isinstance(params[key], dict):
            # pprint(show_shape(params))
            # pprint(show_shape(state))
            print(key, key_)
            new_value = state[key_].detach().cpu().numpy()
            if len(new_value.shape) == 2:
                new_value = jnp.transpose(new_value, (1, 0))
            params[key] = new_value
        else:
            print(key, key_)
            transfer_torch2jax_llayer(params[key], state[key_])
            
# if __name__ == "__main__":
if False:
    def _get_dymn():
        from models.dymn.model import get_model as get_dymn
        model = get_dymn(pretrained_name="dymn04_as", width_mult=0.4, num_classes=10)
        return model
    
    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output
        return hook
    
    model = _get_dymn()
    model.eval()
    model.layers[0].proj_norm.register_forward_hook(get_activation('fc3'))
    state_dict_py = parse_to_tree(model.state_dict())["layers"]["0"]
    # pprint(model.layers[0].hparams)
    # pprint(show_shape(state_dict_py))
    # pprint(show_shape(params))
    # pprint(show_shape(state_dict_py))
    transfer_torch2jax_llayer(params['params'], state_dict_py)
    transfer_torch2jax_llayer(params['batch_stats'], state_dict_py)
    
    import torch
    numpy_array = jax.device_get(input_data)
    torch_tensor = torch.from_numpy(numpy_array)
    outpy = model.layers[0](torch_tensor)
    
    # outpy = activation['fc3']
    # print(outpy)
    
    # a = params["params"]["context_gen"]["joint_conv"]["kernel"]
    # b = model.layers[0].context_gen.state_dict()["joint_conv.weight"]
    outjax = dy_block.apply(params, input_data)
    
    print(outpy.shape, outjax.shape, outjax.dtype)
    import matplotlib.pyplot as plt  
    
    t_out = outpy.detach().cpu().numpy()

    plt.hist(outpy.detach().numpy().reshape(-1))
    plt.show()
    # plt.hist(a.reshape(-1))
    # plt.show()
    
    plt.hist(outjax.reshape(-1))
    plt.show()
    
    print(model.classifier)
    
    import numpy as np
    np.testing.assert_almost_equal(outjax, t_out, decimal=2)
    
    
    
    


    

# %%
