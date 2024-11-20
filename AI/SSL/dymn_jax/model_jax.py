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
from snoring.config import args
# from flax.linen import initializers
# from flax.core import meta
# from flax.linen import initializers
# from flax.linen.dtypes import promote_dtype
# from flax.linen.module import Module, compact
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

from snoring.dymn_jax.dy_block_jax import MyConv, DynamicInvertedResidualConfig, \
    DY_Block, DynamicConv, DyReLUB, DynamicWrapper, Wrapper, \
        my_conv_general_dilated, show_shape

# from models.mn.block_types import InvertedResidualConfig, InvertedResidual

class Sequential(nn.Module):
  layers: None 

  def setup(self):
    self.name_layer_norm_drop = [i.name for i in self.layers 
                             if type(i) in [nn.BatchNorm, nn.Dropout]]
    self.name_DY_Block = [i.name for i in self.layers 
                             if type(i) in [DY_Block]]

  def __call__(self, x, train):
    for layer in self.layers:
        if layer.name in self.name_layer_norm_drop:
            x = layer(x, not train)
            # print("x.btc", x.dtype)
        elif layer.name in self.name_DY_Block:
            x = layer(x, train=train)
            # print("x.dtypeDĐ", x.dtype)
        else:
            x = layer(x)
            # print("x.dtype", x.dtype)
    return x


class ConvNormActivation:
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, ...]] = 3,
        stride: Union[int, Tuple[int, ...]] = 1,
        padding: Optional[Union[int, Tuple[int, ...], str]] = None,
        groups: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,# nn.BatchNorm,
        activation_layer: Optional[Callable[..., nn.Module]] = nn.relu,
        dilation: Union[int, Tuple[int, ...]] = 1,
        inplace: Optional[bool] = True,
        bias: Optional[bool] = None,
        conv_layer: Callable[..., nn.Module] = MyConv,
    ):

        if padding is None:
            if isinstance(kernel_size, int) and isinstance(dilation, int):
                padding = (kernel_size - 1) // 2 * dilation
            else:
                _conv_dim = len(kernel_size) if isinstance(kernel_size, Sequence) else len(dilation)
                kernel_size = self._make_ntuple(kernel_size, _conv_dim)
                dilation = self._make_ntuple(dilation, _conv_dim)
                padding = tuple((kernel_size[i] - 1) // 2 * dilation[i] for i in range(_conv_dim))
        if bias is None:
            bias = norm_layer is None

        layers = [
            conv_layer(
                out_channels,
                kernel_size,
                stride,
                padding,
                kernel_dilation=dilation,
                feature_group_count=groups,
                use_bias=bias,
                conv_general_dilated=my_conv_general_dilated
            )
        ]
        
        # print(layers)

        if norm_layer is not None:
            layers.append(norm_layer())

        if activation_layer is not None:
            params = {} if inplace is None else {"inplace": inplace}
            layers.append(Wrapper(activation_layer))
        
        self.layers = layers
    
    def __call__(self):
        return Sequential(self.layers)

    def _make_ntuple(self, x: Any, n: int) -> Tuple[Any, ...]:
        import collections
        from itertools import repeat
        if isinstance(x, collections.abc.Iterable):
            return tuple(x)
        return tuple(repeat(x, n))
    

class DyMN(nn.Module):
    inverted_residual_setting: List[DynamicInvertedResidualConfig]
    last_channel: int
    num_classes: int = 527
    head_type: str = "mlp"
    block: Optional[Callable[..., nn.Module]] = None
    # norm_layer: Optional[Callable[..., nn.Module]] = None
    dropout: float = 0.2
    in_conv_kernel: int = 3
    in_conv_stride: int = 2
    in_channels: int = 1
    context_ratio: int = 4
    max_context_size: int = 128
    min_context_size: int = 32
    dyrelu_k: int = 2
    dyconv_k: int = 4
    no_dyrelu: bool = False
    no_dyconv: bool = False
    no_ca: bool = False
    temp_schedule: tuple = (30, 1, 1, 0.05)
    
    def setup(self):
        if self.block is None:
            assert False
            block = DY_Block
        
        norm_layer = partial(nn.BatchNorm, epsilon=0.001, momentum=1-0.01, axis=1, param_dtype=args["input_dtype"])

        layers = []
        
        firstconv_output_channels = self.inverted_residual_setting[0].input_channels
        self.in_c = ConvNormActivation(
                self.in_channels,
                firstconv_output_channels,
                kernel_size=(self.in_conv_kernel, self.in_conv_kernel),
                stride=self.in_conv_stride,
                norm_layer=norm_layer,
                activation_layer=nn.activation.hard_swish,
        )()
        
        for cnf in self.inverted_residual_setting:
            if cnf.use_dy_block:
                b = self.block(cnf,
                    context_ratio=self.context_ratio,
                    max_context_size=self.max_context_size,
                    min_context_size=self.min_context_size,
                    dyrelu_k=self.dyrelu_k,
                    dyconv_k=self.dyconv_k,
                    no_dyrelu=self.no_dyrelu,
                    no_dyconv=self.no_dyconv,
                    no_ca=self.no_ca,
                    temp_schedule=self.temp_schedule,
                    )
                
            layers.append(b)
        # print(layers)
        self.layers = Sequential(layers)
                
        lastconv_input_channels = self.inverted_residual_setting[-1].out_channels
        lastconv_output_channels = 6 * lastconv_input_channels
        self.out_c = ConvNormActivation(
            lastconv_input_channels,
            lastconv_output_channels,
            kernel_size=(1, 1),
            norm_layer=norm_layer,
            activation_layer=nn.activation.hard_swish,
        )()
        
        self.classifier = Sequential([
                Wrapper(lambda x: jnp.mean(x, axis=(2, 3), keepdims=True)),
                Wrapper(lambda x: x.reshape((x.shape[0], -1))),
                nn.Dense(self.last_channel),
                Wrapper(lambda x: nn.activation.hard_swish(x)),
                nn.Dropout(self.dropout),
                nn.Dense(self.num_classes),
            ])
            
            
    def __call__(self, x, train: bool=True, debug: bool=False):
        # print("xdtype", x.dtype)
        
        # print()
        # print(x.shape)
        x = self.in_c(x, train)
        # print("xdtype", x.dtype)
        # print(x.shape)
        x = self.layers(x, train)
        # print("xdtype", x.dtype)
        x = self.out_c(x, train)
        # print("xdtype", x.dtype)
        x = self.classifier(x, train)
        # print("xdtype", x.dtype)
        return x
        

    def update_params(self, epoch):
        for module in self.modules():
            if isinstance(module, DynamicConv):
                module.update_params(epoch)


def _dymn_conf(
        width_mult: float = 1.0,
        reduced_tail: bool = False,
        dilated: bool = False,
        strides: Tuple[int] = (2, 2, 2, 2),
        use_dy_blocks: str = "all",
        **kwargs: Any
):
    reduce_divider = 2 if reduced_tail else 1
    dilation = 2 if dilated else 1

    bneck_conf = partial(DynamicInvertedResidualConfig, width_mult=width_mult)
    adjust_channels = partial(DynamicInvertedResidualConfig.adjust_channels, width_mult=width_mult)

    activations = ["RE", "RE", "RE", "RE", "RE", "RE", "HS", "HS", "HS", "HS", "HS", "HS", "HS", "HS", "HS"]

    if use_dy_blocks == "all":
        # per default the dynamic blocks replace all conventional IR blocks
        use_dy_block = [True] * 15
    elif use_dy_blocks == "replace_se":
        use_dy_block = [False, False, False, True, True, True, False, False, False, False, True, True, True, True, True]
    else:
        raise NotImplementedError(f"Config use_dy_blocks={use_dy_blocks} not implemented.")

    inverted_residual_setting = [
        bneck_conf(16, 3, 16, 16, use_dy_block[0], activations[0], 1, 1),
        bneck_conf(16, 3, 64, 24, use_dy_block[1], activations[1], strides[0], 1),  # C1
        bneck_conf(24, 3, 72, 24, use_dy_block[2], activations[2], 1, 1),
        bneck_conf(24, 5, 72, 40, use_dy_block[3], activations[3], strides[1], 1),  # C2
        bneck_conf(40, 5, 120, 40, use_dy_block[4], activations[4], 1, 1),
        bneck_conf(40, 5, 120, 40, use_dy_block[5], activations[5], 1, 1),
        bneck_conf(40, 3, 240, 80, use_dy_block[6], activations[6], strides[2], 1),  # C3
        bneck_conf(80, 3, 200, 80, use_dy_block[7], activations[7], 1, 1),
        bneck_conf(80, 3, 184, 80, use_dy_block[8], activations[8], 1, 1),
        bneck_conf(80, 3, 184, 80, use_dy_block[9], activations[9], 1, 1),
        bneck_conf(80, 3, 480, 112, use_dy_block[10], activations[10], 1, 1),
        bneck_conf(112, 3, 672, 112, use_dy_block[11], activations[11], 1, 1),
        bneck_conf(112, 5, 672, 160 // reduce_divider, use_dy_block[12], activations[12], strides[3], dilation),  # C4
        bneck_conf(160 // reduce_divider, 5, 960 // reduce_divider, 160 // reduce_divider, use_dy_block[13],
                   activations[13], 1, dilation),
        bneck_conf(160 // reduce_divider, 5, 960 // reduce_divider, 160 // reduce_divider, use_dy_block[14],
                   activations[14], 1, dilation),
    ]
    last_channel = adjust_channels(1280 // reduce_divider)

    return inverted_residual_setting, last_channel

def _dymn(
        inverted_residual_setting: List[DynamicInvertedResidualConfig],
        last_channel: int,
        pretrained_name: str,
        **kwargs: Any,
):
    def filter_kwargs(**kwargs):
    # Define the list of valid keyword argument name
    
        valid_kwargs = [
            "inverted_residual_setting",
            "last_channel",
            "num_classes",
            "head_type",
            "block",
            "norm_layer",
            "dropout",
            "in_conv_kernel",
            "in_conv_stride",
            "in_channels",
            "context_ratio",
            "max_context_size",
            "min_context_size",
            "dyrelu_k",
            "dyconv_k",
            "no_dyrelu",
            "no_dyconv",
            "no_ca",
            "temp_schedule",
        ]
        
        

        # Create a new dictionary that only contains the keys from valid_kwargs
        filtered_kwargs = {key: value for key, value in kwargs.items() if key in valid_kwargs}

        return filtered_kwargs

    clean_kwargs = filter_kwargs(**kwargs)

    model = DyMN(inverted_residual_setting, last_channel, **clean_kwargs)

    # load pre-trained model using specified name
    # if pretrained_name:
    #     # download from GitHub or load cached state_dict from 'resources' folder
    #     model_url = pretrained_models.get(pretrained_name)
    #     state_dict = load_state_dict_from_url(model_url, model_dir=model_dir, map_location="cpu")
    #     cls_in_state_dict = state_dict['classifier.5.weight'].shape[0]
    #     cls_in_current_model = model.classifier[5].out_features
    #     if cls_in_state_dict != cls_in_current_model:
    #         print(f"The number of classes in the loaded state dict (={cls_in_state_dict}) and "
    #               f"the current model (={cls_in_current_model}) is not the same. Dropping final fully-connected layer "
    #               f"and loading weights in non-strict mode!")
    #         del state_dict['classifier.5.weight']
    #         del state_dict['classifier.5.bias']
    #         model.load_state_dict(state_dict, strict=False)
    #     else:
    #         model.load_state_dict(state_dict)
    return model


def dymn(pretrained_name: str = None, **kwargs: Any):
    inverted_residual_setting, last_channel = _dymn_conf(**kwargs)
    return _dymn(inverted_residual_setting, last_channel, pretrained_name, **kwargs)


def get_model(num_classes: int = 527,
              pretrained_name: str = None,
              width_mult: float = 1.0,
              strides: Tuple[int, int, int, int] = (2, 2, 2, 2),
              # Context
              context_ratio: int = 4,
              max_context_size: int = 128,
              min_context_size: int = 32,
              # Dy-ReLU
              dyrelu_k: int = 2,
              no_dyrelu: bool = False,
              # Dy-Conv
              dyconv_k: int = 4,
              no_dyconv: bool = False,
              T_max: float = 30.0,
              T0_slope: float = 1.0,
              T1_slope: float = 0.02,
              T_min: float = 1,
              pretrain_final_temp: float = 1.0,
              # Coordinate Attention
              no_ca: bool = False,
              use_dy_blocks="all"):


    block = DY_Block
    if pretrained_name:
        # if model is pre-trained, set Dy-Conv temperature to 'pretrain_final_temp'
        # pretrained on ImageNet -> 30
        # pretrained on AudioSet -> 1
        T_max = pretrain_final_temp

    temp_schedule = (T_max, T_min, T0_slope, T1_slope)

    m = dymn(num_classes=num_classes,
             pretrained_name=pretrained_name,
             block=block,
             width_mult=width_mult,
             strides=strides,
             context_ratio=context_ratio,
             max_context_size=max_context_size,
             min_context_size=min_context_size,
             dyrelu_k=dyrelu_k,
             dyconv_k=dyconv_k,
             no_dyrelu=no_dyrelu,
             no_dyconv=no_dyconv,
             no_ca=no_ca,
             temp_schedule=temp_schedule,
             use_dy_blocks=use_dy_blocks
             )
    # print(m)
    return m


class BTHead(nn.Module):
    # dymn_model: nn.Module 

    def setup(self):
        self.dymn_model = get_model(width_mult=0.4)
        self.head = get_projection_head()
    
    def __call__(self, x, train=True):
        x = self.dymn_model(x, train)
        x = self.head(x, train)
        return x
    

def get_projection_head(dims=[527, 1024*2, 1024*5, 1024*5]):
    return Sequential([
        nn.Dense(dims[1]),
        nn.BatchNorm(epsilon=0.001, momentum=1-0.01, axis=1, param_dtype=args["input_dtype"]),
        jax.nn.relu,
        nn.Dense(dims[2]),
        nn.BatchNorm(epsilon=0.001, momentum=1-0.01, axis=1, param_dtype=args["input_dtype"]),
        jax.nn.relu,
        nn.Dense(dims[3])
    ]
    )


if __name__ == "__main__":
    # from dymn_jax.model_jax import get_model as get_model_j
    import jax  
    from jax import numpy as jnp

    model_j = get_model(width_mult=0.4)

    params = model_j.init(jax.random.PRNGKey(0), jnp.ones((2,3,128,625)), False)
    pprint(show_shape(params))
    # MyConv(features=16, kernel_size=(1, 1), strides=(1, 1), padding='VALID', use_bias=False, conv_general_dilated=my_conv_general_dilated)
    # MyConv(
    #     # attributes
    #     features = 16,
    #     kernel_size = (3,3),
    #     # strides = 2,
    #     # padding = 1,
    #     # input_dilation = 1,
    #     # kernel_dilation = 1,
    #     # feature_group_count = 1,
    #     use_bias = False,
    #     # mask = None,
    #     # dtype = None,
    #     # param_dtype = jnp.float32,
    #     # precision = None,
    #     # kernel_init = init,
    #     # bias_init = zeros,
    #     conv_general_dilated = my_conv_general_dilated,
    #     # conv_general_dilated_cls = None,
    # ).init(jax.random.PRNGKey(0), jnp.ones((2,3,128,128)))


# %%
