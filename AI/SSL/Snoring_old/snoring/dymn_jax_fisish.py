
# %%
from dymn_jax.model_jax import get_model as get_model_j, show_shape, pprint
from dymn_jax.dy_block_jax import parse_to_tree, transfer_torch2jax_llayer
import jax  
from jax import numpy as jnp

# model_j = get_model_j(width_mult=0.4)

# model_j.init(jax.random.PRNGKey(0), jnp.ones((2,3,128,128)))

import jax  
from jax import numpy as jnp

model_j = get_model_j(pretrained_name="dymn04_as", width_mult=0.4)
input_data = jnp.ones((2,1,128,128))
params = model_j.init(jax.random.PRNGKey(0), input_data)

def _get_dymn():
    from models.dymn.model import get_model as get_dymn
    model = get_dymn(pretrained_name="dymn04_as", width_mult=0.4)
    return model

model = _get_dymn()

state_dict_py = parse_to_tree(model.state_dict())

from flax.core.frozen_dict import unfreeze
params = unfreeze(params)
params['params'] = jax.tree_map(lambda x:jnp.array(0), params['params'])

transfer_torch2jax_llayer(params['params'], state_dict_py)
transfer_torch2jax_llayer(params['batch_stats'], state_dict_py)

def check_zero(x):
    assert (x!=0).all(), print(x)
    return x
jax.tree_map(check_zero, params['params'])

    
# pprint(show_shape(params))

import torch
numpy_array = jax.device_get(input_data)
torch_tensor = torch.from_numpy(numpy_array)
outpy = model(torch_tensor)[0]

# outpy = activation['fc3']
# print(outpy)

# a = params["params"]["context_gen"]["joint_conv"]["kernel"]
# b = model.layers[0].context_gen.state_dict()["joint_conv.weight"]
outjax = model_j.apply(params, input_data)

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
np.testing.assert_almost_equal(outjax, t_out, decimal=1)



# %%
# from flax.training.common_utils import shard
# from flax.training import train_state
from flax.jax_utils import replicate